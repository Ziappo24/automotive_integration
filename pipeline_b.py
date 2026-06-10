"""
pipeline_b.py
=============
Pipeline B – LLM-Assisted Hybrid Integration Pipeline.

Architecture overview:
  ┌──────────────────────────────────────────────────────┐
  │  Phase 1: Schema Alignment (LLM-assisted)            │
  │    - LLM maps source columns to mediated schema      │
  │    - Fallback: Pipeline A's Jaro-Winkler mapping     │
  ├──────────────────────────────────────────────────────┤
  │  Phase 2: Hybrid Record Linkage                      │
  │    - Blocking identical to Pipeline A                │
  │    - Score < 0.30  → automatic NON-MATCH             │
  │    - Score > 0.85  → automatic MATCH                 │
  │    - 0.30 ≤ score ≤ 0.85 → LLM call (borderline)    │
  │    - LLM failure → fallback to Pipeline A threshold  │
  ├──────────────────────────────────────────────────────┤
  │  Phase 3: Data Fusion (LLM-assisted on conflicts)    │
  │    - Non-conflicting attributes: Majority Voting     │
  │    - Conflicting attributes: LLM semantic decision   │
  │    - LLM failure → Majority Voting fallback          │
  └──────────────────────────────────────────────────────┘

LLM calls are logged, cached, and never manually corrected.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from config import (
    LINKAGE_MATCH_THRESHOLD,
    LINKAGE_NONMATCH_THRESHOLD,
    MEDIATED_SCHEMA,
    SOURCE_PRIORITY,
    setup_logging,
)
from llm_client import call_llm
from pipeline_a import (
    align_schema,
    build_candidate_pairs,
    compute_similarity_score,
    fuse_cluster,
)

logger = logging.getLogger("integration.pipeline_b")

# ---------------------------------------------------------------------------
# Failure statistics (collected for evaluation report)
# ---------------------------------------------------------------------------
_stats: Dict[str, int] = {
    "schema_llm_calls": 0,
    "schema_llm_failures": 0,
    "linkage_llm_calls": 0,
    "linkage_llm_failures": 0,
    "linkage_auto_match": 0,
    "linkage_auto_nonmatch": 0,
    "fusion_llm_calls": 0,
    "fusion_llm_failures": 0,
}


def get_stats() -> Dict[str, int]:
    """Return a copy of the LLM call statistics."""
    return dict(_stats)


# ===========================================================================
# Phase 1 – Schema Alignment (LLM-Assisted)
# ===========================================================================

_SCHEMA_SYSTEM_PROMPT = """You are a data engineering assistant.
Your task is to map source database column names to a canonical mediated schema.

The mediated schema has EXACTLY these six attributes (no others exist):
  - make       (vehicle brand / manufacturer, e.g. Toyota, Ford, BMW)
  - model      (vehicle model name, e.g. Camry, Mustang, 3 Series)
  - year       (model year, a 4-digit integer, e.g. 2018)
  - price      (asking / listing price in currency units)
  - mileage    (odometer reading in miles or kilometres)
  - fuel_type  (fuel category, e.g. gas, diesel, electric)

CRITICAL RULES – violating any rule makes the output unusable:
  1. Return ONLY a raw JSON object. No markdown fences, no preamble, no explanation.
  2. Each KEY must be an exact source column name from the list you are given.
  3. Each VALUE must be one of the six mediated attributes above – EXACTLY as spelled:
         "make", "model", "year", "price", "mileage", "fuel_type"
     Any other value (e.g. "manufacturer", "brand", "odometer", "km") is FORBIDDEN.
  4. Only include a column if it clearly maps to one of the six attributes.
     Omit columns that do not map; do NOT include them with null or wrong values.

Common aliases you MUST recognise:
  - make      ← manufacturer, brand, make_name, franchise_make, marque
  - model     ← model_name, vehicle_model
  - year      ← yearOfRegistration, year_of_registration, model_year, vehicle_year
  - price     ← listing_price, asking_price, sale_price
  - mileage   ← odometer, kilometer, km, miles, vehicle_mileage
  - fuel_type ← fuel, fuelType, fuel_type_display, fueltype

Example output:
{
  "manufacturer": "make",
  "model_name": "model",
  "odometer": "mileage",
  "price": "price",
  "yearOfRegistration": "year"
}"""


def align_schema_llm(
    source_columns: List[str],
    sample_rows: List[Dict[str, Any]],
    source_name: str,
    fallback_mapping: Dict[str, Optional[str]],
) -> Dict[str, Optional[str]]:
    """
    Use the local LLM to map source columns to the mediated schema.

    The LLM receives:
      - The list of source column names.
      - Up to 3 sample rows for context.

    On failure (invalid JSON, missing keys, hallucinated attributes), the
    function falls back to Pipeline A's deterministic mapping.

    Parameters
    ----------
    source_columns : list[str]
        Raw column names from the source.
    sample_rows : list[dict]
        Up to 3 sample rows (dicts) from the source for context.
    source_name : str
        Source identifier.
    fallback_mapping : dict
        Pipeline A's deterministic mapping for this source.

    Returns
    -------
    dict
        {source_column: mediated_attribute | None}
    """
    global _stats
    _stats["schema_llm_calls"] += 1

    user_message = (
        f"Source: {source_name}\n"
        f"Columns: {json.dumps(source_columns)}\n"
        f"Sample rows (up to 3):\n{json.dumps(sample_rows[:3], default=str, indent=2)}\n\n"
        f"Map each column to the mediated schema. Return strict JSON."
    )

    result = call_llm(
        system_prompt=_SCHEMA_SYSTEM_PROMPT,
        user_message=user_message,
        required_keys=None,
        call_tag=f"schema_alignment_{source_name}",
    )

    if result is None:
        _stats["schema_llm_failures"] += 1
        logger.warning(
            "[Schema/%s] LLM failed. Using Pipeline A fallback mapping.", source_name
        )
        return fallback_mapping

    # Validate values: must be a valid mediated attribute
    # Pre-fill with the Pipeline A fallback so any LLM miss is recovered deterministically.
    cleaned: Dict[str, Optional[str]] = {
        col: fallback_mapping.get(col) for col in source_columns
    }
    has_hallucination = False

    for col, mapped_val in result.items():
        if col not in source_columns:
            continue  # Ignore hallucinated keys (key not in actual source columns)
        if mapped_val is not None and mapped_val not in MEDIATED_SCHEMA:
            logger.warning(
                "[Schema/%s] LLM hallucinated attribute '%s' for column '%s'. "
                "Using Pipeline A fallback '%s' instead.",
                source_name, mapped_val, col, fallback_mapping.get(col),
            )
            has_hallucination = True
            # Keep the deterministic fallback already in cleaned[col]
        else:
            # Accept the LLM mapping (overrides the fallback for this column)
            cleaned[col] = mapped_val

    if has_hallucination:
        _stats["schema_llm_failures"] += 1

    logger.info("[Schema/%s] LLM mapping: %s", source_name, cleaned)
    return cleaned


# ===========================================================================
# Phase 2 – Hybrid Record Linkage (LLM on Borderline Cases)
# ===========================================================================

_LINKAGE_SYSTEM_PROMPT = """You are an automotive data matching expert.
Given two vehicle records, decide whether they refer to the SAME real-world vehicle.

IMPORTANT RULES:
  - Return ONLY a valid JSON object with exactly these three keys:
      "match": boolean (true if same vehicle, false otherwise),
      "confidence": float between 0.0 and 1.0,
      "explanation": string (one sentence explaining the decision).
  - No preamble, no markdown, no extra keys.
  - Consider that the same vehicle can appear with minor textual variants across
    different marketplaces (abbreviations, typos, extra spaces, unit differences).

Example:
{"match": true, "confidence": 0.88, "explanation": "Both records describe a 2018 Ford Mustang GT with matching mileage within normal listing variance."}"""


def _serialise_record(rec: pd.Series) -> str:
    """Serialise a record to 'COL <name> VAL <value>' format (Ditto-style)."""
    parts = []
    for attr in MEDIATED_SCHEMA:
        val = rec.get(attr)
        if pd.notna(val) and str(val) not in ("nan", "None", ""):
            parts.append(f"COL {attr} VAL {val}")
    return " ".join(parts)


def link_records_hybrid(
    df: pd.DataFrame,
    candidate_pairs: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str, float, str]], List[Set[str]]]:
    """
    Hybrid record linkage: automatic decisions for clear cases,
    LLM invocation only for borderline pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset.
    candidate_pairs : list of (id_left, id_right)

    Returns
    -------
    matches : list of (id_a, id_b, score, decision_source)
        decision_source ∈ {'auto_high', 'llm', 'fallback_a'}
    clusters : list of set
    """
    global _stats
    logger.info(
        "Hybrid linkage: processing %d candidate pairs …", len(candidate_pairs)
    )
    t0 = time.perf_counter()

    indexed = df.set_index("record_id")
    matches: List[Tuple[str, str, float, str]] = []

    for id_a, id_b in candidate_pairs:
        rec_a = indexed.loc[id_a]
        rec_b = indexed.loc[id_b]
        score = compute_similarity_score(rec_a, rec_b)

        # --- Automatic decisions ---
        if score < LINKAGE_NONMATCH_THRESHOLD:
            _stats["linkage_auto_nonmatch"] += 1
            continue  # Definite non-match

        if score > 0.85:
            _stats["linkage_auto_match"] += 1
            matches.append((id_a, id_b, score, "auto_high"))
            continue

        # --- Borderline: invoke LLM ---
        _stats["linkage_llm_calls"] += 1
        user_message = (
            f"Record A: {_serialise_record(rec_a)}\n"
            f"Record B: {_serialise_record(rec_b)}\n"
            f"Syntactic similarity score: {score:.3f}\n\n"
            f"Are these the same vehicle? Return strict JSON."
        )

        llm_result = call_llm(
            system_prompt=_LINKAGE_SYSTEM_PROMPT,
            user_message=user_message,
            required_keys=["match", "confidence", "explanation"],
            call_tag="record_linkage_borderline",
        )

        if llm_result is None:
            _stats["linkage_llm_failures"] += 1
            # Fallback: use Pipeline A threshold
            if score >= LINKAGE_MATCH_THRESHOLD:
                matches.append((id_a, id_b, score, "fallback_a"))
            continue

        is_match = llm_result.get("match", False)
        confidence = float(llm_result.get("confidence", 0.0))

        if is_match:
            matches.append((id_a, id_b, confidence, "llm"))

    logger.info(
        "Hybrid linkage done: %d matches in %.2f s "
        "(auto=%d, llm=%d, fallback=%d, llm_fail=%d)",
        len(matches),
        time.perf_counter() - t0,
        _stats["linkage_auto_match"],
        _stats["linkage_llm_calls"] - _stats["linkage_llm_failures"],
        sum(1 for *_, src in matches if src == "fallback_a"),
        _stats["linkage_llm_failures"],
    )

    # Cluster via Connected Components
    G = nx.Graph()
    G.add_nodes_from(df["record_id"])
    G.add_edges_from([(a, b) for a, b, *_ in matches])
    clusters = [comp for comp in nx.connected_components(G) if len(comp) > 1]
    logger.info("Found %d multi-record clusters", len(clusters))

    return matches, clusters


# ===========================================================================
# Phase 3 – Data Fusion (LLM on Conflicting Attributes)
# ===========================================================================

_FUSION_SYSTEM_PROMPT = """You are a data quality expert specialising in automotive data.
You are given conflicting values for a single attribute from multiple sources.
Select the most plausible canonical value.

IMPORTANT RULES:
  - Return ONLY a valid JSON object with exactly these three keys:
      "selected_value": string (the winning canonical value),
      "confidence": float between 0.0 and 1.0,
      "evidence": string (one sentence explaining why this value was chosen).
  - No preamble, no markdown, no extra keys.
  - If you cannot determine the correct value, output the most common one and set confidence < 0.5."""


def _detect_conflicts(
    cluster: Set[str],
    indexed_df: pd.DataFrame,
) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Identify attributes with conflicting values across records in a cluster.

    Returns
    -------
    dict
        {attribute: [(source, value), ...]} for attributes with > 1 distinct value.
    """
    records = indexed_df.loc[list(cluster)]
    conflicts: Dict[str, List[Tuple[str, Any]]] = {}

    for attr in MEDIATED_SCHEMA:
        if attr not in records.columns:
            continue
        values = records[[attr, "source"]].dropna(subset=[attr])
        unique_vals = values[attr].unique()
        if len(unique_vals) > 1:
            conflicts[attr] = list(zip(values["source"], values[attr]))

    return conflicts


def fuse_cluster_llm(
    cluster: Set[str],
    indexed_df: pd.DataFrame,
    entity_context: str = "",
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Fuse a cluster, using the LLM to resolve conflicting attribute values.

    Parameters
    ----------
    cluster : set of record_ids
    indexed_df : pd.DataFrame indexed by record_id
    entity_context : str
        Brief textual description of the entity for LLM context.

    Returns
    -------
    fused_record : dict
    conflict_log : list of dicts (for error analysis)
    """
    global _stats
    records = indexed_df.loc[list(cluster)]
    fused: Dict[str, Any] = {"record_ids": list(cluster)}
    conflict_log: List[Dict] = []

    # Detect conflicts
    conflicts = _detect_conflicts(cluster, indexed_df)

    for attr in MEDIATED_SCHEMA:
        if attr not in records.columns:
            fused[attr] = None
            continue

        values = records[attr].dropna()
        if values.empty:
            fused[attr] = None
            continue

        if attr not in conflicts:
            # No conflict: simple majority vote
            fused[attr] = values.value_counts().index[0]
            continue

        # --- Conflicting attribute: invoke LLM ---
        _stats["fusion_llm_calls"] += 1
        source_value_pairs = conflicts[attr]
        options_str = "; ".join(
            f"[{src}] '{val}'" for src, val in source_value_pairs
        )

        user_message = (
            f"Attribute: {attr}\n"
            f"Entity context: {entity_context or 'automotive vehicle'}\n"
            f"Conflicting source values: {options_str}\n\n"
            f"Select the most plausible canonical value. Return strict JSON."
        )

        llm_result = call_llm(
            system_prompt=_FUSION_SYSTEM_PROMPT,
            user_message=user_message,
            required_keys=["selected_value", "confidence", "evidence"],
            call_tag=f"data_fusion_{attr}",
        )

        if llm_result is None:
            _stats["fusion_llm_failures"] += 1
            # Fallback: Majority Voting (Pipeline A logic)
            fallback_val = values.value_counts().index[0]
            fused[attr] = fallback_val
            conflict_log.append({
                "attribute": attr,
                "conflicts": source_value_pairs,
                "llm_decision": None,
                "fallback_used": True,
                "final_value": fallback_val,
            })
            logger.warning(
                "[Fusion] LLM failed for attr '%s'; fallback to '%s'", attr, fallback_val
            )
        else:
            selected = llm_result["selected_value"]
            fused[attr] = selected
            conflict_log.append({
                "attribute": attr,
                "conflicts": source_value_pairs,
                "llm_decision": llm_result,
                "fallback_used": False,
                "final_value": selected,
            })
            logger.debug(
                "[Fusion] '%s' resolved to '%s' (confidence=%.2f)",
                attr, selected, llm_result.get("confidence", 0.0),
            )

    fused["entity_source_count"] = len(records["source"].unique())
    fused["entity_record_count"] = len(cluster)
    return fused, conflict_log


def run_data_fusion_llm(
    df: pd.DataFrame,
    clusters: List[Set[str]],
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Run LLM-assisted data fusion over all detected clusters.

    Returns
    -------
    integrated : pd.DataFrame
    all_conflict_logs : list of dicts (for error analysis in evaluation.py)
    """
    logger.info("Running LLM-assisted data fusion over %d clusters …", len(clusters))
    indexed = df.set_index("record_id")
    clustered_ids: Set[str] = set().union(*clusters) if clusters else set()

    fused_records: List[Dict] = []
    all_conflict_logs: List[Dict] = []

    for cluster in clusters:
        # Build a brief entity context from majority-voted make/model/year
        recs = indexed.loc[list(cluster)]
        make = recs["make"].mode().iloc[0] if "make" in recs.columns else "?"
        model = recs["model"].mode().iloc[0] if "model" in recs.columns else "?"
        year = recs["year"].mode().iloc[0] if "year" in recs.columns else "?"
        context = f"{year} {make} {model}"

        fused, clog = fuse_cluster_llm(cluster, indexed, entity_context=context)
        fused_records.append(fused)
        all_conflict_logs.extend(clog)

    # Singletons pass through unchanged
    singletons = df[~df["record_id"].isin(clustered_ids)]
    for _, row in singletons.iterrows():
        rec = row[MEDIATED_SCHEMA].to_dict()
        rec["record_ids"] = [row["record_id"]]
        rec["entity_source_count"] = 1
        rec["entity_record_count"] = 1
        fused_records.append(rec)

    result = pd.DataFrame(fused_records)
    logger.info(
        "LLM-assisted fusion complete: %d entities (%d conflict resolutions attempted)",
        len(result), _stats["fusion_llm_calls"],
    )
    return result, all_conflict_logs


# ===========================================================================
# Full Pipeline B Runner
# ===========================================================================

def run_pipeline_b(
    df: pd.DataFrame,
    pipeline_a_artefacts: Dict,
    source_raw_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """
    Execute the full Pipeline B (LLM-Assisted) end-to-end.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-loaded combined dataset.
    pipeline_a_artefacts : dict
        Output from pipeline_a.run_pipeline_a(); reuses blocking results.
    source_raw_data : dict, optional
        {source_name: raw_DataFrame} used to send sample rows to the LLM
        during schema alignment. If None, schema alignment uses Pipeline A's result.

    Returns
    -------
    integrated : pd.DataFrame
    artefacts : dict
    conflict_logs : list of dicts
    """
    from typing import Optional  # local import to satisfy type checker

    logger.info("=" * 60)
    logger.info("PIPELINE B – LLM-ASSISTED HYBRID")
    logger.info("=" * 60)

    artefacts: Dict = {}

    # Phase 1 – Schema Alignment (LLM)
    schema_mappings: Dict[str, Dict] = {}
    if source_raw_data:
        for src_name, src_df in source_raw_data.items():
            cols = src_df.columns.tolist()
            samples = src_df.head(3).to_dict(orient="records")
            fallback = pipeline_a_artefacts.get("schema_mappings", {}).get(src_name, {})
            schema_mappings[src_name] = align_schema_llm(cols, samples, src_name, fallback)
    else:
        schema_mappings = pipeline_a_artefacts.get("schema_mappings", {})
    artefacts["schema_mappings"] = schema_mappings

    # Phase 2 – Blocking (reuse Pipeline A candidate pairs)
    candidate_pairs = pipeline_a_artefacts["candidate_pairs"]
    artefacts["candidate_pairs"] = candidate_pairs

    # Phase 2 – Hybrid Record Linkage
    matches, clusters = link_records_hybrid(df, candidate_pairs)
    artefacts["matches"] = matches
    artefacts["clusters"] = clusters

    # Phase 3 – LLM-Assisted Data Fusion
    integrated, conflict_logs = run_data_fusion_llm(df, clusters)

    artefacts["llm_stats"] = get_stats()
    logger.info("Pipeline B finished: %d integrated entities", len(integrated))
    logger.info("LLM call statistics: %s", get_stats())

    return integrated, artefacts, conflict_logs
