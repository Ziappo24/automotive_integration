"""
pipeline_a.py
=============
Pipeline A – Traditional Deterministic Baseline.

Implements the four sequential phases of a classical data integration pipeline:
  1. Schema Alignment   – Jaro-Winkler string similarity
  2. Blocking           – Multi-attribute blocking key (Make + Year)
  3. Record Linkage     – Pairwise similarity + Connected Components clustering
  4. Data Fusion        – Majority Voting with source-priority tiebreaking

All decisions are fully deterministic; no ML or LLM calls are made here.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from jellyfish import jaro_winkler_similarity

from config import (
    LINKAGE_MATCH_THRESHOLD,
    MEDIATED_SCHEMA,
    MILEAGE_TOLERANCE,
    PRICE_TOLERANCE,
    SCHEMA_FALLBACK_MAP,
    SCHEMA_JARO_THRESHOLD,
    SIMILARITY_WEIGHTS,
    SOURCE_PRIORITY,
    YEAR_TOLERANCE,
    setup_logging,
)

logger = logging.getLogger("integration.pipeline_a")


# ===========================================================================
# Phase 1 – Schema Alignment
# ===========================================================================

def align_schema(
    source_columns: List[str],
    source_name: str,
) -> Dict[str, Optional[str]]:
    """
    Map source column names to the mediated schema using Jaro-Winkler distance.

    For each source column the function:
      1. Computes Jaro-Winkler similarity against every mediated attribute.
      2. Accepts the best match if its score ≥ SCHEMA_JARO_THRESHOLD.
      3. Falls back to the hand-crafted SCHEMA_FALLBACK_MAP if no auto-match.
      4. Returns None for columns that cannot be mapped.

    Parameters
    ----------
    source_columns : list[str]
        Raw column names from the source DataFrame.
    source_name : str
        Identifier of the source ('craigslist', 'usedcars', 'ebay').

    Returns
    -------
    dict
        {source_column: mediated_attribute | None}
    """
    mapping: Dict[str, Optional[str]] = {}

    for col in source_columns:
        col_lower = col.lower().strip()
        best_score = 0.0
        best_match: Optional[str] = None

        for attr in MEDIATED_SCHEMA:
            score = jaro_winkler_similarity(col_lower, attr)
            if score > best_score:
                best_score = score
                best_match = attr

        if best_score >= SCHEMA_JARO_THRESHOLD:
            mapping[col] = best_match
            logger.debug(
                "[%s] '%s' → '%s'  (JW=%.3f)", source_name, col, best_match, best_score
            )
        else:
            # Fallback to static dictionary
            fallback = SCHEMA_FALLBACK_MAP.get(col_lower)
            mapping[col] = fallback
            if fallback:
                logger.debug(
                    "[%s] '%s' → '%s'  (fallback dict)", source_name, col, fallback
                )
            else:
                logger.debug("[%s] '%s' → None  (unmapped)", source_name, col)

    return mapping


# ===========================================================================
# Phase 2 – Blocking
# ===========================================================================

def build_candidate_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Generate candidate record pairs using a blocking key of Make + Year.

    Only records that share the same blocking key are compared pairwise,
    drastically reducing the O(n²) Cartesian product.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame from all three sources; must have columns
        'record_id', 'source', and 'blocking_key'.

    Returns
    -------
    list of (record_id_left, record_id_right)
        All cross-source candidate pairs within each block.
    """
    logger.info("Building candidate pairs via blocking key (make + year) …")
    t0 = time.perf_counter()

    blocks: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        blocks[row["blocking_key"]].append(row["record_id"])

    pairs: List[Tuple[str, str]] = []
    id_to_source = df.set_index("record_id")["source"].to_dict()

    for key, ids in blocks.items():
        if len(ids) < 2:
            continue
        # Enumerate all cross-source pairs within the block
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                # Only compare records from different sources
                if id_to_source[id_a] != id_to_source[id_b]:
                    pairs.append((id_a, id_b))

    elapsed = time.perf_counter() - t0
    logger.info(
        "Blocking complete: %d candidate pairs in %.2f s", len(pairs), elapsed
    )
    return pairs


# ===========================================================================
# Phase 3 – Pairwise Record Linkage
# ===========================================================================

def _jaccard_text(a: Optional[str], b: Optional[str]) -> float:
    """Jaccard similarity between token sets of two strings."""
    if not a or not b or pd.isna(a) or pd.isna(b):
        return 0.0
    set_a = set(str(a).lower().split())
    set_b = set(str(b).lower().split())
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _numeric_similarity(a, b, tolerance: float) -> float:
    """
    Linear similarity for numeric values with a maximum tolerance window.
    Returns 1.0 when values are identical, 0.0 when |a - b| >= tolerance.
    """
    try:
        fa, fb = float(a), float(b)
        if np.isnan(fa) or np.isnan(fb):
            return 0.0
        diff = abs(fa - fb)
        return max(0.0, 1.0 - diff / tolerance)
    except (TypeError, ValueError):
        return 0.0


def compute_similarity_score(
    rec_a: pd.Series,
    rec_b: pd.Series,
) -> float:
    """
    Compute a weighted combined similarity score for a pair of records.

    Attributes and their similarity functions:
      - make, model     → Jaro-Winkler (text)
      - year            → linear numeric with ±YEAR_TOLERANCE
      - price           → linear numeric with ±PRICE_TOLERANCE
      - mileage         → linear numeric with ±MILEAGE_TOLERANCE
      - fuel_type       → exact match (0 or 1)

    Parameters
    ----------
    rec_a, rec_b : pd.Series
        Rows from the combined DataFrame.

    Returns
    -------
    float
        Weighted similarity in [0, 1].
    """
    scores: Dict[str, float] = {}

    # Text attributes
    for attr in ("make", "model"):
        val_a = rec_a.get(attr)
        val_b = rec_b.get(attr)
        if pd.isna(val_a) or pd.isna(val_b):
            scores[attr] = 0.0
        else:
            scores[attr] = jaro_winkler_similarity(str(val_a), str(val_b))

    # Numeric attributes
    scores["year"] = _numeric_similarity(
        rec_a.get("year"), rec_b.get("year"), YEAR_TOLERANCE * 2 + 1
    )
    scores["price"] = _numeric_similarity(
        rec_a.get("price"), rec_b.get("price"), PRICE_TOLERANCE
    )
    scores["mileage"] = _numeric_similarity(
        rec_a.get("mileage"), rec_b.get("mileage"), MILEAGE_TOLERANCE
    )

    # Fuel type: exact match
    fa = str(rec_a.get("fuel_type", "")).lower()
    fb = str(rec_b.get("fuel_type", "")).lower()
    scores["fuel_type"] = 1.0 if (fa == fb and fa not in ("nan", "")) else 0.0

    # Weighted sum
    total = sum(
        SIMILARITY_WEIGHTS[attr] * scores[attr]
        for attr in SIMILARITY_WEIGHTS
        if attr in scores
    )
    return round(total, 4)


def link_records(
    df: pd.DataFrame,
    candidate_pairs: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str, float]], List[Set[str]]]:
    """
    Compute pairwise similarity for all candidate pairs and cluster matches
    using Connected Components (NetworkX).

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset indexed by record_id.
    candidate_pairs : list of (id_left, id_right)

    Returns
    -------
    matches : list of (id_left, id_right, score)
        Pairs classified as matches (score ≥ LINKAGE_MATCH_THRESHOLD).
    clusters : list of set
        Each set is a cluster of record_ids representing one real entity.
    """
    logger.info("Computing pairwise similarities for %d pairs …", len(candidate_pairs))
    t0 = time.perf_counter()

    indexed = df.set_index("record_id")
    matches: List[Tuple[str, str, float]] = []

    for id_a, id_b in candidate_pairs:
        rec_a = indexed.loc[id_a]
        rec_b = indexed.loc[id_b]
        score = compute_similarity_score(rec_a, rec_b)
        if score >= LINKAGE_MATCH_THRESHOLD:
            matches.append((id_a, id_b, score))

    logger.info(
        "Linkage: %d matches from %d pairs (%.2f s)",
        len(matches), len(candidate_pairs), time.perf_counter() - t0,
    )

    # Cluster via Connected Components
    G = nx.Graph()
    G.add_nodes_from(df["record_id"])
    G.add_edges_from([(a, b) for a, b, _ in matches])
    clusters = [comp for comp in nx.connected_components(G) if len(comp) > 1]
    logger.info("Found %d multi-record clusters", len(clusters))

    return matches, clusters


# ===========================================================================
# Phase 4 – Data Fusion
# ===========================================================================

def fuse_cluster(
    cluster: Set[str],
    indexed_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Resolve an entity cluster to a single fused record using Majority Voting.

    For each mediated attribute:
      1. Collect all non-null values from records in the cluster.
      2. Pick the most frequent value (majority vote).
      3. On a tie, prefer the value from the highest-priority source.

    Parameters
    ----------
    cluster : set of record_ids
    indexed_df : pd.DataFrame indexed by record_id

    Returns
    -------
    dict
        Fused record with all mediated attributes plus provenance metadata.
    """
    records = indexed_df.loc[list(cluster)]
    fused: Dict[str, object] = {"record_ids": list(cluster)}

    for attr in MEDIATED_SCHEMA:
        if attr not in records.columns:
            fused[attr] = None
            continue

        values = records[attr].dropna()
        if values.empty:
            fused[attr] = None
            continue

        # Count occurrences
        counts = values.value_counts()
        max_count = counts.iloc[0]
        candidates = counts[counts == max_count].index.tolist()

        if len(candidates) == 1:
            fused[attr] = candidates[0]
        else:
            # Tiebreak by source priority
            source_map = records["source"].to_dict()
            best_val = candidates[0]
            best_priority = 999
            for rid, row in records.iterrows():
                val = row.get(attr)
                if val in candidates:
                    priority = SOURCE_PRIORITY.get(row["source"], 999)
                    if priority < best_priority:
                        best_priority = priority
                        best_val = val
            fused[attr] = best_val

    fused["entity_source_count"] = len(records["source"].unique())
    fused["entity_record_count"] = len(cluster)
    return fused


def run_data_fusion(
    df: pd.DataFrame,
    clusters: List[Set[str]],
) -> pd.DataFrame:
    """
    Run data fusion over all detected clusters.

    Singleton records (no cross-source match found) are included as-is.

    Parameters
    ----------
    df : pd.DataFrame
    clusters : list of set

    Returns
    -------
    pd.DataFrame
        One row per integrated entity.
    """
    logger.info("Running data fusion over %d clusters …", len(clusters))
    indexed = df.set_index("record_id")
    clustered_ids: Set[str] = set().union(*clusters) if clusters else set()

    fused_records = []
    for cluster in clusters:
        fused_records.append(fuse_cluster(cluster, indexed))

    # Add singleton records unchanged
    singletons = df[~df["record_id"].isin(clustered_ids)]
    for _, row in singletons.iterrows():
        rec = row[MEDIATED_SCHEMA].to_dict()
        rec["record_ids"] = [row["record_id"]]
        rec["entity_source_count"] = 1
        rec["entity_record_count"] = 1
        fused_records.append(rec)

    result = pd.DataFrame(fused_records)
    logger.info("Data fusion complete: %d integrated entities", len(result))
    return result


# ===========================================================================
# Full Pipeline A Runner
# ===========================================================================

def run_pipeline_a(
    df: pd.DataFrame,
    source_raw_columns: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Execute the full Pipeline A (Traditional Baseline) end-to-end.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-loaded and normalised combined dataset.
    source_raw_columns : dict, optional
        {source_name: [raw_column_names]} for schema alignment evaluation.
        If None, schema alignment is skipped.

    Returns
    -------
    integrated : pd.DataFrame
        Final integrated dataset.
    artefacts : dict
        Intermediate results for evaluation and Pipeline B reuse:
        {
          'schema_mappings': {source: mapping_dict},
          'candidate_pairs': list of (id_a, id_b),
          'matches': list of (id_a, id_b, score),
          'clusters': list of sets,
        }
    """
    logger.info("=" * 60)
    logger.info("PIPELINE A – TRADITIONAL BASELINE")
    logger.info("=" * 60)

    artefacts: Dict = {}

    # Phase 1 – Schema Alignment
    schema_mappings: Dict[str, Dict] = {}
    if source_raw_columns:
        for src, cols in source_raw_columns.items():
            schema_mappings[src] = align_schema(cols, src)
    artefacts["schema_mappings"] = schema_mappings

    # Phase 2 – Blocking
    candidate_pairs = build_candidate_pairs(df)
    artefacts["candidate_pairs"] = candidate_pairs

    # Phase 3 – Record Linkage
    matches, clusters = link_records(df, candidate_pairs)
    artefacts["matches"] = matches
    artefacts["clusters"] = clusters

    # Phase 4 – Data Fusion
    integrated = run_data_fusion(df, clusters)

    logger.info("Pipeline A finished: %d integrated entities", len(integrated))
    return integrated, artefacts
