"""
data_loader.py
==============
Utilities for loading, sampling, and pre-processing the three automotive
sources (Craigslist, US Used Cars, eBay Motors) into a unified format
ready for the integration pipelines.

All functions return DataFrames with a guaranteed 'source' column and a
unique 'record_id' of the form  <source>_<row_index>.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BRAND_SYNONYMS,
    CRAIGSLIST_FILE,
    DATA_DIR,
    EBAY_FILE,
    MEDIATED_SCHEMA,
    USEDCARS_FILE,
    setup_logging,
)

logger = logging.getLogger("integration.data_loader")

# ---------------------------------------------------------------------------
# Column aliases per source (raw name → mediated schema name)
# Applied before any further normalisation.
# ---------------------------------------------------------------------------
CRAIGSLIST_RENAME = {
    "manufacturer": "make",
    "odometer": "mileage",
    "fuel": "fuel_type",
}

USEDCARS_RENAME = {
    "make_name": "make",
    "model_name": "model",
    "milage": "mileage",
    "mileage": "mileage",
    "fuel_type_display": "fuel_type",
    "fuel_type": "fuel_type",
    "price": "price",
}

EBAY_RENAME = {
    "brand": "make",
    "vehicle_mileage": "mileage",
    "kilometer": "mileage",
    "fuel_type_display": "fuel_type",
    "fuelType": "fuel_type",
    "listing_price": "price",
    "price": "price",
    "vehicle_year": "year",
    "yearOfRegistration": "year",
}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip whitespace, replace NaN-like strings with pd.NA."""
    nan_like = {"nan", "none", "n/a", "na", "", "null", "unknown", "-"}
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .replace(nan_like, pd.NA)
    )


def _normalise_make(series: pd.Series) -> pd.Series:
    """Apply brand synonym dictionary to collapse variants."""
    cleaned = _normalise_text(series)
    return cleaned.map(lambda v: BRAND_SYNONYMS.get(v, v) if pd.notna(v) else pd.NA)


def _normalise_numeric(series: pd.Series, dtype=float) -> pd.Series:
    """Strip currency symbols / commas and cast to numeric."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalise_year(series: pd.Series) -> pd.Series:
    """Parse year to integer; mark implausible values as NA."""
    years = _normalise_numeric(series, dtype=float)
    mask = (years >= 1980) & (years <= 2026)
    return years.where(mask).astype("Int64")


def _normalise_fuel(series: pd.Series) -> pd.Series:
    """Collapse fuel type variants to canonical tokens."""
    fuel_map = {
        "gas": "gasoline",
        "petrol": "gasoline",
        "diesel": "diesel",
        "electric": "electric",
        "ev": "electric",
        "hybrid": "hybrid",
        "plug-in hybrid": "hybrid",
        "phev": "hybrid",
        "other": "other",
    }
    cleaned = _normalise_text(series)
    return cleaned.map(lambda v: fuel_map.get(v, v) if pd.notna(v) else pd.NA)


def _add_blocking_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create a normalised blocking key: <make>_<year>."""
    make = df.get("make", pd.Series(["unknown"] * len(df), index=df.index))
    year = df.get("year", pd.Series(["unknown"] * len(df), index=df.index))
    df["blocking_key"] = (
        make.fillna("unknown").astype(str)
        + "_"
        + year.fillna("unknown").astype(str)
    )
    return df


def _keep_mediated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not in the mediated schema (plus housekeeping cols)."""
    keep = MEDIATED_SCHEMA + ["record_id", "source", "blocking_key"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# Per-source loaders with dynamic sampling for real datasets
# ---------------------------------------------------------------------------

_sampled_dfs = {}

def load_source_direct(path: Path, source_name: str, rename_dict: dict, encoding: str = "utf-8") -> pd.DataFrame:
    logger.info("Loading %s directly from %s", source_name, path)
    df = pd.read_csv(path, low_memory=False, encoding=encoding)
    df = df.rename(columns=rename_dict)
    
    if "make" in df.columns:
        df["make"] = _normalise_make(df["make"])
    if "model" in df.columns:
        df["model"] = _normalise_text(df["model"])
    if "year" in df.columns:
        df["year"] = _normalise_year(df["year"])
    if "price" in df.columns:
        df["price"] = _normalise_numeric(df["price"])
    if "mileage" in df.columns:
        df["mileage"] = _normalise_numeric(df["mileage"])
    if "fuel_type" in df.columns:
        df["fuel_type"] = _normalise_fuel(df["fuel_type"])
        
    df = df.dropna(subset=["make", "model"]).reset_index(drop=True)
    df["source"] = source_name
    
    prefix_map = {"craigslist": "cl_", "usedcars": "uc_", "ebay": "eb_"}
    df["record_id"] = prefix_map[source_name] + df.index.astype(str)
    df = _add_blocking_key(df)
    return _keep_mediated_columns(df)


def load_and_sample_real_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Sampling from real datasets to build pipeline datasets and ground truth...")
    rng = np.random.default_rng(42)
    
    # 1. US Used Cars
    logger.info("Loading US Used Cars raw subset from %s", USEDCARS_FILE)
    uc_raw = pd.read_csv(USEDCARS_FILE, low_memory=False, nrows=20000)
    uc_raw = uc_raw.rename(columns=USEDCARS_RENAME)
    
    uc_clean = uc_raw.copy()
    if "make" in uc_clean.columns:
        uc_clean["make"] = _normalise_make(uc_clean["make"])
    if "model" in uc_clean.columns:
        uc_clean["model"] = _normalise_text(uc_clean["model"])
    if "year" in uc_clean.columns:
        uc_clean["year"] = _normalise_year(uc_clean["year"])
    if "price" in uc_clean.columns:
        uc_clean["price"] = _normalise_numeric(uc_clean["price"])
    if "mileage" in uc_clean.columns:
        uc_clean["mileage"] = _normalise_numeric(uc_clean["mileage"])
    if "fuel_type" in uc_clean.columns:
        uc_clean["fuel_type"] = _normalise_fuel(uc_clean["fuel_type"])
        
    uc_clean = uc_clean.dropna(subset=["make", "model", "year", "price", "mileage", "fuel_type"]).reset_index(drop=True)
    
    # 2. Craigslist
    logger.info("Loading Craigslist raw subset from %s", CRAIGSLIST_FILE)
    cl_raw = pd.read_csv(CRAIGSLIST_FILE, low_memory=False, nrows=20000)
    cl_raw = cl_raw.rename(columns=CRAIGSLIST_RENAME)
    
    cl_clean = cl_raw.copy()
    if "make" in cl_clean.columns:
        cl_clean["make"] = _normalise_make(cl_clean["make"])
    if "model" in cl_clean.columns:
        cl_clean["model"] = _normalise_text(cl_clean["model"])
    if "year" in cl_clean.columns:
        cl_clean["year"] = _normalise_year(cl_clean["year"])
    if "price" in cl_clean.columns:
        cl_clean["price"] = _normalise_numeric(cl_clean["price"])
    if "mileage" in cl_clean.columns:
        cl_clean["mileage"] = _normalise_numeric(cl_clean["mileage"])
    if "fuel_type" in cl_clean.columns:
        cl_clean["fuel_type"] = _normalise_fuel(cl_clean["fuel_type"])
        
    cl_clean = cl_clean.dropna(subset=["make", "model", "year", "price", "mileage", "fuel_type"]).reset_index(drop=True)
    
    # 3. eBay Motors
    logger.info("Loading eBay Motors raw subset from %s", EBAY_FILE)
    eb_raw = pd.read_csv(EBAY_FILE, low_memory=False, nrows=20000, encoding="latin1")
    eb_raw = eb_raw.rename(columns=EBAY_RENAME)
    
    eb_clean = eb_raw.copy()
    if "make" in eb_clean.columns:
        eb_clean["make"] = _normalise_make(eb_clean["make"])
    if "model" in eb_clean.columns:
        eb_clean["model"] = _normalise_text(eb_clean["model"])
    if "year" in eb_clean.columns:
        eb_clean["year"] = _normalise_year(eb_clean["year"])
    if "price" in eb_clean.columns:
        eb_clean["price"] = _normalise_numeric(eb_clean["price"])
    if "mileage" in eb_clean.columns:
        eb_clean["mileage"] = _normalise_numeric(eb_clean["mileage"])
    if "fuel_type" in eb_clean.columns:
        eb_clean["fuel_type"] = _normalise_fuel(eb_clean["fuel_type"])
        
    eb_clean = eb_clean.dropna(subset=["make", "model", "year", "price", "mileage", "fuel_type"]).reset_index(drop=True)
    
    # Compute blocking keys for all datasets to perform intelligent sampling
    uc_clean["blocking_key"] = uc_clean["make"].astype(str) + "_" + uc_clean["year"].astype(str)
    cl_clean["blocking_key"] = cl_clean["make"].astype(str) + "_" + cl_clean["year"].astype(str)
    eb_clean["blocking_key"] = eb_clean["make"].astype(str) + "_" + eb_clean["year"].astype(str)
    
    # Group by blocking key to find base entities with distinct make + year combinations
    base_groups = uc_clean.groupby("blocking_key").first().reset_index()
    if len(base_groups) >= 200:
        base_entities = base_groups.iloc[0:200][MEDIATED_SCHEMA].copy().reset_index(drop=True)
        base_blocking_keys = set(base_groups.iloc[0:200]["blocking_key"])
    else:
        logger.warning("Fewer than 200 distinct blocking keys found in US Used Cars. Using sequential sampling.")
        base_entities = uc_clean.iloc[0:200][MEDIATED_SCHEMA].copy().reset_index(drop=True)
        base_blocking_keys = set(uc_clean.iloc[0:200]["blocking_key"])
        
    # Craigslist overlap with noise
    cl_overlap = base_entities.copy()
    cl_overlap["make"] = cl_overlap["make"].apply(
        lambda x: rng.choice([x, x[:3], x.upper()], p=[0.7, 0.2, 0.1])
    )
    cl_overlap["model"] = cl_overlap["model"].apply(
        lambda x: x.replace(" ", "") if rng.random() < 0.15 else x
    )
    cl_overlap["year"] = cl_overlap["year"].apply(
        lambda x: x if rng.random() > 0.05 else x + rng.choice([-1, 1])
    )
    cl_overlap["price"] = cl_overlap["price"].apply(
        lambda x: x if rng.random() > 0.1 else x + rng.integers(-500, 500)
    )
    cl_overlap["mileage"] = cl_overlap["mileage"].apply(
        lambda x: np.nan if rng.random() < 0.15 else x
    )
    fuel_variations = {"gasoline": "gas", "diesel": "diesel", "electric": "ev", "hybrid": "hybrid"}
    cl_overlap["fuel_type"] = cl_overlap["fuel_type"].apply(
        lambda x: fuel_variations.get(x, x) if rng.random() < 0.2 else x
    )
    
    # Craigslist unique records: choose ones with blocking keys not in base_blocking_keys
    cl_unique_pool = cl_clean[~cl_clean["blocking_key"].isin(base_blocking_keys)]
    cl_unique = cl_unique_pool.iloc[0:200][MEDIATED_SCHEMA].copy().reset_index(drop=True)
    if len(cl_unique) < 200:
        logger.warning("Fewer than 200 Craigslist unique records found. Falling back.")
        cl_unique = cl_clean.iloc[200:400][MEDIATED_SCHEMA].copy().reset_index(drop=True)
        
    cl_df = pd.concat([cl_overlap, cl_unique], ignore_index=True)
    cl_df["source"] = "craigslist"
    cl_df["record_id"] = "cl_" + cl_df.index.astype(str)
    cl_df = _add_blocking_key(cl_df)
    cl_df = _keep_mediated_columns(cl_df)
    
    # US Used Cars with conflicts
    uc_overlap = base_entities.copy()
    conflict_idx = rng.choice(200, size=100, replace=False)
    alt_fuels = {"gasoline": "hybrid", "hybrid": "gasoline", "diesel": "gasoline", "electric": "hybrid"}
    for idx in conflict_idx:
        orig = uc_overlap.at[idx, "fuel_type"]
        uc_overlap.at[idx, "fuel_type"] = alt_fuels.get(orig, "gasoline")
        
    uc_unique_pool = uc_clean[~uc_clean["blocking_key"].isin(base_blocking_keys)]
    uc_unique = uc_unique_pool.iloc[0:200][MEDIATED_SCHEMA].copy().reset_index(drop=True)
    if len(uc_unique) < 200:
        logger.warning("Fewer than 200 US Used Cars unique records found. Falling back.")
        uc_unique = uc_clean.iloc[200:400][MEDIATED_SCHEMA].copy().reset_index(drop=True)
        
    uc_df = pd.concat([uc_overlap, uc_unique], ignore_index=True)
    uc_df["source"] = "usedcars"
    uc_df["record_id"] = "uc_" + uc_df.index.astype(str)
    uc_df = _add_blocking_key(uc_df)
    uc_df = _keep_mediated_columns(uc_df)
    
    # eBay overlap with price noise
    eb_overlap = base_entities.iloc[0:150].copy()
    eb_overlap["price"] = eb_overlap["price"].apply(
        lambda x: x if rng.random() > 0.15 else x + rng.integers(-1000, 1000)
    )
    
    eb_unique_pool = eb_clean[~eb_clean["blocking_key"].isin(base_blocking_keys)]
    eb_unique = eb_unique_pool.iloc[0:150][MEDIATED_SCHEMA].copy().reset_index(drop=True)
    if len(eb_unique) < 150:
        logger.warning("Fewer than 150 eBay unique records found. Falling back.")
        eb_unique = eb_clean.iloc[150:300][MEDIATED_SCHEMA].copy().reset_index(drop=True)
        
    eb_df = pd.concat([eb_overlap, eb_unique], ignore_index=True)
    eb_df["source"] = "ebay"
    eb_df["record_id"] = "eb_" + eb_df.index.astype(str)
    eb_df = _add_blocking_key(eb_df)
    eb_df = _keep_mediated_columns(eb_df)
    
    # Write ground truth files dynamically
    GROUND_TRUTH_DIR = DATA_DIR.parent / "ground_truth"
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    
    # positive pairs
    pairs = pd.DataFrame({
        "id_left": [f"cl_{i}" for i in range(200)],
        "id_right": [f"uc_{i}" for i in range(200)],
    })
    pairs_eb = pd.DataFrame({
        "id_left": [f"cl_{i}" for i in range(150)],
        "id_right": [f"eb_{i}" for i in range(150)],
    })
    all_pairs = pd.concat([pairs, pairs_eb], ignore_index=True)
    all_pairs.to_csv(GROUND_TRUTH_DIR / "positive_pairs.csv", index=False)
    logger.info("Wrote %d dynamic positive pairs to ground truth", len(all_pairs))
    
    # conflicts
    conflicts = pd.DataFrame({
        "entity_id": [f"uc_{i}" for i in conflict_idx],
        "attribute": ["fuel_type"] * 100,
        "correct_value": [base_entities.at[i, "fuel_type"] for i in conflict_idx],
    })
    conflicts.to_csv(GROUND_TRUTH_DIR / "conflicting_values.csv", index=False)
    logger.info("Wrote %d dynamic conflicts to ground truth", len(conflicts))
    
    return cl_df, uc_df, eb_df


def _ensure_samples_loaded():
    global _sampled_dfs
    if _sampled_dfs:
        return
        
    is_real = False
    if CRAIGSLIST_FILE.exists() and USEDCARS_FILE.exists() and EBAY_FILE.exists():
        if CRAIGSLIST_FILE.stat().st_size > 1024 * 1024:
            is_real = True
            
    if is_real:
        cl, uc, eb = load_and_sample_real_datasets()
        _sampled_dfs["craigslist"] = cl
        _sampled_dfs["usedcars"] = uc
        _sampled_dfs["ebay"] = eb
    else:
        logger.info("Real datasets not found or are small. Loading as-is (synthetic/samples)...")
        _sampled_dfs["craigslist"] = load_source_direct(CRAIGSLIST_FILE, "craigslist", CRAIGSLIST_RENAME)
        _sampled_dfs["usedcars"] = load_source_direct(USEDCARS_FILE, "usedcars", USEDCARS_RENAME)
        
        # Check encoding for eBay Motors synthetic vs real
        ebay_encoding = "latin1" if "ebay" in str(EBAY_FILE).lower() or "autos" in str(EBAY_FILE).lower() else "utf-8"
        try:
            _sampled_dfs["ebay"] = load_source_direct(EBAY_FILE, "ebay", EBAY_RENAME, encoding=ebay_encoding)
        except UnicodeDecodeError:
            _sampled_dfs["ebay"] = load_source_direct(EBAY_FILE, "ebay", EBAY_RENAME, encoding="latin1")


def load_craigslist(path: Path = CRAIGSLIST_FILE, n: int = 400) -> pd.DataFrame:
    """Load and normalise the Craigslist dataset (supporting cache / dynamic sampling)."""
    _ensure_samples_loaded()
    return _sampled_dfs["craigslist"].head(n).reset_index(drop=True)


def load_usedcars(path: Path = USEDCARS_FILE, n: int = 400) -> pd.DataFrame:
    """Load and normalise the US Used Cars dataset (supporting cache / dynamic sampling)."""
    _ensure_samples_loaded()
    return _sampled_dfs["usedcars"].head(n).reset_index(drop=True)


def load_ebay(path: Path = EBAY_FILE, n: int = 300) -> pd.DataFrame:
    """Load and normalise the eBay Motors dataset (supporting cache / dynamic sampling)."""
    _ensure_samples_loaded()
    return _sampled_dfs["ebay"].head(n).reset_index(drop=True)


def load_all_sources() -> pd.DataFrame:
    """
    Load all three sources and concatenate into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Combined dataset with unique record_ids and a 'source' column.
    """
    cl = load_craigslist()
    uc = load_usedcars()
    eb = load_ebay()
    combined = pd.concat([cl, uc, eb], ignore_index=True)
    logger.info(
        "Combined dataset: %d records (CL=%d, UC=%d, EB=%d)",
        len(combined), len(cl), len(uc), len(eb),
    )
    return combined


# ---------------------------------------------------------------------------
# Synthetic dataset generator (used when real CSVs are not available)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(seed: int = 42) -> None:
    """
    Generate synthetic automotive CSV files in data/ for development/testing.
    Produces:
      - 400 Craigslist records
      - 400 US Used Cars records
      - 300 eBay Motors records
    with deliberate overlaps, conflicts, and noise to satisfy ground truth
    requirements (300+ positive pairs, 100+ conflicting values).
    """
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    makes = ["toyota", "honda", "ford", "chevrolet", "volkswagen",
             "bmw", "mercedes-benz", "nissan", "hyundai", "kia"]
    models = {
        "toyota": ["camry", "corolla", "rav4", "highlander"],
        "honda": ["civic", "accord", "cr-v", "pilot"],
        "ford": ["f-150", "mustang", "escape", "explorer"],
        "chevrolet": ["silverado", "malibu", "equinox", "tahoe"],
        "volkswagen": ["jetta", "passat", "tiguan", "golf"],
        "bmw": ["3 series", "5 series", "x3", "x5"],
        "mercedes-benz": ["c-class", "e-class", "glc", "gle"],
        "nissan": ["altima", "rogue", "sentra", "frontier"],
        "hyundai": ["elantra", "sonata", "tucson", "santa fe"],
        "kia": ["optima", "sorento", "sportage", "soul"],
    }
    fuels = ["gasoline", "diesel", "electric", "hybrid"]
    fuel_weights = [0.55, 0.20, 0.10, 0.15]

    def _make_base_records(n: int) -> pd.DataFrame:
        make_list = rng.choice(makes, size=n)
        model_list = [rng.choice(models[m]) for m in make_list]
        year_list = rng.integers(2010, 2024, size=n)
        price_list = rng.integers(5000, 55000, size=n)
        mileage_list = rng.integers(0, 200000, size=n)
        fuel_list = rng.choice(fuels, size=n, p=fuel_weights)
        return pd.DataFrame({
            "make": make_list,
            "model": model_list,
            "year": year_list,
            "price": price_list,
            "mileage": mileage_list,
            "fuel_type": fuel_list,
        })

    # ---- Create 200 "true" entities ----
    entities = _make_base_records(200)

    # ---- Craigslist: 200 entity copies + 200 unique, with noise ----
    cl_overlap = entities.copy()
    # Add noise: abbreviations, nulls, typos
    cl_overlap["make"] = cl_overlap["make"].apply(
        lambda x: rng.choice([x, x[:3], x.upper()], p=[0.7, 0.2, 0.1])
    )
    cl_overlap["model"] = cl_overlap["model"].apply(
        lambda x: x if rng.random() > 0.15 else x.replace(" ", "")
    )
    cl_overlap["year"] = cl_overlap["year"].apply(
        lambda x: x if rng.random() > 0.05 else x + rng.integers(-1, 2)
    )
    cl_overlap["price"] = cl_overlap["price"].apply(
        lambda x: x if rng.random() > 0.1 else x + int(rng.integers(-500, 501))
    )
    # Introduce nulls (realistic for Craigslist)
    null_mask = rng.random(len(cl_overlap)) < 0.15
    cl_overlap.loc[null_mask, "mileage"] = np.nan
    cl_overlap["fuel_type"] = cl_overlap["fuel_type"].apply(
        lambda x: x if rng.random() > 0.20 else rng.choice(
            ["gas", "petrol", "diesel", "electric"], p=[0.4, 0.2, 0.2, 0.2]
        )
    )
    cl_unique = _make_base_records(200)
    cl_df = pd.concat([cl_overlap, cl_unique], ignore_index=True)
    cl_df.to_csv(CRAIGSLIST_FILE, index=False)
    logger.info("Wrote %d Craigslist records → %s", len(cl_df), CRAIGSLIST_FILE)

    # ---- US Used Cars: 200 entity copies + 200 unique, clean ----
    uc_overlap = entities.copy()
    uc_overlap["mileage"] = uc_overlap["mileage"].apply(
        lambda x: x + int(rng.integers(-2000, 2001))
    )
    # Deliberately introduce 100 conflicting fuel_type values (vs entities)
    conflict_idx = rng.choice(len(uc_overlap), size=100, replace=False)
    alt_fuels = {"gasoline": "hybrid", "hybrid": "gasoline",
                 "diesel": "gasoline", "electric": "hybrid"}
    for idx in conflict_idx:
        orig = uc_overlap.at[idx, "fuel_type"]
        uc_overlap.at[idx, "fuel_type"] = alt_fuels.get(orig, "gasoline")

    uc_unique = _make_base_records(200)
    uc_df = pd.concat([uc_overlap, uc_unique], ignore_index=True)
    # Rename to simulate different column names
    uc_df = uc_df.rename(columns={"make": "make_name", "model": "model_name",
                                   "mileage": "milage"})
    uc_df.to_csv(USEDCARS_FILE, index=False)
    logger.info("Wrote %d US Used Cars records → %s", len(uc_df), USEDCARS_FILE)

    # ---- eBay Motors: 150 entity copies + 150 unique ----
    eb_overlap = entities.sample(n=150, random_state=seed).copy().reset_index(drop=True)
    eb_overlap["price"] = eb_overlap["price"].apply(
        lambda x: x + int(rng.integers(-1000, 1001))
    )
    eb_overlap["make"] = eb_overlap["make"].apply(
        lambda x: rng.choice([x, x.title()], p=[0.6, 0.4])
    )
    eb_unique = _make_base_records(150)
    eb_df = pd.concat([eb_overlap, eb_unique], ignore_index=True)
    # Rename to simulate eBay column names
    eb_df = eb_df.rename(columns={"make": "brand", "mileage": "vehicle_mileage",
                                   "fuel_type": "fuel_type_display",
                                   "price": "listing_price", "year": "vehicle_year"})
    eb_df.to_csv(EBAY_FILE, index=False)
    logger.info("Wrote %d eBay Motors records → %s", len(eb_df), EBAY_FILE)

    # ---- Ground Truth: positive pairs (CL↔UC, first 200) ----
    GROUND_TRUTH_DIR = DATA_DIR.parent / "ground_truth"
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    pairs = pd.DataFrame({
        "id_left": [f"cl_{i}" for i in range(200)],
        "id_right": [f"uc_{i}" for i in range(200)],
    })
    # Add CL↔EB pairs for first 150 overlapping entities
    pairs_eb = pd.DataFrame({
        "id_left": [f"cl_{i}" for i in range(150)],
        "id_right": [f"eb_{i}" for i in range(150)],
    })
    all_pairs = pd.concat([pairs, pairs_eb], ignore_index=True)
    all_pairs.to_csv(GROUND_TRUTH_DIR / "positive_pairs.csv", index=False)
    logger.info("Wrote %d positive pairs", len(all_pairs))

    # ---- Ground Truth: conflicting attribute values ----
    conflicts = pd.DataFrame({
        "entity_id": [f"entity_{i}" for i in conflict_idx],
        "attribute": ["fuel_type"] * 100,
        "correct_value": [entities.at[i, "fuel_type"] for i in conflict_idx],
    })
    conflicts.to_csv(GROUND_TRUTH_DIR / "conflicting_values.csv", index=False)
    logger.info("Wrote %d conflicting value ground-truth rows", len(conflicts))

    # ---- Ground Truth: schema mapping ----
    import json
    schema_gt = {
        "craigslist": {
            "manufacturer": "make",
            "model": "model",
            "year": "year",
            "price": "price",
            "odometer": "mileage",
            "fuel": "fuel_type",
        },
        "usedcars": {
            "make_name": "make",
            "model_name": "model",
            "year": "year",
            "price": "price",
            "milage": "mileage",
            "fuel_type": "fuel_type",
        },
        "ebay": {
            "brand": "make",
            "model": "model",
            "vehicle_year": "year",
            "listing_price": "price",
            "vehicle_mileage": "mileage",
            "fuel_type_display": "fuel_type",
        },
    }
    with open(GROUND_TRUTH_DIR / "schema_ground_truth.json", "w") as f:
        json.dump(schema_gt, f, indent=2)
    logger.info("Wrote schema ground truth → %s", GROUND_TRUTH_DIR / "schema_ground_truth.json")
