"""
config.py
=========
Central configuration file for the LLM-Assisted Big Data Integration pipeline.
Holds similarity thresholds, file paths, Ollama API settings, and logging setup.
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
GROUND_TRUTH_DIR = BASE_DIR / "ground_truth"
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "output"

# Source dataset filenames (place CSVs in data/)
CRAIGSLIST_FILE = DATA_DIR / "craiglist" / "vehicles.csv"
USEDCARS_FILE = DATA_DIR / "us_used_cars" / "used_cars_data.csv"
EBAY_FILE = DATA_DIR / "ebay_motors" / "autos.csv"

# Ground truth files
GT_PAIRS_FILE = GROUND_TRUTH_DIR / "positive_pairs.csv"        # columns: id_left, id_right
GT_CONFLICTS_FILE = GROUND_TRUTH_DIR / "conflicting_values.csv" # columns: entity_id, attribute, correct_value
GT_SCHEMA_FILE = GROUND_TRUTH_DIR / "schema_ground_truth.json"  # {source: {src_col: mediated_col}}

# Output files
INTEGRATED_DATASET_FILE = OUTPUT_DIR / "integrated_dataset.csv"
ERRORS_FILE = OUTPUT_DIR / "errors.json"
METRICS_FILE = OUTPUT_DIR / "metrics_report.json"
LLM_LOG_FILE = LOGS_DIR / "llm_calls.jsonl"
CACHE_DB_FILE = CACHE_DIR / "llm_cache.db"

# ---------------------------------------------------------------------------
# Mediated Schema
# ---------------------------------------------------------------------------
MEDIATED_SCHEMA = ["make", "model", "year", "price", "mileage", "fuel_type"]

# Source priority for data fusion tiebreaking (higher index = lower priority)
SOURCE_PRIORITY = {
    "usedcars": 1,
    "ebay": 2,
    "craigslist": 3,
}

# ---------------------------------------------------------------------------
# Schema Alignment Thresholds
# ---------------------------------------------------------------------------
SCHEMA_JARO_THRESHOLD = 0.85      # Minimum Jaro-Winkler score for auto-mapping
SCHEMA_FALLBACK_MAP = {
    # Craigslist column aliases
    "manufacturer": "make",
    "odometer": "mileage",
    "fuel": "fuel_type",
    # US Used Cars aliases
    "make_name": "make",
    "model_name": "model",
    "milage": "mileage",
    "horsepower": None,           # No mapping in mediated schema
    "engine": None,
    # eBay aliases
    "brand": "make",
    "vehicle_mileage": "mileage",
    "kilometer": "mileage",
    "fuel_type_display": "fuel_type",
    "fueltype": "fuel_type",
    "listing_price": "price",
    "price": "price",
    "vehicle_year": "year",
    "yearofregistration": "year",
}

# ---------------------------------------------------------------------------
# Blocking Configuration
# ---------------------------------------------------------------------------
# Blocking key: normalised make + year (e.g. "toyota_2018")
BLOCKING_KEY_FIELDS = ["make", "year"]

# Brand synonym dictionary for blocking normalisation
BRAND_SYNONYMS = {
    "vw": "volkswagen",
    "chevy": "chevrolet",
    "benz": "mercedes-benz",
    "mercedes": "mercedes-benz",
    "merc": "mercedes-benz",
    "bmw": "bmw",
    "land rover": "landrover",
    "alfa": "alfa romeo",
    "ram": "ram",
}

# ---------------------------------------------------------------------------
# Record Linkage Thresholds
# ---------------------------------------------------------------------------
LINKAGE_MATCH_THRESHOLD = 0.75    # Combined score >= this → confirmed match
LINKAGE_NONMATCH_THRESHOLD = 0.30 # Combined score <  this → confirmed non-match
# Borderline zone: [LINKAGE_NONMATCH_THRESHOLD, LINKAGE_MATCH_THRESHOLD] → LLM call

# Attribute weights for combined similarity score
SIMILARITY_WEIGHTS = {
    "make": 0.20,
    "model": 0.35,
    "year": 0.20,
    "price": 0.10,
    "mileage": 0.10,
    "fuel_type": 0.05,
}

# Numeric tolerance for year and price comparison
YEAR_TOLERANCE = 1      # ± 1 year → similarity degrades linearly
PRICE_TOLERANCE = 2000  # ± $2000 → similarity degrades linearly
MILEAGE_TOLERANCE = 10000  # ± 10k miles

# ---------------------------------------------------------------------------
# Ollama / Local LLM Settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:latest"           # Change to "llama3.1" or "qwen2.5:7b" if needed
OLLAMA_TEMPERATURE = 0.0
OLLAMA_MAX_TOKENS = 500
OLLAMA_TIMEOUT_SECONDS = 60       # Per-request HTTP timeout
OLLAMA_MAX_RETRIES = 2            # Retry on transient errors before fallback

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> logging.Logger:
    """Configure root logger with console and file handlers."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "pipeline.log", mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger("integration")
