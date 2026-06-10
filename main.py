"""
main.py
=======
Orchestrator for the LLM-Assisted Big Data Integration project.

Usage:
  python main.py [--generate-synthetic] [--skip-b] [--skip-eval]

  --generate-synthetic   Create synthetic CSV files in data/ (for testing
                         without real Kaggle datasets).
  --skip-b               Run Pipeline A only (useful for baseline benchmarking).
  --skip-eval            Skip the evaluation step.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import (
    CRAIGSLIST_FILE,
    EBAY_FILE,
    INTEGRATED_DATASET_FILE,
    OUTPUT_DIR,
    USEDCARS_FILE,
    setup_logging,
)

logger = setup_logging()


def main(args: argparse.Namespace) -> None:
    """Main orchestration function."""

    # ------------------------------------------------------------------
    # 0. Optionally generate synthetic data
    # ------------------------------------------------------------------
    if args.generate_synthetic:
        logger.info("Generating synthetic datasets …")
        from data_loader import generate_synthetic_dataset
        generate_synthetic_dataset()

    # ------------------------------------------------------------------
    # 1. Load all sources
    # ------------------------------------------------------------------
    import pandas as pd
    from data_loader import load_all_sources

    raw_sources = {
        "craigslist": pd.read_csv(CRAIGSLIST_FILE, low_memory=False, nrows=100) if CRAIGSLIST_FILE.exists() else pd.DataFrame(),
        "usedcars": pd.read_csv(USEDCARS_FILE, low_memory=False, nrows=100) if USEDCARS_FILE.exists() else pd.DataFrame(),
        "ebay": pd.read_csv(EBAY_FILE, low_memory=False, nrows=100, encoding="latin1") if EBAY_FILE.exists() else pd.DataFrame(),
    }

    source_raw_columns = {src: list(df.columns) for src, df in raw_sources.items()}
    combined_df = load_all_sources()
    logger.info("Total input records: %d", len(combined_df))

    # ------------------------------------------------------------------
    # 2. Run Pipeline A (Traditional Baseline)
    # ------------------------------------------------------------------
    from pipeline_a import run_pipeline_a

    integrated_a, artefacts_a = run_pipeline_a(
        combined_df,
        source_raw_columns=source_raw_columns,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    integrated_a.to_csv(OUTPUT_DIR / "integrated_pipeline_a.csv", index=False)
    logger.info("Pipeline A output: %d entities", len(integrated_a))

    if args.skip_b:
        logger.info("--skip-b flag set; skipping Pipeline B.")
        return

    # ------------------------------------------------------------------
    # 3. Run Pipeline B (LLM-Assisted)
    # ------------------------------------------------------------------
    from pipeline_b import run_pipeline_b

    integrated_b, artefacts_b, conflict_logs = run_pipeline_b(
        combined_df,
        pipeline_a_artefacts=artefacts_a,
        source_raw_data=raw_sources,
    )

    integrated_b.to_csv(INTEGRATED_DATASET_FILE, index=False)
    logger.info("Pipeline B output: %d entities", len(integrated_b))

    # ------------------------------------------------------------------
    # 4. Evaluation
    # ------------------------------------------------------------------
    if args.skip_eval:
        logger.info("--skip-eval flag set; skipping evaluation.")
        return

    from evaluation import run_evaluation

    run_evaluation(
        pipeline_a_artefacts=artefacts_a,
        pipeline_b_artefacts=artefacts_b,
        integrated_a=integrated_a,
        integrated_b=integrated_b,
        conflict_logs=conflict_logs,
    )

    logger.info("All done. Check output/ for results and logs/ for LLM call logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-Assisted Big Data Integration Pipeline"
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic CSV datasets in data/ before running.",
    )
    parser.add_argument(
        "--skip-b",
        action="store_true",
        help="Run Pipeline A only.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the evaluation step.",
    )
    main(parser.parse_args())
