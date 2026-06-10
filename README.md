# LLM-Assisted Big Data Integration Pipeline

**Advanced Topics in Computer Science — Roma Tre Università**

A modular Python pipeline comparing a **Traditional Baseline (Pipeline A)** with an **LLM-Assisted Hybrid (Pipeline B)** for integrating three heterogeneous automotive datasets: Craigslist, US Used Cars, and eBay Motors.

---

## Project Structure

```
automotive_integration/
├── config.py            # Thresholds, paths, Ollama settings
├── data_loader.py       # Per-source loaders + synthetic data generator
├── pipeline_a.py        # Traditional Baseline (schema + blocking + linkage + fusion)
├── pipeline_b.py        # LLM-Assisted Hybrid (selective LLM calls)
├── llm_client.py        # Ollama wrapper with SQLite cache + JSONL logging
├── evaluation.py        # P/R/F1 metrics, error analysis, errors.json
├── main.py              # Orchestrator CLI
├── requirements.txt
├── data/                # Place source CSVs here
├── ground_truth/        # positive_pairs.csv, conflicting_values.csv, schema_ground_truth.json
├── logs/                # pipeline.log, llm_calls.jsonl
├── cache/               # llm_cache.db (SQLite)
└── output/              # integrated_dataset.csv, errors.json, metrics_report.json
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# Download Ollama from https://ollama.com
ollama pull llama3:8b
ollama serve                    # Starts the REST API on localhost:11434
```

### 3. Prepare datasets

**Option A — Real Kaggle datasets** (recommended for submission):

| Source       | Kaggle Dataset                           | File to download                                  |
| ------------ | ---------------------------------------- | ------------------------------------------------- |
| Craigslist   | `austinreese/craigslist-carstrucks-data` | `vehicles.csv` → `data/craigslist_sample.csv`     |
| US Used Cars | `ananaymital/us-used-cars-dataset`       | `used_cars_data.csv` → `data/usedcars_sample.csv` |
| eBay Motors  | `doaaalsenani/ebay-motors`               | `eBay_motors.csv` → `data/ebay_sample.csv`        |

This creates all three CSVs plus ground truth files automatically.

---

## Running the Pipeline

```bash
# Full run: Pipeline A + Pipeline B + Evaluation
python main.py

# Generate synthetic data first, then run everything
python main.py --generate-synthetic

# Pipeline A only (no LLM required)
python main.py --skip-b

# Skip evaluation (pipelines only)
python main.py --skip-eval
```

---

## Academic Constraints Checklist

| Requirement                | Implementation                                   |
| -------------------------- | ------------------------------------------------ |
| 3 heterogeneous sources    | Craigslist, US Used Cars, eBay Motors            |
| ≥ 1,000 total records      | 400 + 400 + 300 = 1,100 records                  |
| ≥ 5 mediated attributes    | make, model, year, price, mileage, fuel_type (6) |
| ≥ 200 integrated entities  | Configurable; synthetic generates 200+           |
| ≥ 300 positive pairs       | Synthetic generates 350 (200 CL↔UC + 150 CL↔EB)  |
| ≥ 100 conflicting values   | 100 deliberately injected fuel_type conflicts    |
| Strict JSON output         | Ollama JSON mode + required_keys validation      |
| Documented fallback policy | Pipeline A decision on any LLM failure           |
| No manual corrections      | Enforced; all failures logged in errors.json     |
| LLM call logging           | `logs/llm_calls.jsonl` – every call with latency |
| Local LLM cache            | SQLite (`cache/llm_cache.db`)                    |

---

## Model Configuration

Edit `config.py` to change:

```python
OLLAMA_MODEL = "llama3"          # or "llama3.1", "qwen2.5:7b"
OLLAMA_TEMPERATURE = 0.0         # Deterministic output
OLLAMA_MAX_TOKENS = 500
OLLAMA_TIMEOUT_SECONDS = 60
```

---

## Outputs

| File                               | Description                                          |
| ---------------------------------- | ---------------------------------------------------- |
| `output/integrated_pipeline_a.csv` | Pipeline A fused entities                            |
| `output/integrated_dataset.csv`    | Pipeline B fused entities (final)                    |
| `output/metrics_report.json`       | All P/R/F1 metrics in machine-readable format        |
| `output/errors.json`               | ≥ 3 detailed error examples for the technical report |
| `logs/llm_calls.jsonl`             | Full log of every LLM call                           |
| `logs/pipeline.log`                | Human-readable pipeline log                          |

---

## Evaluation Metrics

| Component        | Metric                                             |
| ---------------- | -------------------------------------------------- |
| Schema Alignment | Precision, Recall, F1 on attribute mappings        |
| Record Linkage   | Candidate pairs after blocking, Pairwise P/R/F1    |
| Data Fusion      | Accuracy on conflicting attributes vs ground truth |

---

## Failure Policy

When the LLM returns invalid JSON, missing required fields, hallucinated attribute names, or times out:

1. The failure is **logged** in `logs/llm_calls.jsonl` with `"fallback_triggered": true`.
2. The system **falls back silently** to Pipeline A's deterministic decision.
3. The fused record is still produced — **no manual corrections, no interruptions**.
4. Failures accumulate in `pipeline_b._stats` and are reported in `metrics_report.json`.

---

## Prompts

All prompts are version-controlled inside `pipeline_b.py`:

- `_SCHEMA_SYSTEM_PROMPT` — Schema alignment
- `_LINKAGE_SYSTEM_PROMPT` — Borderline record matching
- `_FUSION_SYSTEM_PROMPT` — Conflicting attribute resolution

---

## Authors

- Edoardo Piazzolla

Roma Tre Università — Advanced Topics in Computer Science
