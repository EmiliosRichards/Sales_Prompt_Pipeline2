### Systems overview (high level)

This repo contains a few **systems** (pipelines + helpers) that can be run independently or chained.

---

### 1) Phone-only extraction system (`phone_extract.py`)

#### What it does
For each input row:
- Scrapes the website (Playwright, with optional `httpx` fallback).
- Extracts regex phone candidates + snippets.
- Uses an LLM to classify/validate candidates.
- Consolidates candidates into **ranked, de-duplicated numbers**.
- Writes **incremental live outputs** and final outputs.

#### Typical use case
Run first to enrich a large dataset with phone numbers, then run the full pipeline to generate pitches without re-running phone retrieval.

#### Operational note (parallel runs)
Very high `--workers` values can stall due to Playwright/browser concurrency.
If you need large throughput, prefer moderate values (e.g. 10–25) and monitor `_live_status.json`.

#### Inputs
- CSV/XLSX with columns mapped via `--input-profile` (see `AppConfig.INPUT_COLUMN_PROFILES`).

#### Outputs (always written under `output_data/<run_id>/`)
- `phone_extraction_results_<run_id>_live.csv` / `.jsonl`
- `input_augmented_<run_id>_live.csv` / `.jsonl` (CSV inputs only)
- `phone_extract_<run_id>_live_status.json`
- Stable finals (copied from live):
  - `phone_extraction_results_<run_id>.csv` / `.jsonl`
  - `input_augmented_<run_id>.csv` / `.jsonl` (CSV inputs only)
- Worker artifacts (when `--workers > 1`): `workers/wXofN/...`

---

### 2) Full sales pitch generation system (`main_pipeline.py`)

#### What it does
For each input row:
- Scrapes the website (or reuses cached scraped text).
- Summarizes the website (LLM).
- Extracts structured attributes (LLM).
- Optionally: retrieves phone numbers (phone retrieval adapter).
- Matches a golden partner (LLM).
- Generates a sales pitch (LLM).
- Writes **incremental live outputs** and final reports.

#### Typical use case
Generate outreach-ready rows (pitch + match + attribute columns) for rows that have a usable phone.

#### Inputs
- CSV/XLSX with columns mapped via `--input-profile`.
- Often chained after `phone_extract.py` (use a profile that maps `PhoneNumber_Found` into `PhoneNumber`).

#### Outputs (always written under `output_data/<run_id>/`)
- `SalesOutreachReport_<run_id>_live.csv` / `.jsonl`
- `input_augmented_<run_id>_live.csv` / `.jsonl`
- `SalesOutreachReport_<run_id>_live_status.json` (parallel progress)
- Stable finals (copied from live)

---

### 3) Resume system for sales pitches (`scripts/resume_sales_pitches.py`)

#### What it does
Creates a filtered “remaining rows” CSV by diffing:
- an input CSV (usually the augmented input from phone_extract) vs
- an existing SalesOutreachReport live/final CSV (rows that already have a pitch)

Then calls `main_pipeline.py` with `--workers` and (typically) `--skip-phone-retrieval` and scrape reuse.

#### Inputs
- `--input`: augmented input CSV
- `--resume-from`: existing SalesOutreachReport CSV (live or final)

#### Outputs
Creates a new run under `output_data/` (standard full pipeline outputs).

---

### 4) Stitching system (optional, for “single deliverable”)

If a run was completed in multiple passes (partial run + resume run), use:
- `scripts/stitch_5_30_full_output.py`

It overlays sales outputs onto a base augmented input file and can produce a “only inputs” report variant to avoid stray rows.

---

### 5) Repair helper for stalled phone runs (`scripts/repair_stalled_phone_run.py`)

When a parallel `phone_extract.py` run stalls, you may end up with partial `_live.*` outputs.
The repair helper can rebuild a complete final output set by:
- taking the base run’s partial live results
- overlaying one or more retry runs (missing row ranges)

It writes standard final artifacts into the base run folder and records what happened in `REPAIR_SUMMARY.md`.

