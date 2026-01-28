### Output file conventions (all scripts)

The goal is: **you can monitor progress live**, and you can always find a “final” copy after completion.

---

### Naming conventions

For a given run id `<run_id>`, outputs are written under:

- `output_data/<run_id>/...`

And follow these patterns:

- **Live, append-only outputs**
  - `*_live.csv` (human friendly)
  - `*_live.jsonl` (machine friendly; one row/object per line)
  - `*_live_status.json` (progress counters)

- **Stable final copies**
  - `*.csv` + `*.jsonl` (copied from live outputs once the run finishes)
  - After stable finals are confirmed to exist and be non-empty, the pipeline may delete the `_live.*` files to reduce clutter.

---

### Phone-only (`phone_extract.py`)

#### Files
- `phone_extraction_results_<run_id>_live.csv`
- `phone_extraction_results_<run_id>_live.jsonl`
- `input_augmented_<run_id>_live.csv` (CSV inputs only; preserves delimiter)
- `input_augmented_<run_id>_live.jsonl` (CSV inputs only)
- `phone_extract_<run_id>_live_status.json`

Final copies:
- `phone_extraction_results_<run_id>.csv` / `.jsonl`
- `input_augmented_<run_id>.csv` / `.jsonl`

#### Workers
When `--workers > 1`:
- `workers/wXofN/phone_extraction_results_wXofN.csv` (+ `.jsonl`)
- plus `failed_rows_*.csv`, `run_metrics_*.md`, and `scraped_content/`.

---

### Full pipeline (`main_pipeline.py`)

#### Files
- `SalesOutreachReport_<run_id>_live.csv`
- `SalesOutreachReport_<run_id>_live.jsonl`
- `input_augmented_<run_id>_live.csv`
- `input_augmented_<run_id>_live.jsonl`
- `SalesOutreachReport_<run_id>_live_status.json`

Final copies:
- `SalesOutreachReport_<run_id>.csv` (+ sometimes `.xlsx`)
- `SalesOutreachReport_<run_id>.jsonl`
- `input_augmented_<run_id>.csv` / `.jsonl`

Notes:
- In parallel mode (`--workers > 1`), final outputs drop internal `__meta_*` fields used for live monitoring.
- When phone retrieval is skipped (`--skip-phone-retrieval`), the pipeline does **not** rerun phone scraping/LLM logic.
  It can still populate convenience fields like `found_number` / `PhoneNumber_Status` based on existing phone columns
  already present in the input.

---

### Stitching (single deliverable): `scripts/stitch_5_30_full_output.py`

Use this when you want a single combined deliverable that is:
- phone-extraction `input_augmented` as the base, plus
- sales fields appended from one or more `SalesOutreachReport_*.csv` files.

Outputs (under the chosen output folder):
- `input_augmented_5_30_STITCHED.csv` / `.jsonl` (semicolon-delimited CSV)
- `SalesOutreachReport_5_30_STITCHED.csv` / `.jsonl` (comma-delimited CSV)
- `SalesOutreachReport_5_30_STITCHED_only_inputs.csv` / `.jsonl` (optional “only base inputs” variant)
- `STITCH_SUMMARY.json` (counts: matched rows, orphans, etc.)

---

### Repairing stalled phone runs (best-effort)

If a parallel `phone_extract.py` run stalls (commonly due to high concurrency + Playwright), you may end up with only:
- `phone_extraction_results_<run_id>_live.*`
- `input_augmented_<run_id>_live.*`
- partial worker outputs under `workers/`

Recommended approach:
- rerun missing row ranges (smaller `--workers`), then
- rebuild a complete run folder using `scripts/repair_stalled_phone_run.py`

This produces (in the original base run folder):
- final + merged results and augmented files
- `REPAIR_SUMMARY.md` documenting which retry runs were merged

---

### CSV vs JSONL for complex columns

Some columns are lists/dicts (e.g. `RegexCandidateSnippets`, `LLMExtractedNumbers`).

- **CSV**: complex values are JSON-stringified (you may see `"[]"`)
- **JSONL**: complex values are actual arrays/objects

If you want to program against these fields reliably: prefer JSONL.

