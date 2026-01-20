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

---

### CSV vs JSONL for complex columns

Some columns are lists/dicts (e.g. `RegexCandidateSnippets`, `LLMExtractedNumbers`).

- **CSV**: complex values are JSON-stringified (you may see `"[]"`)
- **JSONL**: complex values are actual arrays/objects

If you want to program against these fields reliably: prefer JSONL.

