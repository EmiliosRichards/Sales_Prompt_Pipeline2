### Run manifests & run history (auditability)

This repo now records a **machine-readable snapshot** of how each pipeline run was executed.

#### What gets written

- **Per-run manifest** (full detail):
  - Location: `output_data/<run_id>/run_manifest.json`
  - Includes:
    - run id, start/end timestamps, duration, status
    - CLI `argv` + parsed args (where available)
    - sanitized `AppConfig` snapshot (prompt paths, model, scraper/phone flags, etc.)
    - input file metadata (path, size, mtime)
    - git versioning (commit, branch, dirty)
    - output file inventory + optional CSV row counts

- **Global run history index** (quick lookup):
  - Location: `output_data/_run_history.jsonl`
  - Format: JSONL (one JSON object per run; append-only)
  - Stores a compact summary (run id, pipeline name, status, input path, git info, key row counts).

#### Which entrypoints write manifests

- `phone_extract.py` (phone-only pipeline)
- `main_pipeline.py` (full pipeline; single-process and `--workers` parallel mode)

#### Row counting behavior

By default, the manifest attempts robust CSV row counting (safe with quoted multiline fields) for key stable outputs.
You can disable it by setting:

```bash
set RUN_MANIFEST_COUNT_ROWS=False
```

#### Notes / safety

- Secrets (API keys, tokens, passwords) are **redacted** in config snapshots.
- Manifest writing is **best-effort**: it should never crash a run.

