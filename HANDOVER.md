### Project handover: Sales Prompt Pipeline (full pipeline + phone-only)

This repo contains **two related pipelines**:

- **Full pipeline**: scrapes + summarizes + extracts attributes + partner-matches + generates sales pitch, and optionally retrieves a phone number. Entry point: `main_pipeline.py`.
- **Phone-only pipeline (isolated)**: does *only* phone retrieval (scrape → regex candidates → LLM classify → consolidate → choose top numbers) and writes results + an augmented input file. Entry point: `phone_extract.py`.

---

### 1) Quickstart (Windows / PowerShell)

#### Install
- **Python**: 3.10+ recommended
- **Install deps**:

```bash
pip install -r requirements.txt
python -m playwright install
```

#### Configure
- Create `.env` in the repo root (see `env.example` in this repo — copy it to `.env`).
- **Minimum required** to run any LLM-dependent pipeline:
  - `GEMINI_API_KEY=...`

---

### 2) How to run: phone-only (recommended for bulk phone enrichment)

#### Common commands
- **Single-process (simplest)**:

```bash
python phone_extract.py -i "data\Apollo Shopware Partner Germany 1(Sheet1).csv" -s phone_only
```

- **Process a slice** (1-based row range, inclusive):

```bash
python phone_extract.py -i "data\Apollo Shopware Partner Germany 1(Sheet1).csv" -r 1-50 -s quick50
```

- **Parallel** (splits the requested row range into chunks and runs multiple worker processes):

```bash
python phone_extract.py -i "data\Apollo Shopware Partner Germany 1(Sheet1).csv" -r 1-190 -s full25 --workers 25
```

#### Output layout (phone-only)
Phone-only runs always write into **one master run folder**:

- `output_data/<master_run_id>/phone_extraction_results_<master_run_id>_merged.csv`
- `output_data/<master_run_id>/input_augmented_<master_run_id>.csv` (CSV inputs only)
- `output_data/<master_run_id>/phone_extract_<master_run_id>.log`
- `output_data/<master_run_id>/workers/`
  - `workers/wXofN/phone_extraction_results_wXofN.csv`
  - `workers/wXofN/failed_rows_wXofN.csv`
  - `workers/wXofN/run_metrics_wXofN.md`
  - `workers/wXofN/scraped_content/...` (scraped cleaned-text artifacts)

#### What columns you get
The phone-only result CSV includes many internal columns; the **augmented CSV** is intended for end-users and includes:

- `PhoneNumber_Found`
- `PhoneType_Found`
- `PhoneSources_Found`
- `PhoneExtract_Outcome`
- `PhoneExtract_FaultCategory`
- `HttpFallbackAttempted`
- `HttpFallbackUsed`
- `HttpFallbackResult`

Phone-only also preserves your original input columns because it slices and re-attaches them when writing `input_augmented_*.csv`.

#### Phone number formatting (E.164 + Excel text protection)
- Regex extraction produces candidates in **E.164** where possible (e.g. `+49...`).
- When writing the augmented CSV, `phone_extract.py` normalizes the chosen phone to E.164 using:
  - `src/phone_retrieval/data_handling/normalizer.py:normalize_phone_number`
  - Region hints from `TargetCountryCodes` (defaults to `DE,AT,CH`).
- **Excel “text protect”** for CSV: set
  - `AUGMENTED_PHONE_TEXT_PREFIX="'"`
  so Excel keeps `+49...` as text (not scientific notation, not numeric).

#### Fixing old multi-worker runs (row-order misalignment)
If a previous parallel run merged worker outputs in the wrong order (symptom: phone/source URLs don’t match the original row), regenerate from worker outputs:

```bash
python scripts\regenerate_augmented_from_workers.py --run-dir output_data\<run_id> --input-csv "data\Apollo Shopware Partner Germany 1(Sheet1).csv" --start-row 1 --out-suffix <ANY_SUFFIX>
```

This writes:
- `output_data/<run_id>/phone_extraction_results_<SUFFIX>_merged.csv`
- `output_data/<run_id>/input_augmented_<SUFFIX>.csv`

---

### 3) How to run: full pipeline (sales outreach)

#### Common commands
- **Run everything for a file**:

```bash
python main_pipeline.py -i data\final_80k.csv -s demo
```

- **Run a slice**:

```bash
python main_pipeline.py -i data\final_80k.csv -r 1-200 -s batch1
```

- **Parallel end-to-end (includes sales pitch generation)**:
  - One command spawns multiple worker processes and streams results back to the master.

```bash
python main_pipeline.py -i data\final_80k.csv -r 1-200 -s batch1 --workers 20
```

#### Useful flags
- `--input-profile <name>`: override `INPUT_FILE_PROFILE_NAME` at runtime.
- `--skip-prequalification`: bypass B2B/capacity filtering (runs the rest of the flow).
- `--pitch-from-description`: skip scraping and build pitch from description fields only.
- `--force-phone-extraction`: always run phone retrieval even if an input phone exists.
- `--workers <N>`: run the **full pipeline** in parallel (scrape + summarize + attributes + partner match + sales pitch + optional phone retrieval).
- `-t/--test`: routes Slack notifications to the test channel (if configured).

#### Recommended mode for phone-first outreach (your described use-case)
- Skip B2B/serves-1000 checks, attempt phone retrieval for every row, and generate a pitch **only when a usable phone number exists**:

```bash
python main_pipeline.py -i data\YOUR_FILE.csv -r 1-200 -s run1 --skip-prequalification --force-phone-extraction
```

#### Output layout (full pipeline)
Outputs live under:

- `output_data/<run_id>/SalesOutreachReport_<run_id>.csv`
- `output_data/<run_id>/SalesOutreachReport_<run_id>.xlsx`
- `output_data/<run_id>/SalesOutreachReport_<run_id>_live.csv` (incremental progress)
- `output_data/<run_id>/SalesOutreachReport_<run_id>_live.jsonl` (incremental progress; one JSON object per processed row)
- `output_data/<run_id>/SalesOutreachReport_<run_id>_live_status.json` (progress counters: jobs/rows written)
- `output_data/<run_id>/failed_rows_<run_id>.csv` (row-level failure log)
- `output_data/<run_id>/run_metrics_<run_id>.md`
- `output_data/<run_id>/scraped_content/...` (scraped cleaned-text artifacts)
- `output_data/<run_id>/llm_context/...` and `output_data/<run_id>/llm_requests/...` (LLM prompt/response artifacts)

When running with `--workers > 1`, additional parallel artifacts live under:
- `output_data/<run_id>/jobs/`
  - `jobs/jobXXXX/...` (per-slice artifacts + per-job failures)
  - `jobs/_worker_logs/worker_pid<PID>.log` (per-worker process logs)

#### Monitoring parallel progress
- Watch counters:
  - `output_data/<run_id>/SalesOutreachReport_<run_id>_live_status.json`
- Watch rows as they are produced:
  - `output_data/<run_id>/SalesOutreachReport_<run_id>_live.csv`
  - `output_data/<run_id>/SalesOutreachReport_<run_id>_live.jsonl`

---

### 4) Configuration: what must be set (and what’s typical)

All configuration is centralized in `src/core/config.py` (`AppConfig`). Both pipelines call `dotenv` loading (entrypoints use `load_dotenv(override=False)` so shell env vars can override `.env`, plus `AppConfig`’s own multi-path `.env` discovery).

#### Required
- **`GEMINI_API_KEY`**: required for all LLM tasks (full pipeline + phone-only classification/validation).

#### Typical settings (good defaults)
- **Input**:
  - `INPUT_EXCEL_FILE_PATH=data\...csv`
  - `INPUT_FILE_PROFILE_NAME=<profile>`
  - `ROW_PROCESSING_RANGE=` (empty for all rows) or e.g. `1-500`
- **Output**:
  - `OUTPUT_BASE_DIR=output_data`
- **Logging**:
  - `LOG_LEVEL=INFO`
  - `CONSOLE_LOG_LEVEL=INFO`
- **Phone normalization**:
  - `TARGET_COUNTRY_CODES=DE,CH,AT`
  - `DEFAULT_REGION_CODE=DE`
- **Scraping controls** (sane for most sites):
  - `SCRAPER_MAX_PAGES_PER_DOMAIN=20`
  - `MAX_DEPTH_INTERNAL_LINKS=1`
  - `RESPECT_ROBOTS_TXT=True`

#### German outputs for summary + attributes (industry/USP/etc.)
The full pipeline’s *summary* and *attribute extraction* language is determined by the prompt templates configured in `.env`:
- `PROMPT_PATH_WEBSITE_SUMMARIZER` (company summary text)
- `PROMPT_PATH_ATTRIBUTE_EXTRACTOR` (industry, products/services, USPs, target segments, etc.)

This repo includes German variants:
- `prompts/website_summarizer_prompt_de.txt`
- `prompts/attribute_extractor_prompt_de.txt`

#### HTTP fallback (phone-only scraper)
The phone-only scraper (`src/phone_retrieval/scraper/scraper_logic.py`) can fall back to a lightweight `httpx` GET if Playwright fails or returns unusable HTML. These are controlled by:

- `SCRAPER_HTTP_FALLBACK_ENABLED=True`
- `SCRAPER_HTTP_FALLBACK_TIMEOUT_SECONDS=15`
- `SCRAPER_HTTP_FALLBACK_MAX_BYTES=2000000`
- `SCRAPER_HTTP_FALLBACK_MIN_TEXT_CHARS=200`
- `SCRAPER_HTTP_FALLBACK_BLOCK_KEYWORDS=access denied,captcha,cloudflare,verify you are human,...`

Per-row tracking is written to output columns:
- `HttpFallbackAttempted`, `HttpFallbackUsed`, `HttpFallbackResult`

---

### 5) Input profiles (mapping your CSV columns)

Profiles live in `src/core/config.py` under `AppConfig.INPUT_COLUMN_PROFILES`.

- The phone-only CLI defaults to:
  - `--input-profile apollo_shopware_partner_germany`
- To add a new dataset mapping, add a new profile dict mapping **input column names** → **canonical names** like:
  - `CompanyName`, `GivenURL`, `GivenPhoneNumber` (or `PhoneNumber` depending on pipeline)

Tip: include `_original_phone_column_name` if you want “augmented input” output to update/annotate the correct column.

---

### 6) Key files worth reading (the “spine”)

#### Entrypoints
- `main_pipeline.py`: full pipeline CLI + orchestration.
- `phone_extract.py`: phone-only CLI (supports `--workers` and augmented CSV).
- `scripts/regenerate_augmented_from_workers.py`: regenerate merged+augmented outputs from worker CSVs (no re-scrape).

#### Configuration
- `src/core/config.py`: `AppConfig` (all env vars, input profiles, prompt paths, scraper/LLM tuning).
- `src/core/logging_config.py`: log formatting + file/console routing.
- `requirements.txt`: dependencies (note: `playwright install` required after pip install).

#### Full pipeline core
- `src/data_handling/loader.py`: “smart read” + profile rename + pipeline column initialization.
- `src/processing/url_processor.py`: URL cleanup + TLD probing + validation.
- `src/scraper/scraper_logic.py`: Playwright scraper for the full pipeline (caching, proxy support).
- `src/processing/pipeline_flow.py`: per-row flow + optional phone retrieval + live CSV writer.
- `src/reporting/main_report_orchestrator.py`: writes SalesOutreach report(s) + Slack notify.
- `src/reporting/csv_reporter.py`: flattens `GoldenPartnerMatchOutput` → CSV/XLSX.
- `src/llm_clients/gemini_client.py`: Gemini client wrapper + retries.

#### Phone-only core
- `src/phone_retrieval/processing/pipeline_flow.py`: scrape → regex candidates → LLM chunking → consolidation → outcomes.
- `src/phone_retrieval/scraper/scraper_logic.py`: Playwright scraper + httpx HTML fallback + fallback instrumentation.
- `src/phone_retrieval/extractors/regex_extractor.py`: candidate extraction using `phonenumbers.PhoneNumberMatcher` + snippets.
- `src/phone_retrieval/extractors/llm_extractor.py` + `llm_chunk_processor.py`: LLM classification (Gemini) with chunking.
- `src/phone_retrieval/data_handling/consolidator.py`: dedupe + consolidate candidates into `Top_Number_1..3` etc.
- `src/phone_retrieval/reporting/*`: phone-only reports + metrics markdown.
- `src/phone_retrieval/retrieval_wrapper.py`: adapter so the full pipeline can call phone retrieval.

---

### 7) Debugging & auditing workflow (high-signal)

#### “Where did this phone number come from?”
Use the augmented columns:
- `PhoneSources_Found` (URLs that contained the candidate)
- `HttpFallbackUsed` + `HttpFallbackResult` (whether httpx fallback was used)

Then inspect scraped artifacts in the run folder:
- Phone-only: `output_data/<run_id>/workers/.../scraped_content/<domain>/..._cleaned.txt`
  - If httpx fallback was used, the filename includes `__httpx`.
- Full pipeline: `output_data/<run_id>/scraped_content/<domain>/..._cleaned.txt`

#### URL mismatch / redirect surprises
Both pipelines normalize and may end up scraping a **canonical landed URL** (redirects, DNS fallbacks, TLD probing). Look for:
- Full pipeline: `CanonicalEntryURL` and `ScrapingStatus`
- Phone-only: `CanonicalEntryURL`, `Final_Row_Outcome_Reason`, plus `Top_SourceURL_1`

If you see row-level mismatches in a historical multi-worker phone run, regenerate with `scripts/regenerate_augmented_from_workers.py` (see section 2).

---

### 8) Tests & smoke checks

#### Full pipeline smoke test
- `run_integration_test.py` runs the full pipeline against `data/test_data.csv` and asserts that expected rows appear in the outputs.

#### Phone-only pipeline test
- `phone_extraction_test.py` / `performance_test.py` run scrape → regex → LLM chunk processing using `https://www.exxomove.de/`.
  - Requires `GEMINI_API_KEY` and working Playwright install.

Run:

```bash
pytest -q
```

---

### 9) Known quirks / maintenance notes

- **No `.env.example` existed originally** (and this environment blocks committing dotfiles); use `env.example` in the repo root as the canonical template, and copy it to `.env`.
- **Two scrapers exist**:
  - `src/scraper/*` for the full pipeline
  - `src/phone_retrieval/scraper/*` for phone-only (includes httpx fallback)
- **Full pipeline is single-process** today (no `--workers` for `main_pipeline.py`).
- **Phone-only parallelization** uses multiprocessing; keep worker counts reasonable to avoid:
  - Playwright overhead
  - Gemini rate limiting (`429 ResourceExhausted`)
  - IP-based blocking (consider proxies/caching)

