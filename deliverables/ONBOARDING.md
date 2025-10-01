### A) Executive Summary
The project is a Python-based sales outreach pipeline that turns a list of companies and URLs into a structured, actionable sales report. It scrapes websites, summarizes what the company does, extracts key attributes, assesses B2B suitability and capacity, and generates a tailored phone sales line by comparing each company to “golden partner” profiles. The pipeline also attempts to find a primary phone number when not provided, using regex and LLM validation on scraped content. It produces a live updating CSV during processing and final CSV/XLSX reports, along with markdown run metrics and row-level failure logs for auditing. Configuration is centralized via environment variables, supporting flexible input formats, scraping depth/limits, LLM model parameters, and Slack notifications. The runtime centers on a CLI entry point (`main_pipeline.py`) that orchestrates scraping (Playwright), text extraction (BeautifulSoup), LLM calls (Google Gemini), and reporting. Outputs are organized under `output_data/<run_id>` with subfolders for LLM artifacts and scraped text for reproducibility and debugging. A performance test and a full integration test are included to validate critical parts of the workflow. The design favors resiliency: it falls back to descriptions when scraping yields no text, logs attrition reasons when steps fail, and saves artifacts for later inspection. Typical use cases include qualifying prospects at scale, generating quick sales pitches, and enriching lead lists with contact numbers. Limitations include reliance on an external LLM API, dynamic website variability, and Playwright/browser requirements.

### B) Architecture Diagram
```mermaid
flowchart TD
  A[Input file (CSV/XLSX)] --> B[main_pipeline.py]
  B --> C[src/utils/helpers.py
  - generate_run_id
  - setup_output_directories]
  B --> D[src/data_handling/loader.py
  - load_and_preprocess_data]
  B --> E[GeminiClient
  src/llm_clients/gemini_client.py]
  B --> F[Golden partners
  src/data_handling/partner_data_handler.py]
  B --> G[execute_pipeline_flow
  src/processing/pipeline_flow.py]

  G --> H[scrape_website
  src/scraper/scraper_logic.py]
  H --> H1[Cleaned text files
  output_data/<run_id>/scraped_content]

  G --> I[LLM: Website summary
  src/extractors/llm_tasks/summarize_task.py]
  G --> J[LLM: Attribute extract
  src/extractors/llm_tasks/extract_attributes_task.py]
  G --> K[LLM: B2B & capacity
  src/extractors/llm_tasks/b2b_capacity_check_task.py]
  G --> L[LLM: Partner match
  src/extractors/llm_tasks/match_partner_task.py]
  G --> M[LLM: Sales pitch
  src/extractors/llm_tasks/generate_sales_pitch_task.py]
  G --> N[Optional: Phone retrieval
  src/phone_retrieval/retrieval_wrapper.py]

  G --> O[Live CSV reporter
  src/reporting/live_csv_reporter.py]
  G --> P[generate_all_reports
  src/reporting/main_report_orchestrator.py]
  P --> Q[SalesOutreachReport CSV/XLSX
  src/reporting/csv_reporter.py]
  P --> R[Row attrition report
  src/reporting/report_generator.py]
  P --> S[Slack notify (optional)
  src/reporting/slack_notifier.py]

  B --> T[write_run_metrics
  src/reporting/metrics_manager.py]
  E -. uses .-> U[Google Gemini API]
  H -. uses .-> V[Playwright + Chromium]
```

### C) Repo Map
- `main_pipeline.py`: CLI entry; orchestrates config, scraping, LLM flow, and reporting.
- `run_integration_test.py`: End-to-end smoke test that sets `.env`, runs pipeline, and asserts outputs.
- `performance_test.py`: Async perf test for scraping, regex extraction, and LLM chunk processing.
- `regenerate_report.py`: Rebuilds a focused final report from saved LLM artifacts and the original input.
- `requirements.txt`: Python dependencies.
- `src/core/config.py`: `AppConfig` loads all env/config, prompt paths, scraping and LLM settings.
- `src/core/schemas.py`: Pydantic models for all structured inputs/outputs (e.g., `GoldenPartnerMatchOutput`).
- `src/core/logging_config.py`: `setup_logging` for console/file logging with rotation.
- `src/data_handling/loader.py`: Reads CSV/Excel with “smart read”, applies input profile, initializes columns.
- `src/data_handling/consolidator.py`: URL canonicalization helper (`get_canonical_base_url`).
- `src/data_handling/partner_data_handler.py`: Golden partner data load/summarize (referenced in `main_pipeline.py`).
- `src/processing/pipeline_flow.py`: Core row loop including URL processing, scraping, LLM calls, phone retrieval, and live reporting.
- `src/scraper/scraper_logic.py`: Playwright-based crawler; saves cleaned text and returns status/canonical URL/summary text.
- `src/extractors/llm_tasks/*.py`: LLM tasks for summary, attribute extraction, b2b check, partner match, and sales pitch.
- `src/llm_clients/gemini_client.py`: Gemini client wrapper with retries.
- `src/reporting/main_report_orchestrator.py`: Orchestrates reports from pipeline outputs.
- `src/reporting/csv_reporter.py`: Writes `SalesOutreachReport_<run_id>.(csv|xlsx)`.
- `src/reporting/report_generator.py`: Writes row attrition and canonical domain summary reports.
- `src/reporting/live_csv_reporter.py`: Appends a live row after each processed input.
- `src/reporting/metrics_manager.py`: Writes markdown run metrics file.
- `src/utils/helpers.py`: Run ID, output dir setup, URL canonical helpers, DataFrame initialization, metrics scaffolding.
- `src/phone_retrieval/*`: Focused phone retrieval pipeline pieces (used when no phone provided).
- `prompts/*.txt`: Prompt templates for all LLM stages.
- `output_data/` (runtime): Per-run outputs under `output_data/<run_id>/`.

### D) Data Flow
1. `main_pipeline.py:main` loads `.env`, builds `AppConfig`, creates a `run_id`, and sets output dirs via `src/utils/helpers.py:setup_output_directories`.
2. Input file path is resolved (`resolve_path`), data is read and normalized by `src/data_handling/loader.py:load_and_preprocess_data` using the active input profile (see `AppConfig.INPUT_COLUMN_PROFILES`).
3. Logging is configured to both console and `output_data/<run_id>/pipeline_run_<run_id>.log` via `src/core/logging_config.py:setup_logging`.
4. `GeminiClient` is initialized; golden partners are loaded/summarized.
5. `src/processing/pipeline_flow.py:execute_pipeline_flow` iterates rows:
   - Validates/processes URL (`src/processing/url_processor.py:process_input_url`).
   - Scrapes site with `src/scraper/scraper_logic.py:scrape_website` (Playwright) → saves cleaned text and returns summary text and canonical URL.
   - If no text, falls back to description; otherwise calls LLM tasks:
     - `summarize_task.generate_website_summary`
     - `extract_attributes_task.extract_detailed_attributes`
     - `b2b_capacity_check_task.check_b2b_and_capacity`
     - `match_partner_task.match_partner` → `generate_sales_pitch_task.generate_sales_pitch`
   - If input lacks phone number, attempts retrieval via `src/phone_retrieval/retrieval_wrapper.py:retrieve_phone_numbers_for_url`.
   - Appends a row to live CSV (`src/reporting/live_csv_reporter.py`).
6. `src/reporting/main_report_orchestrator.py:generate_all_reports` writes the final `SalesOutreachReport` (CSV/XLSX) using `src/reporting/csv_reporter.py:write_sales_outreach_report`, and a `row_attrition_report` when applicable.
7. `src/reporting/metrics_manager.py:write_run_metrics` writes a markdown summary with durations, stats, and failure summaries.

### E) Inputs & Outputs Table
Name | Type | Required? | Where defined | Example
--- | --- | --- | --- | ---
Input file | CSV/XLSX | Yes | `.env` via `AppConfig.input_excel_file_path` | `data/test_data.csv`
Input profile | Enum | No (default `default`) | `AppConfig.INPUT_COLUMN_PROFILES` | `final_80k`
Company URL | String column | Yes per profile | Input file column mapped to `GivenURL` | `https://www.exxomove.de/`
Company Name | String column | Yes per profile | Mapped to `CompanyName` | `Exxomove`
Golden partners | CSV/XLSX | Yes for partner matching | `AppConfig.PATH_TO_GOLDEN_PARTNERS_DATA` | `data/kgs_001_ER47_20250626.xlsx`
LLM prompts | Text files | Yes | `prompts/*.txt` via `AppConfig` paths | `prompts/website_summarizer_prompt.txt`
SalesOutreachReport | CSV/XLSX | Output | `output_data/<run_id>/` | `SalesOutreachReport_<run_id>.csv`
Row attrition report | XLSX | Output (conditional) | `output_data/<run_id>/` | `row_attrition_report_<run_id>.xlsx`
Run metrics | Markdown | Output | `output_data/<run_id>/` | `run_metrics_<run_id>.md`
Failure log | CSV | Output | `output_data/<run_id>/` | `failed_rows_<run_id>.csv`
LLM artifacts | Files | Output | `output_data/<run_id>/llm_context` and `llm_requests` | Saved prompts/responses per row
Scraped text | Files | Output | `output_data/<run_id>/scraped_content` | Cleaned text per page

### F) Environment & Config Table
Name | Default/Example | Used in file(s) | Notes/Secrets
--- | --- | --- | ---
GEMINI_API_KEY | `<API_KEY>` | `src/llm_clients/gemini_client.py` | Required for LLM calls; secret
LLM_MODEL_NAME | `gemini-1.5-pro-latest` | LLM tasks, `GeminiClient` | Override with compatible model
INPUT_EXCEL_FILE_PATH | `data_to_be_inputed.xlsx` | `src/core/config.py`, `main_pipeline.py` | Can be CSV or Excel; CLI `-i` overrides
INPUT_FILE_PROFILE_NAME | `default` | Loader/pipeline | Profiles in `AppConfig.INPUT_COLUMN_PROFILES` (e.g., `final_80k`, `lean_formatted`)
ROW_PROCESSING_RANGE | `` (all) | Loader/Config | Formats: `N-M`, `N-`, `-M`, `N`
OUTPUT_BASE_DIR | `output_data` | Helpers/Reports | Run outputs under `<base>/<run_id>`
LOG_LEVEL | `INFO` | Logging | File log level
CONSOLE_LOG_LEVEL | `WARNING` | Logging | Console log level
PATH_TO_GOLDEN_PARTNERS_DATA | `data/kgs_001_ER47_20250626.xlsx` | Main pipeline | Required for partner match and pitch
SALES_PROMPT_LANGUAGE | `en` | LLM tasks | German prompts supported via `de`
TARGET_COUNTRY_CODES | `DE,CH,AT` | Phone number normalization | Affects country inference
SCRAPER_MAX_PAGES_PER_DOMAIN | `20` | Scraper | Set `0` for no limit; see other `SCRAPER_*` vars in `config.py`
CACHING_ENABLED | `True` | Scraper | Uses `cache/` to avoid repeat fetches
PROXY_ENABLED | `False` | Scraper | With `PROXY_LIST`, `PROXY_ROTATION_STRATEGY`
ENABLE_SLACK_NOTIFICATIONS | `False` | Slack notifier | Requires `SLACK_BOT_TOKEN`, `SLACK_CHANNEL_ID`
SLACK_BOT_TOKEN | `<xoxb-...>` | Slack notifier | Secret
SLACK_CHANNEL_ID | `<Cxxxx>` | Slack notifier | Optional if notifications disabled

(See `src/core/config.py` for many additional scraper and LLM tuning parameters.)

### G) 5-minute Quickstart
- Install Python 3.10+ and Git. Create/activate a virtualenv.
```bash
pip install -r requirements.txt
python -m playwright install
```
- Create `.env` in the project root (minimal):
```bash
# Required
GEMINI_API_KEY=<API_KEY>
INPUT_EXCEL_FILE_PATH=data/test_data.csv
INPUT_FILE_PROFILE_NAME=final_80k
# Optional (better logs)
LOG_LEVEL=INFO
CONSOLE_LOG_LEVEL=INFO
# Optional Slack
ENABLE_SLACK_NOTIFICATIONS=False
SLACK_BOT_TOKEN=
SLACK_CHANNEL_ID=
```
- Run the pipeline:
```bash
python main_pipeline.py -i data/test_data.csv -r 1-50 -s demo
```
- Smoke-test (full integration):
```bash
python run_integration_test.py
```
- Outputs will be under `output_data/<run_id>/` with live and final reports.

### H) Runbook: Debugging & Common Issues
- LLM initialization fails (ValueError: GEMINI_API_KEY):
  - Set `GEMINI_API_KEY` in `.env` or environment. Verify no quotes or spaces.
- Playwright errors / browser not found:
  - Run `python -m playwright install`. On CI/Linux, consider `--with-deps`.
- Robots disallowed / HTTP errors:
  - See `ScrapingStatus` in outputs. Review `output_data/<run_id>/pipeline_run_<run_id>.log`.
- No text available; used fallback:
  - The pipeline sets status `Used_Fallback_Description` and skips LLM if both scrape and fallback empty.
- Slow runs / timeouts:
  - Tune `SCRAPER_*` timeouts/limits, enable caching, reduce `LLM_MAX_CHUNKS_PER_URL`.
- Slack upload skipped:
  - Ensure `ENABLE_SLACK_NOTIFICATIONS=True` and both `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` are set.
- Where to look:
  - Live CSV at `output_data/<run_id>/SalesOutreachReport_<run_id>_live.csv`.
  - LLM prompts/responses in `llm_context/` and request payloads in `llm_requests/`.
  - Final CSV/XLSX in `output_data/<run_id>/`.
  - Failure details in `failed_rows_<run_id>.csv` and `row_attrition_report_<run_id>.xlsx`.

### I) Glossary
- **Golden Partner**: An internal reference profile used to evaluate similarity and craft sales lines.
- **Canonical URL**: Normalized base domain used to group scraped paths.
- **Attrition**: Rows that failed to produce outputs (e.g., invalid URL, scraping/LLM failures).
- **LLM Artifacts**: Saved prompts, responses, and payloads for reproducibility.
- **Run ID**: Timestamp-based unique identifier appended to outputs.
- **Live CSV**: Incrementally updated CSV reflecting progress per processed row.
- **Profiles (Input)**: Named mappings for input columns to canonical keys (see `AppConfig.INPUT_COLUMN_PROFILES`).

### J) Quality & Improvement Backlog
- Quick Wins
  - Add `.env.example` covering required/optional variables with comments.
  - Provide a small sample input under `data/` with each supported profile.
  - Pin dependency versions in `requirements.txt` to reduce drift; add `playwright` install note to README.
  - Add CLI `--dry-run` to process N rows without LLM calls for faster smoke tests.
  - Improve error messages for prompt file missing paths with remediation hints.
- Risks / Tech Debt (Next Steps)
  - Duplicate pipelines under `src/` and `src/phone_retrieval/`: consolidate phone retrieval logic to a single path.
  - Heavy reliance on external LLM and Playwright: add fallbacks/mocks for offline testing.
  - Scraper variability: expand link scoring config to be data-driven and add more unit tests.
  - Secrets handling: introduce `.env.example`, CI secrets, and secret scanning.
  - Robustness for very large inputs: batch processing and checkpointing.
- Guardrail Tests
  - Unit: URL processing, canonicalization, JSON extraction from LLM text, prompt loading.
  - Unit: Schema validation for `WebsiteTextSummary`, `DetailedCompanyAttributes`, `GoldenPartnerMatchOutput`.
  - Integration: Single-row pipeline with a deterministic site; verify all report files exist.
  - Regression: Ensure `Used_Fallback_Description` path emits expected attrition entries.

### K) FAQ
- How do I run only a subset of rows?
  - Use `-r` like `-r 1-100`, `-r 500-`, or `-r -200`.
- Can I use CSV instead of Excel?
  - Yes. The loader detects extension and reads appropriately.
- Where do I change which input columns are used?
  - Update `AppConfig.INPUT_COLUMN_PROFILES` in `src/core/config.py` and set `INPUT_FILE_PROFILE_NAME`.
- How do I switch prompts or language?
  - Point `PROMPT_PATH_*` env vars to files in `prompts/` and set `SALES_PROMPT_LANGUAGE` (`en` or `de`).
- The scraper is slow. What can I do?
  - Reduce `SCRAPER_MAX_PAGES_PER_DOMAIN`, increase thresholds, enable caching, and lower `LLM_MAX_CHUNKS_PER_URL`.
- Where are logs and artifacts?
  - Logs: `output_data/<run_id>/pipeline_run_<run_id>.log`. Artifacts: `llm_context/` and `llm_requests/` under the run dir.
- How does phone retrieval work when number is missing?
  - It scrapes, finds regex candidates, validates with LLM, consolidates, then picks a primary/secondary (`src/phone_retrieval/*`).
- Can I post results to Slack?
  - Yes; set `ENABLE_SLACK_NOTIFICATIONS=True` and provide `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID`.
- What’s the minimal required configuration?
  - `GEMINI_API_KEY`, `INPUT_EXCEL_FILE_PATH`, and an appropriate `INPUT_FILE_PROFILE_NAME`.
- How do I regenerate a focused final report from artifacts?
  - Run `python regenerate_report.py <run_id> <input_file>`.

### References to Key Symbols
- `src/processing/pipeline_flow.py:execute_pipeline_flow`
- `src/scraper/scraper_logic.py:scrape_website`
- `src/extractors/llm_tasks/summarize_task.py:generate_website_summary`
- `src/extractors/llm_tasks/extract_attributes_task.py:extract_detailed_attributes`
- `src/reporting/main_report_orchestrator.py:generate_all_reports`
- `src/reporting/csv_reporter.py:write_sales_outreach_report`
- `src/reporting/metrics_manager.py:write_run_metrics`
- `src/utils/helpers.py:setup_output_directories, generate_run_id, get_input_canonical_url`

### L) Findings: Flow Gaps, Errors, and Cohesion Assessment
- Critical gaps in A→Z flow
  - Attrition list not populated: `src/processing/pipeline_flow.py:execute_pipeline_flow` logs failures via `log_row_failure(...)` but never appends detailed entries to `attrition_data_list`. As a result, `row_attrition_report_<run_id>.xlsx` will often be empty and `run_metrics` attrition analysis undercounts. Suggestion: create a helper to build a dict for each failure (row id, company, URL, stage, reason, canonical domain, timestamps, duplicate flags), append to `attrition_data_list`, and pass it through to `generate_all_reports` and `write_run_metrics`.
  - Metrics key mismatch: The pipeline records `tasks["pipeline_main_loop_duration_seconds"]`, but `src/reporting/metrics_manager.py:write_run_metrics` expects `pass1_main_loop_duration_seconds` for the “Average Pass 1 Main Loop Duration.” Rename the metric in the pipeline or read either key in metrics writer to ensure averages are shown.
  - Canonical domain summary not generated: `src/reporting/report_generator.py:write_canonical_domain_summary_report` exists but is never called in `src/reporting/main_report_orchestrator.py:generate_all_reports`. If domain-level insights are desired, call it with `canonical_domain_journey_data` and record counts in `run_metrics`.
  - Environment/config naming inconsistencies: In `src/scraper/scraper_logic.py` the code checks `SCRAPER_PAGES_FOR_SUMMARY_COUNT`, while `AppConfig` defines `scraper_pages_for_summary_count`. Because of the case mismatch, the default of 3 is always used and the env var cannot override. Align the name (prefer the existing AppConfig attribute) or add a fallback to `getattr(config_instance, 'scraper_pages_for_summary_count', 3)`.
  - Phone retrieval pipeline duplication: There are two parallel areas (`src/*` and `src/phone_retrieval/*`) with similar responsibilities (LLM client, pipeline, reporting). `retrieve_phone_numbers_for_url` calls the `src/phone_retrieval` pipeline with a different interface than the main pipeline. This increases drift risk. Suggest extracting a thin, shared “phone retrieval service” interface and deleting or deprecating one of the duplicates.

- Potential failures and robustness concerns
  - LLM client compat risk: `src/llm_clients/gemini_client.py` uses `from google.generativeai.client import configure` and `from google.generativeai.generative_models import GenerativeModel`. These imports are version-sensitive. Pin `google-generativeai` to a tested version (and document it) or adapt to the stable API (`import google.generativeai as genai; genai.configure(...); genai.GenerativeModel(...)`).
  - Async in sync loop: `execute_pipeline_flow` calls `asyncio.run(scrape_website(...))`. This fails if an event loop is already running (e.g., future async integration). It’s fine for the current CLI but consider isolating scraping behind a small sync wrapper (running a dedicated loop or trio) to be future-proof.
  - `httpx` with `verify=False`: In `scraper_logic.py` HTTPS verification is disabled. This is pragmatic for scraping but a security risk. Consider making it configurable and defaulting to verification on.
  - Integration test assumptions: `run_integration_test.py` expects `data/test_data.csv` to exist and that the pipeline will create `output_data/`. If `data/` or the CSV is missing, it will fail before the pipeline runs. Add a small fixture CSV to the repo or guard with a clearer error.
  - Slack notifier branding: Message header says “Shop Confirmation System Pipeline Run Complete,” which seems out of date for this project. Cosmetic, but may confuse users.

- Smaller correctness/consistency nits
  - Live CSV header vs. row fields: The header includes both `description` and the original `Description`. Ensure the intended final column is consistent (you currently map the LLM summary back into `description`).
  - Metrics counts: Some counters in `run_metrics` (e.g., sites processed for LLM) are not consistently incremented around each LLM call. If you rely on these in `write_run_metrics`, ensure they are updated.
  - Prompt paths: `AppConfig.get_clean_path` has careful normalization; still, on Windows paths with OneDrive/space characters, consider logging resolved prompt paths at startup for quick diagnosis.

- Overall cohesion assessment
  - The architecture is solid and thoughtfully layered: configuration (`AppConfig`), schemas, scraping, LLM tasks, and reporting are cleanly separated. Logging and artifact capture are strong, making the system auditable and debuggable. The live CSV and metrics give good operational visibility.
  - The main risks are duplication between the general pipeline and the `phone_retrieval` subtree, a few naming/config inconsistencies, and a couple of metrics/reporting gaps. None are fundamental design flaws; they’re fixable hardening tasks.
  - With the noted adjustments (populate attrition, align metrics keys, call the domain summary if valuable, pin LLM client version, and unify the phone retrieval path), the project reads as production-ready for controlled environments.
