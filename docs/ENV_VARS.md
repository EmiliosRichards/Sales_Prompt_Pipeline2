### Environment variables (high-signal)

All config is defined in `src/core/config.py` (`AppConfig`). Entry points load `.env` with `load_dotenv(override=False)` so **shell env vars win** over `.env`.

---

### Required
- **`GEMINI_API_KEY`**: required for any LLM steps (phone classification + full pipeline summarization/attributes/match/pitch).

---

### Pitch-from-description: German short summary (human-readable)
When running `main_pipeline.py --pitch-from-description`, the pipeline can still run a **small** LLM call to generate a **German ≤100-word summary** for the output `Short German Description` column.

- **`PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION`** (default `prompts/german_short_summary_from_description_prompt.txt`): prompt template path.
- **`LLM_MAX_INPUT_CHARS_FOR_DESCRIPTION_DE_SUMMARY`** (default `12000`): max input chars sent to this step.
- **`LLM_MAX_TOKENS_DESCRIPTION_DE_SUMMARY`** (default `256`): max output tokens for this step.

---

### Phone-only and phone retrieval (quality/cost)
- **`FULL_LLM_PROVIDER`** (default `gemini`): provider for the full sales pipeline LLM steps. Supported values: `gemini`, `openai`.
- **`PHONE_LLM_PROVIDER`** (default `gemini`): provider for the phone-only pipeline LLM steps. Supported values: `gemini`, `openai`.
- **`PHONE_LLM_MAX_CANDIDATES_TOTAL`** (default `120`): cap candidates sent to the phone LLM per pathful canonical URL.
- **`PHONE_LLM_MIN_CANDIDATE_SCORE`** (default `0`): drop very low-signal regex candidates before calling the phone LLM (fallback: if this would drop everything, we still send the top-ranked candidates). Keep `0` if you want maximum recall.
- **`PHONE_LLM_PREFER_URL_PATH_KEYWORDS`**: comma-separated keywords that increase candidate priority (default includes `kontakt,impressum,...`).
- **`PHONE_LLM_PREFER_SNIPPET_KEYWORDS`**: comma-separated snippet keywords that increase candidate priority (default includes `tel,telefon,zentrale,...`).
- **`ENABLE_PHONE_LLM_RERANK`** (default `True`): run a second LLM call that produces the **only** operational call list (`Top_Number_1..3`) and optionally a `MainOffice_*` backup. If the reranker is disabled, fails, or returns no ranked numbers, the `Top_*` fields remain blank (no heuristic fallback).
- **`PHONE_LLM_RERANK_MAX_CANDIDATES`** (default `25`): maximum numbers to send to the second-stage ranking LLM per canonical base domain.
- **`PROVIDER_MAX_INFLIGHT_DEFAULT`** (default `6`): default provider concurrency ceiling used to clamp `--workers`.
- **`PROVIDER_MAX_INFLIGHT_OPENAI`** (default: same as `PROVIDER_MAX_INFLIGHT_DEFAULT`, currently `6`): OpenAI-specific concurrency ceiling.
- **`PROVIDER_MAX_INFLIGHT_GEMINI`** (default: same as `PROVIDER_MAX_INFLIGHT_DEFAULT`, currently `6`): Gemini-specific concurrency ceiling.
- **`PROVIDER_BACKPRESSURE_COOLDOWN_SECONDS`** (default `30`): cooldown duration after repeated provider throttling/service failures.

Current recommended phone-only configuration:
- `PHONE_LLM_PROVIDER="openai"`
- `OPENAI_MODEL_NAME="gpt-5.4-mini-2026-03-17"`
- `OPENAI_SERVICE_TIER="flex"`

When `PHONE_LLM_PROVIDER=openai`, these additional variables matter:
- **`OPENAI_API_KEY`**: required for OpenAI-backed phone extraction/reranking/homepage context.
- **`OPENAI_MODEL_NAME`** (default `gpt-5.1-2025-11-13`): OpenAI model used for phone-only structured extraction. The current recommended production setting is `gpt-5.4-mini-2026-03-17`.
- **`OPENAI_MODEL_NAME_SALES_INSIGHTS`** (default: same as `OPENAI_MODEL_NAME`): optional override for full-pipeline sales pitch generation when `FULL_LLM_PROVIDER=openai`.
- **`OPENAI_SERVICE_TIER`** (default `flex`): service tier passed to the Responses API.
- **`OPENAI_TIMEOUT_SECONDS`** (default `900`): per-request timeout.
- **`OPENAI_FLEX_MAX_RETRIES`** (default `5`): retry count for transient/flex failures.
- **`OPENAI_FLEX_FALLBACK_TO_AUTO`** (default `True`): if flex still fails, retry once on `auto`.
- **`OPENAI_PROMPT_CACHE`** / **`OPENAI_PROMPT_CACHE_RETENTION`**: enable prompt caching for repeated phone prompts. The client now uses a short digest-based cache key so longer model names still work with the Responses API cache-key length limits.
- **`OPENAI_REASONING_EFFORT`**: optional Responses API reasoning setting; usually leave blank for this phone workflow.

---

### Scrape reuse (skip Playwright when you already have `*_cleaned.txt`)
- **`REUSE_SCRAPED_CONTENT_IF_AVAILABLE`** (`True/False`): enable reuse.
- **`SCRAPED_CONTENT_CACHE_DIRS`**: comma/semicolon-separated list of `scraped_content` directories to read from.
- **`SCRAPED_CONTENT_CACHE_MIN_CHARS`** (default `500`): minimum text length to treat cached text as usable (full pipeline).

Notes:
- `main_pipeline.py --reuse-scraped-content-from <run_dir_or_scraped_content_dir>` enables this at runtime (no `.env` edit needed).
- When reuse triggers, rows commonly show `ScrapingStatus=Success_CacheHit`.

---

### Phone results cache (skip regex/LLM when you already extracted phones for a domain)
- **`REUSE_PHONE_RESULTS_IF_AVAILABLE`** (`True/False`): enable reuse of cached consolidated results per canonical base.
- **`PHONE_RESULTS_CACHE_DIR`** (default `cache/phone_results_cache`): where cached per-domain JSON files are stored.

Note: this is a **per-domain reuse accelerator**, not a “resume a stalled parallel run” feature. It helps avoid repeating
work across runs for the same canonical base domain, but it does not automatically fill missing rows in a stuck run.

---

### Full pipeline phone behavior
- **`ENABLE_PHONE_RETRIEVAL_IN_FULL_PIPELINE`** (`True/False`): if `False`, the full pipeline never runs phone retrieval; it will still use input phones for pitch gating.
- **`FORCE_PHONE_EXTRACTION`** (`True/False`): force phone retrieval even when an input phone exists.

---

### Partner matching retrieval and reranking
- **`MAX_GOLDEN_PARTNERS_IN_PROMPT`** (default `10`): hard cap on how many fused shortlist candidates are shown to the reranker prompt.
- **`PARTNER_MATCH_SPARSE_TOP_K`** (default `15`): how many candidates survive the field-aware lexical retrieval channel before fusion.
- **`PARTNER_MATCH_DENSE_TOP_K`** (default `15`): how many candidates survive the local dense retrieval channel before fusion.
- **`PARTNER_MATCH_FUSED_TOP_K`** (default `10`): how many fused candidates are passed into the final LLM reranker.
- **`PARTNER_MATCH_RRF_K`** (default `60`): Reciprocal Rank Fusion smoothing constant used when combining sparse and dense retrieval ranks.

Notes:
- Retrieval now uses a clean partner profile built from audience, products/services, USP, industry, and business model fields.
- Broad narrative notes are excluded from retrieval scoring.
- The reranker prompt receives stable `partner_id` values plus evidence-oriented fields, and runtime outputs now expose `match_confidence`, `match_overlap_type`, evidence columns, runner-up partner IDs, and `match_acceptance_reason`.

---

### CSV writing quality-of-life
- **`AUGMENTED_PHONE_TEXT_PREFIX`**: optional prefix for augmented CSV phones, e.g. set to `'` to keep `+49...` as text in Excel.

