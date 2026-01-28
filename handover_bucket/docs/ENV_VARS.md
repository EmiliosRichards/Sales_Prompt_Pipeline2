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
- **`PHONE_LLM_MAX_CANDIDATES_TOTAL`** (default `120`): cap candidates sent to the phone LLM per pathful canonical URL.
- **`PHONE_LLM_MIN_CANDIDATE_SCORE`** (default `0`): drop very low-signal regex candidates before calling the phone LLM (fallback: if this would drop everything, we still send the top-ranked candidates). Keep `0` if you want maximum recall.
- **`PHONE_LLM_PREFER_URL_PATH_KEYWORDS`**: comma-separated keywords that increase candidate priority (default includes `kontakt,impressum,...`).
- **`PHONE_LLM_PREFER_SNIPPET_KEYWORDS`**: comma-separated snippet keywords that increase candidate priority (default includes `tel,telefon,zentrale,...`).
- **`ENABLE_PHONE_LLM_RERANK`** (default `True`): run a second LLM call that produces the **only** operational call list (`Top_Number_1..3`) and optionally a `MainOffice_*` backup. If the reranker is disabled, fails, or returns no ranked numbers, the `Top_*` fields remain blank (no heuristic fallback).
- **`PHONE_LLM_RERANK_MAX_CANDIDATES`** (default `25`): maximum numbers to send to the second-stage ranking LLM per canonical base domain.

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

### CSV writing quality-of-life
- **`AUGMENTED_PHONE_TEXT_PREFIX`**: optional prefix for augmented CSV phones, e.g. set to `'` to keep `+49...` as text in Excel.

