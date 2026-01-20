### Environment variables (high-signal)

All config is defined in `src/core/config.py` (`AppConfig`). Entry points load `.env` with `load_dotenv(override=False)` so **shell env vars win** over `.env`.

---

### Required
- **`GEMINI_API_KEY`**: required for any LLM steps (phone classification + full pipeline summarization/attributes/match/pitch).

---

### Phone-only and phone retrieval (quality/cost)
- **`PHONE_LLM_MAX_CANDIDATES_TOTAL`** (default `120`): cap candidates sent to the phone LLM per pathful canonical URL.
- **`PHONE_LLM_PREFER_URL_PATH_KEYWORDS`**: comma-separated keywords that increase candidate priority (default includes `kontakt,impressum,...`).
- **`PHONE_LLM_PREFER_SNIPPET_KEYWORDS`**: comma-separated snippet keywords that increase candidate priority (default includes `tel,telefon,zentrale,...`).

---

### Scrape reuse (skip Playwright when you already have `*_cleaned.txt`)
- **`REUSE_SCRAPED_CONTENT_IF_AVAILABLE`** (`True/False`): enable reuse.
- **`SCRAPED_CONTENT_CACHE_DIRS`**: comma/semicolon-separated list of `scraped_content` directories to read from.
- **`SCRAPED_CONTENT_CACHE_MIN_CHARS`** (default `500`): minimum text length to treat cached text as usable (full pipeline).

---

### Phone results cache (skip regex/LLM when you already extracted phones for a domain)
- **`REUSE_PHONE_RESULTS_IF_AVAILABLE`** (`True/False`): enable reuse of cached consolidated results per canonical base.
- **`PHONE_RESULTS_CACHE_DIR`** (default `cache/phone_results_cache`): where cached per-domain JSON files are stored.

---

### Full pipeline phone behavior
- **`ENABLE_PHONE_RETRIEVAL_IN_FULL_PIPELINE`** (`True/False`): if `False`, the full pipeline never runs phone retrieval; it will still use input phones for pitch gating.
- **`FORCE_PHONE_EXTRACTION`** (`True/False`): force phone retrieval even when an input phone exists.

---

### CSV writing quality-of-life
- **`AUGMENTED_PHONE_TEXT_PREFIX`**: optional prefix for augmented CSV phones, e.g. set to `'` to keep `+49...` as text in Excel.

