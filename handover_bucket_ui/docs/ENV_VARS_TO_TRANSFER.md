### Env vars to transfer into the UI project

This system is configured by `src/core/config.py` (`AppConfig`). The entrypoints use `load_dotenv(override=False)` so **shell env wins** over `.env`.

---

### Required (for any LLM functionality)
- **`GEMINI_API_KEY`**: required for all LLM steps.

---

### Golden partner data (required for partner match + pitch)
- **`PATH_TO_GOLDEN_PARTNERS_DATA`**: path to your Golden Partners Excel/CSV.
  - In this repo it defaults to `data/kgs_001_ER47_20250626.xlsx`.

---

### Prompt paths (only needed if you relocate prompts)
If you copy prompts into the UI repo and keep the same paths, you don’t need to set these.

- **`PROMPT_PATH_ATTRIBUTE_EXTRACTOR`** (default `prompts/attribute_extractor_prompt.txt`)
- **`PROMPT_PATH_GERMAN_PARTNER_MATCHING`** (default `prompts/german_partner_matching_prompt.txt`)
- **`PROMPT_PATH_GERMAN_SALES_PITCH_GENERATION`** (default `prompts/german_sales_pitch_generation_prompt.txt`)

For the UI “description-driven” mode (recommended):
- **`PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION`** (default `prompts/german_short_summary_from_description_prompt.txt`)
- **`LLM_MAX_INPUT_CHARS_FOR_DESCRIPTION_DE_SUMMARY`** (default `12000`)
- **`LLM_MAX_TOKENS_DESCRIPTION_DE_SUMMARY`** (default `256`)

---

### Optional: phone extraction (if exposed as a UI toggle)
If you enable phone extraction in the UI backend, you may want:
- **`ENABLE_PHONE_LLM_RERANK`** (default `True`)
- **`PHONE_LLM_RERANK_MAX_CANDIDATES`** (default `25`)
- **`PHONE_LLM_MAX_CANDIDATES_TOTAL`** (default `120`)
- **`TARGET_COUNTRY_CODES`** (default `DE,CH,AT`)
- **`DEFAULT_REGION_CODE`** (default `DE`)

Scraper tuning (optional):
- `SCRAPER_MAX_PAGES_PER_DOMAIN`, `SCRAPER_MIN_SCORE_TO_QUEUE`, `TARGET_LINK_KEYWORDS`, etc.

