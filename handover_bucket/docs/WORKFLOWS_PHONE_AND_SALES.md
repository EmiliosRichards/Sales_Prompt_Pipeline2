### Workflows: phone extraction → sales pitch (and caching)

This repo supports **two-stage workflows**:

1) **Phone extraction** (`phone_extract.py`) to find and rank callable numbers (`Top_Number_1..3`, `MainOffice_*`, etc.)
2) **Sales pitch pipeline** (`main_pipeline.py`) to generate partner match + sales pitch **on top of** the phone outputs

This doc explains the recommended run modes, how caching prevents repeated work, and what to expect in output files.

---

### Phone fields you should rely on (high level)

- **Operational call list** (best first calls): `Top_Number_1..3` (+ matching `Top_Type_*`, `Top_SourceURL_*`)
- **Main line backup**: `MainOffice_Number` (+ `MainOffice_Type`, `MainOffice_SourceURL`)
- **Exclude/avoid buckets**:
  - `SuspectedOtherOrgNumbers`: do not call
  - `DeprioritizedNumbers`: callable but not worth calling first
- **Audit**:
  - `LLMPhoneRanking`: full second-stage ranking JSON
  - `LLMPhoneRankingError`: reranker failure/parse error string (when it happens)
  - `LLMContextPath`: prompt/raw/error artifact paths

Important: `Top_Number_*` is now **LLM-reranker only**. If the reranker returns nothing (or fails), Top fields are left blank (no heuristic fallback).

---

### Caching / avoiding repeated work (what exists)

There are **two** different “cache-like” behaviors:

#### 1) Scraped-content reuse (recommended)
If you already have cleaned scrape artifacts (`*_cleaned.txt`), the full scraper can reuse them and skip network scraping.

- Enabled by:
  - CLI: `main_pipeline.py --reuse-scraped-content-from <path>`
  - Env: `REUSE_SCRAPED_CONTENT_IF_AVAILABLE=True` and `SCRAPED_CONTENT_CACHE_DIRS=...`

What to expect:
- `ScrapingStatus` becomes `Success_CacheHit`
- The run should not create lots of new `scraped_content/` output for those domains.

#### 2) Old JSON “cache_dir” (legacy)
`src/scraper/caching.py` implements a separate cache keyed by `(url, company, run_id)`.
This is less useful for cross-run reuse because it includes `run_id` in the key.

---

### Workflow A (recommended): phone_extract → full pipeline (no repeated scraping)

#### Step 1: run phone extraction
Example:

```bash
python phone_extract.py --input-file data\\source_rows_augmented_unique_3k_ce_b2_apol.csv --range -300 --workers 10 --input-profile company_semicolon_phone_extract --suffix ce_b2_apol_phone
```

This produces:
- `output_data/<phone_run>/input_augmented_<phone_run>.csv` (+ `.jsonl`)
- `output_data/<phone_run>/phone_extraction_results_<phone_run>.csv` (+ `.jsonl`)

#### Step 2: run full pipeline **on top** of the phone-augmented file
Key idea: you feed the phone-augmented CSV into the sales pipeline, and you:
- **skip phone retrieval** (don’t redo phones)
- **reuse scraped content** from the phone_extract run (don’t redo scraping)

Example:

```bash
python main_pipeline.py ^
  --input-file output_data\\<phone_run>\\input_augmented_<phone_run>.csv ^
  --input-profile company_semicolon_phone_found ^
  --range -300 ^
  --skip-phone-retrieval ^
  --reuse-scraped-content-from output_data\\<phone_run> ^
  --suffix ce_b2_apol_sales_on_phone
```

Output augmented file from the sales pipeline:
- **Contains the phone columns (Top/MainOffice/LLMPhoneRanking/etc.) and appends sales fields**
  like `sales_pitch`, `matched_golden_partner`, `match_reasoning`, attribute columns, etc.

Notes:
- **Column canonicalization**: the full pipeline applies input-profile renaming, so some raw columns
  may be mapped into canonical names (e.g. `PhoneNumber_Found` → `PhoneNumber` under
  `company_semicolon_phone_found`).
- **Duplicate canonical columns**: phone-augmented files can already contain canonical columns
  (`CompanyName`, `GivenURL`, `Description`). If a profile renames `Company`/`Website`/`Short Description`
  into those same names, pandas would normally create duplicate columns. The loader now **coalesces**
  duplicates (takes the first non-empty value) and **dedupes** to keep the pipeline stable.
  If you want a “no surprises” deliverable, use the stitching workflow below.

---

### Workflow B: use input descriptions instead of website summarization

If your input already has strong text fields (e.g. Apollo):
- `Short Description`
- `Keywords`
- `reasoning`

You can run:
- `--pitch-from-description` to skip website scraping, and use a combined input text blob.
  The pipeline also makes a **small LLM call** to generate a **German ≤100-word summary** for the output `Short German Description` column (human-readable).

Example:

```bash
python main_pipeline.py --input-file data\\source_rows_augmented_unique_3k_ce_b2_apol.csv --input-profile company_semicolon --range -300 --pitch-from-description --skip-prequalification --suffix desc_only
```

What changes:
- `ScrapingStatus` is set to `Used_Description_Only`
- the pipeline uses a combined text blob from `Short Description` + `Keywords` + `reasoning`
  - This combined blob is used for attribute extraction and sales pitch generation context.
  - The output `Short German Description` column becomes the short German summary (when the LLM call succeeds).

---

### Skip-phone behavior and outputs

If you run the full pipeline with `--skip-phone-retrieval`:
- the pipeline does **not** re-run phone scraping/LLM logic.
- it can still populate high-level phone convenience columns like `found_number` and `PhoneNumber_Status`
  based on the existing phone columns already present in the input (or a usable input phone), while keeping
  the detailed phone-extraction columns (`Top_*`, `MainOffice_*`, diagnostics) intact.

The **SalesOutreachReport** still includes phone fields and the pipeline’s `found_number` field,
because that’s the deliverable used for outreach/pitch gating.

---

### Workflow C (recommended for a single “final deliverable”): stitch phone + sales outputs

If you want a final CSV that is exactly:
- **base phone-extraction `input_augmented`** (all phone columns “as-is”), plus
- **sales columns appended**,

use:
- `scripts/stitch_5_30_full_output.py`

Example:

```bash
python scripts\\stitch_5_30_full_output.py ^
  --base-augmented "output_data\\<phone_run>\\input_augmented_<phone_run>.csv" ^
  --sales-report "output_data\\<sales_run>\\SalesOutreachReport_<sales_run>.csv" ^
  --out-dir "output_data\\<timestamp>_STITCHED"
```

Outputs:
- `input_augmented_5_30_STITCHED.csv` (+ `.jsonl`)  ← **use this as your “send to outreach” file**
- `SalesOutreachReport_5_30_STITCHED.csv` (+ `.jsonl`)
- `STITCH_SUMMARY.json` (counts + diagnostics)

This is the safest way to avoid any ambiguity about which columns “won” during profile renaming.

---

### Parallel mode (`--workers > 1`) notes

When you run `main_pipeline.py --workers N`:
- a master process writes:
  - `SalesOutreachReport_<run_id>_live.csv/.jsonl`
  - `input_augmented_<run_id>_live.csv/.jsonl`
- final copies are written at completion, and `_live.*` files may be deleted after finals are confirmed.

The final parallel outputs:
- do **not** contain internal `__meta_*` fields (those are only for live monitoring).

