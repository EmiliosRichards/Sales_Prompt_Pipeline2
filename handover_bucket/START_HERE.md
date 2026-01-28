### Start here (handover bucket)

This folder is meant to be **copied into another project** where a new agent will implement a downstream “ops prep” pipeline for the Sales Prompt Pipeline outputs.

The upstream system (this repo) produces a **stitched outreach-ready file** that contains:
- **all phone extraction columns** (as-is from `phone_extract.py`), plus
- **all sales generation columns** (from `main_pipeline.py`).

---

### 1) The one output file you should use as input

Use the stitched deliverable:

- **`input_augmented_5_30_STITCHED.csv`** (semicolon-delimited)

Produced at:

- `output_data/20260126_093300_apol4_sales_desc_on_phone_v2_STITCHED/input_augmented_5_30_STITCHED.csv`

Copy that file into your other project and treat it as the canonical input.

Notes:
- The file has **572 rows**. Not every row has sales output (see `artifacts/STITCH_SUMMARY.json`).
- Phone columns are already DACH-focused by upstream, but downstream should still validate/filter for DACH.

---

### 2) Folder layout (what’s in this bucket)

- `START_HERE.md`: this guide
- `PROMPT_FOR_NEW_AGENT.md`: a ready-to-paste prompt to another agent describing exactly what to build
- `docs/`: copies of the key documentation from this repo (phone selection logic, workflows, output conventions)
- `artifacts/STITCH_SUMMARY.json`: counts from the stitch step (how many sales rows matched the phone base)

---

### 3) How to interpret phone numbers in the stitched file (high level)

You will see three groups of phone-related columns:

#### A) Operational call list (ranked)
- `Top_Number_1`, `Top_Type_1`, `Top_SourceURL_1`
- `Top_Number_2`, `Top_Type_2`, `Top_SourceURL_2`
- `Top_Number_3`, `Top_Type_3`, `Top_SourceURL_3`

These are the “best numbers to call” chosen by the phone reranker LLM.

#### B) Main line backup
- `MainOffice_Number`, `MainOffice_Type`, `MainOffice_SourceURL`

This is a generic switchboard / zentrale backup when identified.

#### C) Person-associated info (optional)
- `BestPersonContactName`, `BestPersonContactRole`, `BestPersonContactDepartment`, `BestPersonContactNumber`
- `PersonContacts` (JSON list; best-effort)

Rule of thumb:
- If `BestPersonContactNumber == Top_Number_1`, your best-first call is a specific person.

#### D) Exclusion / deprioritization buckets
- `SuspectedOtherOrgNumbers` (never call)
- `DeprioritizedNumbers` (callable but low-value; keep as last resort)

#### E) Fallback phone
- `PhoneNumber_Found` + `PhoneType_Found` + `PhoneSources_Found`

If Top numbers are blank, `PhoneNumber_Found` may still carry a usable input phone.

---

### 4) What your other project’s pipeline should do (spec)

Input: the stitched CSV above.

Steps (in this order):

1) **Deduplicate rows**
   - Use `CompanyName` and URL (`CanonicalEntryURL` or `GivenURL`) to dedupe companies.
   - Prefer keeping the row that has a usable DACH first-call number (see step 2).

2) **Phone validation + DACH filtering**
   - Inspect candidate phone fields in priority order:
     - `Top_Number_1..3`
     - `MainOffice_Number`
     - `PhoneNumber_Found` (fallback)
   - Exclude anything clearly not callable:
     - types containing `fax` (case-insensitive)
     - anything in `SuspectedOtherOrgNumbers`
   - Validate/normalize to E.164 when possible.
   - **If the company has no DACH number at all → drop the row.**

3) **Text-protect phone outputs for Excel**
   - Add a leading apostrophe `'` when writing output numbers so `+49...` remains text.

4) **Sales pitch text extraction**
   - Use `sales_pitch` as the source text.
   - Create additional operational columns (example):
     - `sales_pitch_excerpt` (short excerpt needed for downstream ops)

5) **Add explicit “first call” + “main line” columns (even if they duplicate upstream fields)**
   - `first_call_number` = the selected number to dial first (from Top list, DACH-valid)
   - `first_call_type`, `first_call_source_url`
   - `first_call_person_name`, `first_call_person_role`, `first_call_person_department`
   - `main_line_backup_number` = `MainOffice_Number` if DACH-valid
   - `main_line_backup_type`, `main_line_backup_source_url`

This makes the final CSV “obviously usable” without understanding all upstream columns.

---

### 5) Mapping to the other project’s existing scripts

Your other project has scripts you mentioned:

- `filter_phone_numbers.py`
  - update it to read from **Top/MainOffice/PhoneNumber_Found** columns (not just one “phone column”)
  - enforce DACH-region filtering
  - apply Excel text prefix `'`

- `single_dedupe.py`
  - dedupe based on `CompanyName` + normalized URL (`CanonicalEntryURL` preferred; fallback to `GivenURL`)

- `extract_pitch_text.py`
  - input column: `sales_pitch`
  - output column(s): your chosen excerpt fields

---

### 6) Deep-dive references (read these first)

- `docs/READING_PHONE_EXTRACTION_OUTPUTS.md`
- `docs/PHONE_NUMBER_FIELDS.md`
- `docs/WORKFLOWS_PHONE_AND_SALES.md`
- `docs/OUTPUT_FILES.md`
- `docs/PERSON_ASSOCIATED_NUMBERS.md`

