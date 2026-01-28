### How to read `input_augmented_...csv` for “best first call + main line backup”

Think of the phone-related columns as **three buckets**:

#### 1) **Primary call list (what to dial first)**
Use these in order:

- **`Top_Number_1` / `Top_Type_1` / `Top_SourceURL_1`**
- **`Top_Number_2` / `Top_Type_2` / `Top_SourceURL_2`**
- **`Top_Number_3` / `Top_Type_3` / `Top_SourceURL_3`**

These are the **LLM-ranked best callable business numbers**. Fax / Non‑Business are excluded, and “low value” numbers should not be placed here.

If you want to show *who* you’re calling (when it’s a person/direct dial), also read:
- **`BestPersonContactName` / `BestPersonContactRole` / `BestPersonContactDepartment` / `BestPersonContactNumber`**
  - If `BestPersonContactNumber == Top_Number_1`, then your best-first call is a specific person.

#### 2) **Main line backup (switchboard / zentrale)**
Use:
- **`MainOffice_Number` / `MainOffice_Type` / `MainOffice_SourceURL`**

This is your “if direct dial fails, call the front desk” backup. It can be the same as `Top_Number_1` or different.

#### 3) **Numbers to exclude / avoid for first call**
Use these as filters:
- **`SuspectedOtherOrgNumbers`**: *never call* (belongs to another org/entity).
- **`DeprioritizedNumbers`**: callable but **not worth calling first** (keep as last-resort follow-ups).

### Practical selection logic (per row)

1) Build an **exclude set** from:
   - all numbers in `SuspectedOtherOrgNumbers`
   - (optionally) all numbers in `DeprioritizedNumbers` for “first-call only”

2) Pick **best first call**:
   - take the first non-empty of `Top_Number_1`, then `Top_Number_2`, then `Top_Number_3`
   - skip anything in the exclude set

3) Pick **main line backup**:
   - use `MainOffice_Number` if present and not excluded
   - if it’s missing, a reasonable fallback is “the first Top number whose `Top_Type_*` looks like `Main Office`”

4) If **all Top numbers are blank**:
   - use **`PhoneNumber_Found`** as the fallback “call now” number
   - check **`PhoneType_Found`**:
     - if it’s `"Input"`, that means it came from the input `Company Phone` (not from website extraction)
   - important: `PhoneNumber_Found` may be populated even when scraping fails (it can carry forward the best usable input phone).

### Notes on auditing / debugging
- **`LLMPhoneRankingError`**: tells you if the reranker failed for that row.
- **`LLMContextPath`**: points to saved artifacts (reranker prompt/raw/error) for audit.
- For deep debugging, prefer the **JSONL** outputs (`input_augmented_...jsonl` / `phone_extraction_results_...jsonl`) because nested fields are easier to parse reliably than in CSV.

---

### Using these phone outputs in the sales pitch pipeline

Recommended workflow:
- Run `phone_extract.py` first to populate `Top_Number_*` / `MainOffice_*` etc.
- Then run `main_pipeline.py` **on the phone-augmented CSV** with `--skip-phone-retrieval`
  so the pipeline appends sales columns without redoing phone retrieval.

See `docs/WORKFLOWS_PHONE_AND_SALES.md` for concrete commands and caching options.

---

### If you want a single “phone + sales” deliverable

After you run the sales pipeline, the safest “final file” for outreach is typically the **stitched** augmented CSV:
- `output_data/<...>_STITCHED/input_augmented_5_30_STITCHED.csv`

This keeps the phone extraction columns “as-is” (from the phone run) and appends sales fields from the sales report.

