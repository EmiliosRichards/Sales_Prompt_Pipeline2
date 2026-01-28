### Prompt for another agent (copy/paste)

You are working in a separate “ops scripts” repo. Your job is to build a small orchestrator script that prepares outreach-ready CSVs from the output of the Sales Prompt Pipeline.

You have been given a handover bucket containing docs and a stitched output produced by the Sales Prompt Pipeline project.

#### Input

The canonical input file is:

- `input_augmented_5_30_STITCHED.csv` (semicolon-delimited)

It contains **phone extraction columns + sales generation columns**.

#### Required behavior

Implement a single pipeline script (or orchestrator) that:

1) **Loads** the stitched CSV (preserve delimiter and multiline fields).

2) **Deduplicates companies**:
   - dedupe key: normalized URL (`CanonicalEntryURL` if present else `GivenURL`) + `CompanyName`
   - keep the row that has the best “callable DACH number” (see step 3)

3) **Selects and validates call numbers** (DACH-only):
   - candidate columns in priority order:
     - `Top_Number_1`, `Top_Number_2`, `Top_Number_3`
     - `MainOffice_Number`
     - `PhoneNumber_Found` (fallback)
   - never select:
     - numbers typed as fax (if the associated type contains “fax” / “telefax”)
     - numbers listed in `SuspectedOtherOrgNumbers`
   - normalize to E.164 when possible
   - enforce DACH region: keep only DE/AT/CH numbers
   - **if a row has no DACH number at all, drop it**

4) **Excel text protection**
   - ensure output phone numbers are written with a leading apostrophe `'` so `+49...` stays as text in Excel.

5) **Sales pitch excerpt**
   - input column: `sales_pitch`
   - produce a new operational column (e.g. `sales_pitch_excerpt`) by extracting the portion required for downstream tasks.

6) **Add explicit operational columns at the end**
   - `first_call_number`, `first_call_type`, `first_call_source_url`
   - `first_call_person_name`, `first_call_person_role`, `first_call_person_department`
   - `main_line_backup_number`, `main_line_backup_type`, `main_line_backup_source_url`

Use upstream fields as follows:
- `first_call_*` should mirror `Top_Number_1`/`Top_Type_1`/`Top_SourceURL_1` when that number is DACH-valid.
- Person info comes from `BestPersonContactName/Role/Department/Number` (best-effort).
- Main line backup comes from `MainOffice_Number/Type/SourceURL` (best-effort).

#### Output

Write a final CSV with:
- the original columns (kept intact), plus
- the new operational columns appended at the end.

Also write a small log/summary (counts):
- input rows
- rows dropped by dedupe
- rows dropped by “no DACH phone”
- rows kept

#### Notes

Read these docs from the bucket for exact field meaning:
- `docs/READING_PHONE_EXTRACTION_OUTPUTS.md`
- `docs/PHONE_NUMBER_FIELDS.md`
- `docs/PERSON_ASSOCIATED_NUMBERS.md`

