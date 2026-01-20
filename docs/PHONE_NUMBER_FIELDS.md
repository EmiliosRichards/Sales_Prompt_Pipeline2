### Phone number fields (what they mean, what’s redundant)

This pipeline has to support multiple “views” of phone numbers:
- **Input phones** (already in your data)
- **Extracted phones** (from web scraping + regex + LLM classification)
- **Operational call list** (what you dial)
- **Diagnostics** (why the system chose what it chose)

---

### Input phone columns (common)

- **`Company Phone`**: a raw phone column commonly present in Apollo exports.
- **`GivenPhoneNumber`**: canonical name used by the phone-only pipeline (mapped by input profiles).
- **`PhoneNumber`**: canonical name used by the full pipeline (after phone_extract has produced `PhoneNumber_Found`).

---

### Output phone selection columns (phone-only and/or full pipeline)

#### “Top” numbers (operational)
- **`Top_Number_1..3`**: the system’s ranked “best numbers to call”.
- **`Top_Type_1..3`**: comma-separated set of types seen for that number across sources (e.g., `Main Line`, `Sales`, `Fax`).
- **`Top_SourceURL_1..3`**: comma-separated source URLs where that number was found.

If you only care about dialing: use `Top_Number_1` (fallback to `Top_Number_2`, then `Top_Number_3`).

#### “Primary / Secondary” numbers (diagnostics + backward compatibility)
- **`Primary_Number_1`**: best number whose **classification == Primary**
- **`Secondary_Number_1..2`**: best numbers whose **classification == Secondary**

These are derived from the same consolidated pool as `Top_*`. They can look redundant, but are helpful for:
- debugging why `Top_Number_1` is a “direct dial” sometimes,
- keeping compatibility with older downstream templates expecting Primary/Secondary columns.

---

### Full pipeline-specific columns

- **`found_number`**: the full pipeline’s “number to call” used for pitch gating.
  - If phone retrieval ran, it’s derived from retrieved/consolidated numbers.
  - If phone retrieval was skipped, it can be copied from a usable input number.

- **`PhoneNumber_Status`**: high-level explanation (e.g. `Found_Primary`, `Found_Secondary`, `Provided_In_Input_CompanyPhone`, `Skipped_Phone_Retrieval`).

---

### Debug / trace columns (why the system chose what it chose)

These are most useful in JSONL (CSV uses JSON strings).

- **`RegexCandidateSnippets`**: candidate items extracted before the phone LLM (number + snippet + URL).
- **`LLMExtractedNumbers`**: phone LLM output per candidate (validity/type/classification).
- **`LLMContextPath`**: paths to saved request/response artifacts for auditability.
- **`BestMatchedPhoneNumbers`**: the top-ranked numbers (often aligns with `Top_Number_1..3`).
- **`OtherRelevantNumbers`**: additional consolidated numbers not in the top list.
- **`ConfidenceScore`**: heuristic confidence (higher when the best number is classified Primary).

---

### What’s redundant?

If you want to simplify downstream:
- Treat **`Top_Number_1..3` as the canonical call list**.
- Keep **`Primary_/Secondary_` only if you need compatibility/debugging**.

We keep both today because existing templates and prior runs referenced Primary/Secondary, while operational dialing needs Top_*.

