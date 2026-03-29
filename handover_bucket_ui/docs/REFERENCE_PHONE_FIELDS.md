### Reference: phone fields (what the UI should show)

If you enable phone extraction in the UI, the key outputs to display are:

- **Top callable numbers (best first calls)**:
  - `Top_Number_1..3`
  - `Top_Type_1..3`
  - `Top_SourceURL_1..3`

- **Main line backup (keep even if a direct dial outranks it)**:
  - `MainOffice_Number`
  - `MainOffice_Type`
  - `MainOffice_SourceURL`

- **Do-not-call / avoid**
  - `SuspectedOtherOrgNumbers`: do not call
  - `DeprioritizedNumbers`: callable but low value (try last)
  - Fax numbers should never be selected as callable Top numbers.

- **Audit / debugging**
  - `LLMPhoneRanking` (second-stage reranker output JSON)
  - `LLMPhoneRankingError` (when ranking fails)

