### Person-associated numbers (direct dials): current behavior

The phone extraction system can (optionally) link a number to a specific person, e.g.:
“Max Mustermann (Geschäftsführer) +49 …”.

This is **best-effort**: we only record person data when the snippet strongly supports it.

---

### Where person data comes from

#### Stage A: regex candidates + snippets
We first extract phone-like candidates from scraped text and keep a small snippet excerpt around each candidate.

#### Stage B: phone LLM classification
The phone LLM may add optional fields when the snippet clearly indicates a person:
- `associated_person_name`
- `associated_person_role`
- `associated_person_department`
- `is_direct_dial`

These fields are stored for auditability alongside the raw phone output.

---

### Where it shows up in outputs

You’ll see person-linked data in two places:

#### 1) Full per-candidate trace (audit)
- **`LLMExtractedNumbers`**: list of extracted numbers with optional `associated_person_*` fields (JSON list in CSV; real objects in JSONL).

#### 2) Per-row “summary” fields (easy to use)
- **`PersonContacts`**: JSON list of person-linked contacts extracted from all numbers for that row (may be empty).
- **`BestPersonContactName`**
- **`BestPersonContactRole`**
- **`BestPersonContactDepartment`**
- **`BestPersonContactNumber`**

How `BestPersonContact*` is chosen:
- If the second-stage reranker produced `Top_Number_1..3`, we prefer a person contact that is actually one of those Top numbers.
- Otherwise we fall back to a local “best person” picker from `PersonContacts`.

---

### How this interacts with `Top_Number_1..3`

`Top_Number_1..3` is the operational “call list” produced by the ranking logic.

- Sometimes `Top_Number_1` is a **direct dial to a person** (great).
- Sometimes the best option is still a **main office / switchboard** (gatekeeper) depending on what the site exposes.

Practical rule of thumb:
- If `BestPersonContactNumber == Top_Number_1`, your best-first call is a specific person.
- If `BestPersonContactNumber` is present but not in the Top list, treat it as **context/backup**, not automatically the best dial.

---

### Caveats / limitations

- Person linkage is only extracted when the snippet is clear; many sites list numbers without names.
- Don’t assume `PersonContacts` is exhaustive; it’s a “high precision” signal, not a guaranteed roster.

