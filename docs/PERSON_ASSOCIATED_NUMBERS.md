### Person-associated numbers (direct dials): implemented

You’re right: “James Brandon (CEO) +49…” is often more valuable than a generic switchboard number.

Today, the pipeline keeps:
- the **number**,
- a **snippet**,
- a **type/classification** inferred by the phone LLM.

But it does not reliably extract and persist:
- the **person name**,
- the **role/department**,
- whether the number is a **direct dial**.

---

### Data model

`PhoneNumberLLMOutput` includes optional fields:
- `associated_person_name: Optional[str]`
- `associated_person_role: Optional[str]`
- `associated_person_department: Optional[str]`
- `is_direct_dial: Optional[bool]`

We propagate these fields into:
- `LLMExtractedNumbers` (always, for auditability)
- `ConsolidatedPhoneNumberSource` (per-source), so we can surface “best direct dial to a decision maker” across multiple pages.

---

### Extraction approach (robust)

#### Stage A: regex candidates + snippets (already exists)
We already capture snippets around each candidate number (`extract_numbers_with_snippets_from_text`).

#### Stage B: LLM classification (extended prompt)
`prompts/phone_extraction_prompt.txt` instructs the LLM to:
- classify (valid/type/classification) **and**
- extract person name/role/department **only when the snippet clearly links the number to a specific person** (or explicitly indicates direct-dial / extension).

This avoids brittle regex heuristics for names/roles and works for German formatting.

---

### How this impacts “Top_Number_1..3” ranking

We add a ranking layer that can prefer direct dials when they target high-value roles (but does **not** treat generic department lines as “person contacts” unless a person or explicit direct-dial is present):

Example scoring ideas:
- `CEO/Geschäftsführer/Founder` direct dial: +100
- `Head of Sales/Vertrieb/Business Development`: +80
- `Support/Service`: +20
- `Fax`: -50

This ranking would:
- keep the switchboard as a fallback,
- but bubble up “right person” numbers when present.

---

### Output columns (CSV + JSONL)

We store the full structured data in JSONL and mirror it in CSV as JSON strings (for list/dict fields), plus a small summary:

- `BestPersonContactName`
- `BestPersonContactRole`
- `BestPersonContactDepartment`
- `BestPersonContactNumber`
- `PersonContacts` (JSON list; JSON-string in CSV)

`PersonContacts` item shape:
- `name`, `role`, `department`
- `number`
- `classification`, `type`, `source_url`
- `is_direct_dial`

---

### Why this is worth it

It enables:
- higher connect rates (calling the correct person),
- better personalization (pitch can reference role/department),
- better filtering (ignore irrelevant numbers).

