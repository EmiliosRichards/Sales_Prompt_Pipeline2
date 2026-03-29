### API Contract: single-company sales pitch (+ optional phones)

This is a suggested request/response contract for the UI backend.

The UI can call a single endpoint with:
- a URL
- description inputs (provided by the UI)
- optional toggles (e.g. phone extraction)

---

### Request (JSON)

```json
{
  "company_url": "https://example.com",
  "company_name": "Example GmbH",
  "description": "Free text description from the UI (often English).",
  "keywords": "Optional keywords string (comma-separated or free text).",
  "reasoning": "Optional reasoning / notes string.",
  "run_phone_extraction": false
}
```

Notes:
- `company_name` is optional; the system can operate without it.
- `description` is required for the “description-driven” mode (recommended for UI).
- `keywords` + `reasoning` are optional, but improve extraction.

---

### Response (JSON)

```json
{
  "inputs": {
    "company_url": "https://example.com",
    "company_name": "Example GmbH"
  },
  "short_german_description": "German ≤100-word summary/translation.",
  "attributes": {
    "b2b_indicator": true,
    "phone_outreach_suitability": true,
    "target_group_size_assessment": "Appears Medium",
    "industry": "…",
    "products_services_offered": ["…"],
    "usp_key_selling_points": ["…"],
    "customer_target_segments": ["…"],
    "business_model": "…",
    "company_size_category_inferred": "SME",
    "innovation_level_indicators_text": "…",
    "website_clarity_notes": "…"
  },
  "partner_match": {
    "match_score": "High",
    "matched_partner_name": "…",
    "match_rationale_features": ["…", "…"]
  },
  "sales_pitch": {
    "phone_sales_line": "German phone pitch (contains the programmatic placeholder).",
    "match_rationale_features": ["…", "…"],
    "matched_partner_name": "…",
    "matched_partner_description": "…",
    "avg_leads_per_day": 10,
    "rank": 5
  },
  "phones": {
    "enabled": false,
    "top_numbers": [],
    "main_office_backup": null,
    "diagnostics": null
  },
  "audit": {
    "errors": [],
    "artifacts_dir": null
  }
}
```

Phone output suggestion (when enabled):
- `top_numbers`: up to 3 items with `{number, type, source_url, person_name?, person_role?}`
- `main_office_backup`: `{number, type, source_url}`
- never include fax as callable.

