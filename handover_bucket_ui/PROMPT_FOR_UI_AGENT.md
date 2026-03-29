### Prompt for the UI agent (copy/paste)

You are building a small webpage UI that lets a user run our **single-company Sales Pitch pipeline**.

## Goal
Build a UI where the user:
- pastes a **company URL**
- pastes/provides a **company description** (text)
- optionally provides **keywords** and **reasoning**
- optionally toggles **phone extraction**

Backend should run the existing system’s LLM steps:
- (UI mode) Create a **German ≤100-word short summary/translation** of the provided description → return as `Short German Description` in the UI.
- Use the combined description blob (Description + optional Keywords + optional Reasoning) as the context for:
  - **attribute extraction** (`extract_detailed_attributes`)
  - **golden partner matching** (`match_partner`)
  - **sales pitch generation** (`generate_sales_pitch`)

Phone extraction:
- If toggled on, run the existing phone extraction system and return:
  - Top 1–3 callable numbers (+ type + source url, and person name/role when present)
  - Main office backup if present
  - Never return fax as a callable number

## Repo assets
Use the handover bucket folder `handover_bucket_ui/`:
- `START_HERE.md` explains the system + files.
- `code/single_company_service.py` is a direct-Python wrapper for the LLM calls.
- `code/fastapi_example_app.py` is a minimal API you can extend.
- `docs/API_CONTRACT.md` shows a recommended request/response schema.
- `docs/ENV_VARS_TO_TRANSFER.md` lists env vars needed for the UI project.

## Requirements / constraints
- Do NOT change or delete existing pipeline logic; only add UI/backend glue.
- Make it easy to run locally (single command).
- Include good error visibility: return error messages + keep LLM artifacts on disk for auditing.
- Assume a second “company evaluation” bucket exists from another project: design the UI/backend so it can call that second system and merge results cleanly.

## Deliverables
- Simple webpage (React/Next.js or minimal HTML) + backend API.
- A short README for how to run it.

