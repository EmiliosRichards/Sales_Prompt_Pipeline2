### Handover bucket: Single-company Sales Pitch UI integration

This folder is a **handover bucket** meant to be copied into another repo/project and given to an agent that will build a **web UI** for running the **Sales Pitch system** on **one company at a time**.

The UI use-case:
- User pastes a **company URL**.
- User provides a **description** (and optionally keywords/reasoning) in the UI.
- Backend runs the **LLM pipeline** to:
  - extract structured company attributes
  - select the best Golden Partner match + reasoning
  - generate a German sales pitch tailored to that match
- Optional: run **phone extraction** (Playwright + LLM) and return Top 1–3 callable numbers + a main-line backup.

This bucket **does not delete or modify** anything in the main repo. It only adds documentation + integration code.

---

### What’s in this bucket

- `docs/`
  - `API_CONTRACT.md`: request/response schema for the UI backend
  - `ENV_VARS_TO_TRANSFER.md`: env vars that must be carried into the UI project
  - `INTEGRATION_OPTIONS.md`: two recommended integration styles (direct-python vs subprocess)
  - `REFERENCE_*`: curated references to how the pipeline works and how outputs/phone fields should be interpreted
- `code/`
  - `single_company_service.py`: a small **direct-Python** service wrapper you can call from a UI backend
  - `fastapi_example_app.py`: a minimal FastAPI server exposing `/analyze` for the UI
  - `run_single_company_local.py`: CLI smoke-test runner (no UI) for validating env + prompts
- `prompts/`
  - Copies of the prompts used by the pipeline steps in this repo (so the UI agent can see/adjust them safely)
- `PROMPT_FOR_UI_AGENT.md`
  - A ready-to-paste prompt to give another agent that will build the UI + backend glue.
- `env.ui.example`
  - A minimal env template for the UI project.

---

### Quick start (backend-only, local)

1) **Set env vars**
- Copy `env.ui.example` → `.env` in the UI project.
- Set `GEMINI_API_KEY`.
- Ensure `PATH_TO_GOLDEN_PARTNERS_DATA` points to your golden partner Excel/CSV.

2) **Install deps**
- Reuse the main repo’s `requirements.txt` (this system uses `google-generativeai`, `pydantic`, `pandas`, `openpyxl`, etc.).
- If you enable phone extraction, you must also run `playwright install` once.
- If you run the FastAPI demo app, install `handover_bucket_ui/requirements_ui.txt` too.

3) **Run the example API**

```bash
python handover_bucket_ui/code/fastapi_example_app.py
```

4) **Call it**
- Use the schema in `docs/API_CONTRACT.md`.

---

### Notes about the “second bucket” (other project)

You mentioned you will include a second bucket from another project that “evaluates the company”.
Design assumption for the UI:
- The UI backend should call **both systems** and then merge results in the UI response.
- This bucket focuses on the **Sales Pitch** system. The UI agent should treat the “second bucket” as an additional service/module and keep the integration boundary clean (separate request/response payload, then merge).

