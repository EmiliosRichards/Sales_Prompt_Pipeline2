### Files to vendor / include in the UI repo

The code in `handover_bucket_ui/code/` is intentionally thin: it calls the **real pipeline modules** in this repo.

If the UI is built in a *separate* repo, you have two options:

#### Option 1: Vendor this repo as a subfolder / module
- Copy these folders into the UI repo (preserving paths):
  - `src/`
  - `prompts/` (or update env vars to point to the bucket copies)
  - `requirements.txt`
  - `data/<your golden partner file>` (or store elsewhere and set `PATH_TO_GOLDEN_PARTNERS_DATA`)

Then you can call:
- `handover_bucket_ui/code/single_company_service.py` directly from your backend.

#### Option 2: Keep this repo separate and call it as a subprocess
- UI backend writes a 1-row input file and runs:
  - `python main_pipeline.py --pitch-from-description ...`
  - optional: `python phone_extract.py ...`

This option requires the UI backend machine to have this repo available and runnable.

