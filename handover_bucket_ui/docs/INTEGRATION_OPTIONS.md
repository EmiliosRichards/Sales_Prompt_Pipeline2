### Integration options for the UI webpage

You have two good ways to “plug” the existing system into a UI.

---

### Option A (recommended): Direct-Python integration

Use the service wrapper in `code/single_company_service.py`.

Pros:
- Lowest latency (no subprocess, no temp files).
- Easy to return a structured JSON response directly to UI.
- Easier to add caching at the request level.

Cons:
- You must ensure the UI backend runs in an environment where the pipeline’s dependencies exist.
- If you enable phone extraction, Playwright concurrency must be controlled.

---

### Option B: Subprocess orchestration (most “faithful” to existing pipelines)

Backend writes a 1-row CSV → runs `main_pipeline.py` / `phone_extract.py` → reads output JSONL/CSV → returns to UI.

Pros:
- Uses the exact same entrypoints and output logic as batch runs.
- Very easy to keep parity with “production batch” behavior.

Cons:
- Higher latency (process startup, disk IO).
- Requires robust temp-file cleanup and locking management.

---

### Recommended UI flow (given your description inputs)

Since the UI provides description text, prefer:
- **description-driven mode** (skip scraping)
- generate short German summary (≤100 words) for UI display
- attribute extraction → partner match → sales pitch generation
- optional phone extraction toggle

