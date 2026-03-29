### Reference: system overview (from this repo)

This UI is integrating the existing “full pipeline” logic that already exists in this repo.

#### Systems
- **Phone extraction** (`phone_extract.py`)
  - Scrape site → regex candidates → LLM classify → consolidate → LLM rerank into Top 1–3 + main line backup.
- **Sales pitch pipeline** (`main_pipeline.py`)
  - Summarize (or reuse description blob) → extract attributes → partner match → sales pitch.

#### UI integration recommendation
For a single-company UI, prefer **description-driven mode**:
- The UI provides `description` (and optionally `keywords`, `reasoning`).
- Backend generates a short German summary (≤100 words) for human display.
- Backend uses the combined blob for attribute extraction + partner matching + sales pitch generation.
- Phone extraction can be an optional toggle (higher latency/cost).

