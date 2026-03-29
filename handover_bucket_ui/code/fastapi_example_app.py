"""
Minimal FastAPI app for the single-company sales pitch UI.

Run:
  python handover_bucket_ui/code/fastapi_example_app.py
Then POST to:
  http://127.0.0.1:8000/analyze

Notes:
- If `run_phone_extraction=true`, this may invoke Playwright and take longer.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Ensure repo root is on sys.path so `handover_bucket_ui` + `src` imports work when running this file directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from handover_bucket_ui.code.single_company_service import run_single_company_sales_pitch


class AnalyzeRequest(BaseModel):
    company_url: str = Field(..., description="Company website URL")
    company_name: Optional[str] = Field(default="", description="Optional company name (UI display)")
    description: str = Field(..., description="Description text from UI")
    keywords: Optional[str] = Field(default=None, description="Optional keywords")
    reasoning: Optional[str] = Field(default=None, description="Optional reasoning")
    run_phone_extraction: bool = Field(default=False, description="Optional phone extraction (not wired in this minimal demo)")


app = FastAPI(title="Single-company Sales Pitch API")


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    # For a UI backend, you likely want per-request artifacts stored under a run folder.
    artifacts_dir = os.path.normpath("output_data/ui_single_company_requests")
    res = run_single_company_sales_pitch(
        company_url=req.company_url,
        company_name=req.company_name or "",
        description=req.description,
        keywords=req.keywords,
        reasoning=req.reasoning,
        artifacts_dir=artifacts_dir,
        run_phone_extraction=bool(req.run_phone_extraction),
    )
    return {
        "inputs": {
            "company_url": req.company_url,
            "company_name": req.company_name,
        },
        "short_german_description": res.short_german_description,
        "attributes": res.attributes,
        "partner_match": res.partner_match,
        "sales_pitch": res.sales_pitch,
        "phones": {
            "enabled": bool(req.run_phone_extraction),
            "result": res.phones,
        },
        "audit": {
            "errors": res.errors,
            "artifacts_dir": artifacts_dir,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

