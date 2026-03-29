"""
CLI smoke test: run the sales pitch system for a single company.

Example:
  python handover_bucket_ui/code/run_single_company_local.py ^
    --url "https://example.com" ^
    --name "Example GmbH" ^
    --description "We sell ..." ^
    --keywords "b2b, saas" ^
    --reasoning "UI-provided notes"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repo root is on sys.path so `handover_bucket_ui` + `src` imports work when running this file directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from handover_bucket_ui.code.single_company_service import run_single_company_sales_pitch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--description", required=True)
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--artifacts-dir", default="output_data/ui_single_company_cli")
    args = ap.parse_args()

    res = run_single_company_sales_pitch(
        company_url=args.url,
        company_name=args.name,
        description=args.description,
        keywords=args.keywords,
        reasoning=args.reasoning,
        artifacts_dir=args.artifacts_dir,
        run_phone_extraction=False,
    )

    print(json.dumps({
        "short_german_description": res.short_german_description,
        "attributes": res.attributes,
        "partner_match": res.partner_match,
        "sales_pitch": res.sales_pitch,
        "phones": res.phones,
        "errors": res.errors,
        "artifacts_dir": args.artifacts_dir,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

