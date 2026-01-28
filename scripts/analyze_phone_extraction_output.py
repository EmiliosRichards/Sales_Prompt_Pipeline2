"""
Quick QA / deep-dive helper for phone-only (or full-pipeline) phone extraction outputs.

This script prefers JSONL outputs because they preserve nested structures:
- output_data/<run_id>/phone_extraction_results_<run_id>.jsonl

Example:
  python scripts/analyze_phone_extraction_output.py ^
    --jsonl output_data\\20260121_085535_results_stitched_phone_smoke\\phone_extraction_results_20260121_085535_results_stitched_phone_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


def _safe_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    # Some pipelines may stringify lists into CSV/JSONL. We don't try to eval those here.
    return []


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze phone extraction JSONL outputs.")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to phone_extraction_results_<run_id>.jsonl")
    args = ap.parse_args()

    rows = list(_iter_jsonl(args.jsonl))
    n = len(rows)
    if n == 0:
        print("No JSON objects found.")
        return 1

    outcome_ctr = Counter()
    fault_ctr = Counter()
    top1_type_ctr = Counter()
    primary_type_ctr = Counter()
    llm_type_ctr = Counter()
    llm_class_ctr = Counter()

    rows_with_any_top = 0
    rows_with_primary = 0
    rows_with_secondary = 0
    rows_with_debug_regex = 0
    rows_with_llm = 0
    rows_with_person_assoc_in_llm = 0
    rows_with_best_person = 0

    fax_as_primary: List[str] = []

    for r in rows:
        outcome_ctr[_safe_str(r.get("Final_Row_Outcome_Reason")) or ""] += 1
        fault_ctr[_safe_str(r.get("Determined_Fault_Category")) or ""] += 1

        top1 = _safe_str(r.get("Top_Number_1"))
        if top1:
            rows_with_any_top += 1
        top1_type = _safe_str(r.get("Top_Type_1"))
        if top1_type:
            top1_type_ctr[top1_type] += 1

        primary = _safe_str(r.get("Primary_Number_1"))
        if primary:
            rows_with_primary += 1
        ptype = _safe_str(r.get("Primary_Type_1"))
        if ptype:
            primary_type_ctr[ptype] += 1
            if "fax" in ptype.lower():
                fax_as_primary.append(_safe_str(r.get("CompanyName") or r.get("firma") or r.get("run_id")))

        if _safe_str(r.get("Secondary_Number_1")):
            rows_with_secondary += 1

        if _safe_list(r.get("RegexCandidateSnippets")):
            rows_with_debug_regex += 1

        llm_items = _safe_list(r.get("LLMExtractedNumbers"))
        if llm_items:
            rows_with_llm += 1
        for it in llm_items:
            if not isinstance(it, dict):
                continue
            llm_type_ctr[_safe_str(it.get("type")) or ""] += 1
            llm_class_ctr[_safe_str(it.get("classification")) or ""] += 1
            if _safe_str(it.get("associated_person_name")):
                rows_with_person_assoc_in_llm += 1
                break

        if _safe_str(r.get("BestPersonContactName")):
            rows_with_best_person += 1

    print(f"Rows: {n}")
    print(f"Rows with Top_Number_1: {rows_with_any_top} ({rows_with_any_top/n:.0%})")
    print(f"Rows with Primary_Number_1: {rows_with_primary} ({rows_with_primary/n:.0%})")
    print(f"Rows with Secondary_Number_1: {rows_with_secondary} ({rows_with_secondary/n:.0%})")
    print(f"Rows with RegexCandidateSnippets (debug): {rows_with_debug_regex} ({rows_with_debug_regex/n:.0%})")
    print(f"Rows with LLMExtractedNumbers: {rows_with_llm} ({rows_with_llm/n:.0%})")
    print(f"Rows with any person-associated LLM item: {rows_with_person_assoc_in_llm} ({rows_with_person_assoc_in_llm/n:.0%})")
    print(f"Rows with BestPersonContactName set: {rows_with_best_person} ({rows_with_best_person/n:.0%})")
    print("")

    print("Outcome counts (Final_Row_Outcome_Reason):")
    for k, v in outcome_ctr.most_common():
        print(f"  - {k or '<empty>'}: {v}")
    print("")

    print("Fault counts (Determined_Fault_Category):")
    for k, v in fault_ctr.most_common():
        print(f"  - {k or '<empty>'}: {v}")
    print("")

    print("Top_Type_1 counts:")
    for k, v in top1_type_ctr.most_common(15):
        print(f"  - {k or '<empty>'}: {v}")
    print("")

    print("Primary_Type_1 counts:")
    for k, v in primary_type_ctr.most_common(15):
        print(f"  - {k or '<empty>'}: {v}")
    print("")

    print("LLM extracted classification counts (across all items):")
    for k, v in llm_class_ctr.most_common(15):
        print(f"  - {k or '<empty>'}: {v}")
    print("")

    print("LLM extracted type counts (across all items):")
    for k, v in llm_type_ctr.most_common(15):
        print(f"  - {k or '<empty>'}: {v}")
    print("")

    if fax_as_primary:
        sample = fax_as_primary[:10]
        print("Potential anomaly: Primary_Type_1 contains 'Fax' in these rows (sample):")
        for x in sample:
            print(f"  - {x}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

