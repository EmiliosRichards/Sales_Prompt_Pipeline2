"""
Deep-dive helper for the two-pass phone system (regex -> LLM classify -> LLM rerank).

This script reads the phone-only JSONL output and prints:
- Coverage stats for Top_1..3, MainOffice_*, and LLMPhoneRanking
- A few concrete example rows to audit reranking quality (incl. potential "other org" candidates)

Usage (PowerShell):
  python scripts\\deepdive_phone_rerank_output.py --jsonl output_data\\<run_dir>\\phone_extraction_results_<run_id>.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _s(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _nb(v: Any) -> bool:
    """Not blank-ish."""
    s = _s(v)
    return bool(s) and s.lower() not in {"nan", "none", "null"}


def _list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def _dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _row_title(r: Dict[str, Any]) -> str:
    return _s(r.get("CompanyName") or r.get("firma") or r.get("run_id") or r.get("GivenURL"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Deep-dive analysis for phone reranking outputs (JSONL).")
    ap.add_argument("--jsonl", required=True, help="Path to phone_extraction_results_<run_id>.jsonl")
    ap.add_argument("--examples", type=int, default=3, help="How many examples to print per section.")
    args = ap.parse_args()

    rows = list(_iter_jsonl(args.jsonl))
    n = len(rows)
    if n == 0:
        print("No JSON objects found.")
        return 1

    cnt = Counter()
    top1_types = Counter()
    top2_types = Counter()
    top3_types = Counter()

    rows_top1_ne_main = []
    rows_with_best_person = []
    rows_with_low_value_ranked = []
    rows_with_team_kader_candidates = []

    for r in rows:
        if _nb(r.get("Top_Number_1")):
            cnt["top1"] += 1
            if _nb(r.get("Top_Type_1")):
                top1_types[_s(r.get("Top_Type_1"))] += 1
        if _nb(r.get("Top_Number_2")):
            cnt["top2"] += 1
            if _nb(r.get("Top_Type_2")):
                top2_types[_s(r.get("Top_Type_2"))] += 1
        if _nb(r.get("Top_Number_3")):
            cnt["top3"] += 1
            if _nb(r.get("Top_Type_3")):
                top3_types[_s(r.get("Top_Type_3"))] += 1

        if _nb(r.get("MainOffice_Number")):
            cnt["main_office"] += 1

        if _dict(r.get("LLMPhoneRanking")):
            cnt["llm_ranking"] += 1

        t1 = _s(r.get("Top_Number_1"))
        mo = _s(r.get("MainOffice_Number"))
        if t1 and mo and t1 != mo:
            rows_top1_ne_main.append(r)

        if _nb(r.get("BestPersonContactName")):
            rows_with_best_person.append(r)

        ranking = _dict(r.get("LLMPhoneRanking"))
        ranked_numbers = _list(ranking.get("ranked_numbers"))
        for item in ranked_numbers:
            if isinstance(item, dict) and _s(item.get("priority_label")).lower() == "low value":
                rows_with_low_value_ranked.append(r)
                break

        # "Other org" smell: candidate evidence from team roster pages.
        # We'll detect this from the ranking prompt artifacts (best), but for JSONL we use source URLs.
        # If any RegexCandidateSnippets has a source_url containing /teams/kader/ treat as a flag.
        for sn in _list(r.get("RegexCandidateSnippets")):
            if not isinstance(sn, dict):
                continue
            u = _s(sn.get("source_url")).lower()
            if "/teams/kader/" in u or "/kader/" in u:
                rows_with_team_kader_candidates.append(r)
                break

    print(f"Rows: {n}")
    print(f"Top_Number_1 filled: {cnt['top1']} ({cnt['top1']/n:.0%})")
    print(f"Top_Number_2 filled: {cnt['top2']} ({cnt['top2']/n:.0%})")
    print(f"Top_Number_3 filled: {cnt['top3']} ({cnt['top3']/n:.0%})")
    print(f"MainOffice_Number filled: {cnt['main_office']} ({cnt['main_office']/n:.0%})")
    print(f"LLMPhoneRanking present: {cnt['llm_ranking']} ({cnt['llm_ranking']/n:.0%})")
    print(f"Top_1 != MainOffice: {len(rows_top1_ne_main)} ({len(rows_top1_ne_main)/n:.0%})")
    print(f"Rows with BestPersonContactName: {len(rows_with_best_person)} ({len(rows_with_best_person)/n:.0%})")
    print(f"Rows where ranking included priority_label=Low Value: {len(rows_with_low_value_ranked)} ({len(rows_with_low_value_ranked)/n:.0%})")
    print(f"Rows with /teams/kader/ candidates (smell): {len(rows_with_team_kader_candidates)} ({len(rows_with_team_kader_candidates)/n:.0%})")

    def _print_examples(title: str, candidates: List[Dict[str, Any]]) -> None:
        print("")
        print(title)
        for r in candidates[: max(0, int(args.examples))]:
            name = _row_title(r)
            print(f"- {name}")
            print(f"  Top1: {_s(r.get('Top_Number_1'))} ({_s(r.get('Top_Type_1'))})")
            print(f"  Top2: {_s(r.get('Top_Number_2'))} ({_s(r.get('Top_Type_2'))})")
            print(f"  Top3: {_s(r.get('Top_Number_3'))} ({_s(r.get('Top_Type_3'))})")
            print(f"  MainOffice: {_s(r.get('MainOffice_Number'))} ({_s(r.get('MainOffice_Type'))})")
            if _nb(r.get("BestPersonContactName")):
                print(f"  BestPerson: {_s(r.get('BestPersonContactName'))} | {_s(r.get('BestPersonContactRole'))} | {_s(r.get('BestPersonContactNumber'))}")
            rk = _dict(r.get("LLMPhoneRanking"))
            if rk:
                rs = _s(rk.get("reasoning_summary"))
                if rs:
                    print(f"  Rerank summary: {rs}")

    _print_examples("Examples where Top_Number_1 != MainOffice_Number:", rows_top1_ne_main)
    _print_examples("Examples with BestPersonContactName:", rows_with_best_person)
    _print_examples("Examples where reranker included a 'Low Value' ranked item:", rows_with_low_value_ranked)
    _print_examples("Examples with /teams/kader/ candidate sources (other-org smell):", rows_with_team_kader_candidates)

    print("")
    print("Top_Type_1 distribution (top 10):")
    for k, v in top1_types.most_common(10):
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

