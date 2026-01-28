import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _s(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _nb(x: Any) -> bool:
    v = _s(x)
    return bool(v) and v.lower() not in {"nan", "none", "null"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _uniq_llm_items(llm_items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(llm_items, list):
        return out
    for it in llm_items:
        if not isinstance(it, dict):
            continue
        num = _s(it.get("number"))
        if not _nb(num) or num in seen:
            continue
        seen.add(num)
        out.append(it)
    return out


def _summarize_row(r: Dict[str, Any]) -> Dict[str, Any]:
    rk = r.get("LLMPhoneRanking") if isinstance(r.get("LLMPhoneRanking"), dict) else {}
    ranked = rk.get("ranked_numbers") or []
    ranked = [it for it in ranked if isinstance(it, dict)]

    included = {
        _s(r.get("Top_Number_1")),
        _s(r.get("Top_Number_2")),
        _s(r.get("Top_Number_3")),
        _s(r.get("MainOffice_Number")),
    }
    included = {x for x in included if _nb(x)}

    llm_items = _uniq_llm_items(r.get("LLMExtractedNumbers") or [])
    excluded: List[Dict[str, Any]] = []
    for it in llm_items:
        num = _s(it.get("number"))
        if not _nb(num) or num in included:
            continue
        excluded.append(it)

    snippets: List[Dict[str, str]] = []
    for sn in (r.get("RegexCandidateSnippets") or []):
        if not isinstance(sn, dict):
            continue
        cand = _s(sn.get("candidate_number"))
        if not _nb(cand):
            continue
        if cand in included:
            continue
        if any(_s(x.get("candidate_number")) == cand for x in snippets):
            continue
        snippets.append(
            {
                "candidate_number": cand,
                "source_url": _s(sn.get("source_url")),
                "snippet": _s(sn.get("snippet")),
            }
        )
        if len(snippets) >= 2:
            break

    return {
        "CompanyName": _s(r.get("CompanyName")),
        "GivenURL": _s(r.get("GivenURL")),
        "Outcome": _s(r.get("Final_Row_Outcome_Reason")),
        "ScrapingStatus": _s(r.get("ScrapingStatus")),
        "Top": [
            {
                "number": _s(r.get("Top_Number_1")),
                "type": _s(r.get("Top_Type_1")),
                "source_url": _s(r.get("Top_SourceURL_1")),
            },
            {
                "number": _s(r.get("Top_Number_2")),
                "type": _s(r.get("Top_Type_2")),
                "source_url": _s(r.get("Top_SourceURL_2")),
            },
            {
                "number": _s(r.get("Top_Number_3")),
                "type": _s(r.get("Top_Type_3")),
                "source_url": _s(r.get("Top_SourceURL_3")),
            },
        ],
        "MainOffice": {
            "number": _s(r.get("MainOffice_Number")),
            "type": _s(r.get("MainOffice_Type")),
            "source_url": _s(r.get("MainOffice_SourceURL")),
        },
        "BestPerson": {
            "name": _s(r.get("BestPersonContactName")),
            "role": _s(r.get("BestPersonContactRole")),
            "department": _s(r.get("BestPersonContactDepartment")),
            "number": _s(r.get("BestPersonContactNumber")),
        },
        "RerankSummary": _s(rk.get("reasoning_summary")),
        "RerankTop5": [
            {
                "number": _s(it.get("number")),
                "type": _s(it.get("type")),
                "priority_label": _s(it.get("priority_label")),
                "person_role": _s(it.get("associated_person_role")),
                "person_name": _s(it.get("associated_person_name")),
            }
            for it in ranked[:5]
        ],
        "ExcludedCandidates_Unique": [
            {
                "number": _s(it.get("number")),
                "type": _s(it.get("type")),
                "classification": _s(it.get("classification")),
                "is_valid": it.get("is_valid"),
                "person_role": _s(it.get("associated_person_role")),
                "person_name": _s(it.get("associated_person_name")),
                "person_department": _s(it.get("associated_person_department")),
                "source_url": _s(it.get("source_url")),
            }
            for it in excluded
        ],
        "OtherRelevantNumbers": r.get("OtherRelevantNumbers"),
        "DebugSnippets_2": snippets,
    }


def _fmt(x: Any) -> str:
    v = _s(x)
    return v if _nb(v) else "(blank)"


def _row_block(summary: Dict[str, Any], idx: int) -> str:
    top_lines: List[str] = []
    for i, t in enumerate(summary["Top"], start=1):
        if not _nb(t.get("number")) and not _nb(t.get("type")):
            continue
        top_lines.append(
            f"- Top_{i}: **{_fmt(t.get('number'))}** — {_fmt(t.get('type'))} — src: {_fmt(t.get('source_url'))}"
        )
    if not top_lines:
        top_lines = ["- (none)"]

    bp = summary["BestPerson"]
    bp_line = f"{_fmt(bp.get('name'))} | {_fmt(bp.get('role'))} | {_fmt(bp.get('department'))} | {_fmt(bp.get('number'))}"

    rk_lines: List[str] = []
    for it in summary["RerankTop5"]:
        if not _nb(it.get("number")):
            continue
        extra = ""
        if _nb(it.get("person_name")) or _nb(it.get("person_role")):
            extra = f" — person={_fmt(it.get('person_name'))} ({_fmt(it.get('person_role'))})"
        rk_lines.append(
            f"- {_fmt(it.get('number'))} — {_fmt(it.get('type'))} — **{_fmt(it.get('priority_label'))}**{extra}"
        )
    if not rk_lines:
        rk_lines = ["- (no rerank list)"]

    exc_lines: List[str] = []
    for it in (summary["ExcludedCandidates_Unique"] or [])[:8]:
        line = (
            f"- {_fmt(it.get('number'))} — type={_fmt(it.get('type'))}, "
            f"class={_fmt(it.get('classification'))}, valid={it.get('is_valid')}"
        )
        if _nb(it.get("person_name")) or _nb(it.get("person_role")):
            line += f", person={_fmt(it.get('person_name'))} ({_fmt(it.get('person_role'))})"
        if _nb(it.get("source_url")):
            line += f", url={_fmt(it.get('source_url'))}"
        exc_lines.append(line)
    if not exc_lines:
        exc_lines = ["- (none)"]

    sn_lines: List[str] = []
    for sn in summary["DebugSnippets_2"] or []:
        if not _nb(sn.get("candidate_number")):
            continue
        clip = _s(sn.get("snippet"))
        if len(clip) > 240:
            clip = clip[:240] + "…"
        sn_lines.append(f"- {sn['candidate_number']} — {_fmt(sn.get('source_url'))} — \"{clip}\"")
    if not sn_lines:
        sn_lines = ["- (none)"]

    mo = summary["MainOffice"]

    md: List[str] = []
    md.append(f"## Example {idx}: {summary['CompanyName']}")
    md.append(f"- **GivenURL**: `{summary['GivenURL']}`")
    md.append(f"- **Outcome**: **{summary['Outcome']}** | ScrapingStatus={summary['ScrapingStatus']}")
    md.append(
        f"- **MainOffice backup**: **{_fmt(mo.get('number'))}** — {_fmt(mo.get('type'))} — src: {_fmt(mo.get('source_url'))}"
    )
    md.append(f"- **BestPersonContact**: {bp_line}")
    md.append("- **Top numbers selected**:")
    md.extend(["  " + l for l in top_lines])
    md.append(f"- **Rerank summary**: {_fmt(summary['RerankSummary'])}")
    md.append("- **Reranker top (up to 5)**:")
    md.extend(["  " + l for l in rk_lines])
    md.append("- **Excluded candidates (unique, not in Top/MainOffice)**:")
    md.extend(["  " + l for l in exc_lines])
    md.append("- **Debug snippet samples (2)**:")
    md.extend(["  " + l for l in sn_lines])
    md.append("")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Output run folder (e.g. output_data/...)")
    ap.add_argument("--jsonl", required=True, help="Path to the phone extraction results JSONL")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--good", type=int, default=10)
    ap.add_argument("--fail", type=int, default=5)
    ap.add_argument("--out", default="examples_postfix_10good_5notop.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    jsonl_path = Path(args.jsonl)
    rows = _read_jsonl(jsonl_path)

    success = [r for r in rows if _nb(r.get("Top_Number_1"))]
    no_top = [r for r in rows if not _nb(r.get("Top_Number_1"))]

    random.seed(args.seed)
    random.shuffle(success)
    random.shuffle(no_top)

    good_examples = success[: args.good]
    fail_examples = no_top[: args.fail]

    out_md: List[str] = []
    out_md.append("# Phone extraction examples (post-fix run)\n")
    out_md.append(f"Run folder: `{run_dir.as_posix()}`\n")
    out_md.append("## Raw output files")
    out_md.append(f"- Results CSV: `{(run_dir / (jsonl_path.stem + '.csv')).as_posix()}`")
    out_md.append(f"- Results JSONL: `{jsonl_path.as_posix()}`")
    out_md.append(f"- Results (merged) CSV: `{(run_dir / (jsonl_path.stem + '_merged.csv')).as_posix()}`")
    out_md.append(f"- Results (merged) JSONL: `{(run_dir / (jsonl_path.stem + '_merged.jsonl')).as_posix()}`")
    out_md.append(f"- Augmented input CSV: `{(run_dir / ('input_augmented_' + jsonl_path.stem.split('phone_extraction_results_',1)[-1] + '.csv')).as_posix()}`")
    out_md.append(f"- Run log: `{next(run_dir.glob('phone_extract_*.log')).as_posix() if list(run_dir.glob('phone_extract_*.log')) else '(not found)'}`")
    out_md.append("")

    out_md.append("## 10 examples with Top numbers")
    for i, r in enumerate(good_examples, start=1):
        out_md.append(_row_block(_summarize_row(r), i))

    out_md.append("## 5 examples with NO Top_Number_1")
    for i, r in enumerate(fail_examples, start=1):
        out_md.append(_row_block(_summarize_row(r), i))

    out_path = run_dir / args.out
    out_path.write_text("\n".join(out_md), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

