"""
Inspect output_data/_run_history.jsonl (append-only run index).

Examples (PowerShell):
  python scripts/show_run_history.py --last 20
  python scripts/show_run_history.py --pipeline phone_extract --last 10
  python scripts/show_run_history.py --status failed --last 50
  python scripts/show_run_history.py --contains apol4 --last 200
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = (ln or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _fmt_ts(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        # Keep it readable and stable
        return s.replace("T", " ").replace("+00:00", "Z")
    except Exception:
        return s


def _iter_filtered(
    rows: Iterable[Dict[str, Any]],
    pipeline: str,
    status: str,
    contains: str,
) -> Iterable[Dict[str, Any]]:
    for r in rows:
        if pipeline and str(r.get("pipeline_name") or "") != pipeline:
            continue
        if status and str(r.get("status") or "") != status:
            continue
        if contains:
            blob = json.dumps(r, ensure_ascii=False).lower()
            if contains.lower() not in blob:
                continue
        yield r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="", help="Path to _run_history.jsonl (default: output_data/_run_history.jsonl)")
    ap.add_argument("--last", type=int, default=20, help="Show last N (after filtering).")
    ap.add_argument("--pipeline", default="", help="Filter by pipeline_name (e.g. phone_extract, main_pipeline).")
    ap.add_argument("--status", default="", help="Filter by status (completed/failed).")
    ap.add_argument("--contains", default="", help="Case-insensitive substring match against JSON.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    hist = Path(args.history) if args.history else (repo_root / "output_data" / "_run_history.jsonl")
    rows = _load_jsonl(hist)
    if not rows:
        print(f"No history found at: {hist}")
        return 0

    filtered = list(_iter_filtered(rows, pipeline=args.pipeline, status=args.status, contains=args.contains))
    if not filtered:
        print("No matching runs.")
        return 0

    tail = filtered[-max(1, int(args.last)) :]
    for r in tail:
        rid = r.get("run_id")
        pname = r.get("pipeline_name")
        st = r.get("status")
        started = _fmt_ts(r.get("started_at_utc"))
        finished = _fmt_ts(r.get("finished_at_utc"))
        inp = r.get("input_file") or ""
        git = r.get("git") or {}
        commit = (git.get("commit") or "")[:12] if isinstance(git, dict) else ""
        dirty = git.get("dirty") if isinstance(git, dict) else None
        dur = r.get("duration_seconds")
        print(f"{rid}\t{pname}\t{st}\t{dur}\t{started}\t{finished}\t{commit}\tdirty={dirty}\t{os.path.basename(str(inp))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

