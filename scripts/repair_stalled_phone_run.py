"""
Repair a stalled parallel phone_extract run by reusing its partial *_live.jsonl outputs
and merging in completed retry runs (e.g. runs produced with suffixes like `repair_*`).

This script is intentionally conservative:
- It never deletes the base run's *_live.* files unless all repaired artifacts exist and are non-empty.
- It writes standard final artifacts into the base run directory:
  - phone_extraction_results_<base_run_id>.csv/.jsonl
  - phone_extraction_results_<base_run_id>_merged.csv/.jsonl
  - input_augmented_<base_run_id>.csv/.jsonl
  - REPAIR_SUMMARY.md

Usage:
  python scripts/repair_stalled_phone_run.py --base-run-id 20260123_131225_apol4_unique500_3k_phone50w
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _stringify_for_csv(v: object) -> object:
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return v


def _load_jsonl_by_row_1based(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            r = obj.get("__source_row_1based")
            if r is None:
                continue
            try:
                ri = int(r)
            except Exception:
                continue
            out[ri] = obj
    return out


def _read_csv_header(path: Path, delimiter: str) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f, delimiter=delimiter))


def _detect_delimiter_from_header_line(path: Path, default: str) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
        if not sample:
            return default
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|:")
            return dialect.delimiter or default
        except Exception:
            return default
    except Exception:
        return default


def _parse_total_rows_from_master_log(master_log: Path, fallback: int = 0) -> int:
    if not master_log.exists():
        return fallback
    with master_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"total_rows=(\d+)", line)
            if m:
                return int(m.group(1))
    return fallback


def _detect_csv_delimiter(file_path: Path) -> str:
    try:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
        if not sample:
            return ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|:")
            return dialect.delimiter or ","
        except Exception:
            counts = {sep: sample.count(sep) for sep in [";", ",", "\t", "|", ":"]}
            best = max(counts.items(), key=lambda kv: kv[1])
            return best[0] if best[1] > 0 else ","
    except Exception:
        return ","


def _is_fax_label(v: Any) -> bool:
    s = ("" if v is None else str(v)).strip().lower()
    return "fax" in s or "telefax" in s


def _pick_best_callable(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    for i in (1, 2, 3):
        num = row.get(f"Top_Number_{i}")
        typ = row.get(f"Top_Type_{i}")
        src = row.get(f"Top_SourceURL_{i}")
        if _is_fax_label(typ):
            continue
        s = ("" if num is None else str(num)).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            return s, ("" if typ is None else str(typ)), ("" if src is None else str(src))
    # Fallback to input number (keep as text)
    for k in ("NormalizedGivenPhoneNumber", "GivenPhoneNumber"):
        v = row.get(k)
        s = ("" if v is None else str(v)).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            return s, "Input", "input:Company Phone"
    return None, None, None


def _write_csv(
    out_path: Path,
    rows: Iterable[Dict[str, Any]],
    header: List[str],
    delimiter: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", delimiter=delimiter)
        w.writeheader()
        for r in rows:
            w.writerow({k: _stringify_for_csv(r.get(k, "")) for k in header})


def _write_jsonl(out_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run-id", required=True, help="Folder name under output_data/ for the stalled run.")
    ap.add_argument(
        "--input-file",
        required=True,
        help="Original input CSV used for the base run (needed to rebuild input_augmented_<run_id>.csv).",
    )
    ap.add_argument(
        "--output-data-dir",
        default="output_data",
        help="Path to output_data directory (default: output_data).",
    )
    ap.add_argument(
        "--retry-dir-substring",
        default="_repair_apol4_",
        help="Substring that identifies retry run folders (default: _repair_apol4_).",
    )
    ap.add_argument(
        "--retry-dir-suffix",
        default="_w25",
        help="Suffix that identifies retry run folders (default: _w25).",
    )
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    out_data = (repo_root / args.output_data_dir).resolve()
    base_dir = (out_data / args.base_run_id).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Base run dir not found: {base_dir}")

    # Base live artifacts
    base_results_live_jsonl = base_dir / f"phone_extraction_results_{args.base_run_id}_live.jsonl"
    base_aug_live_jsonl = base_dir / f"input_augmented_{args.base_run_id}_live.jsonl"
    base_results_live_csv = base_dir / f"phone_extraction_results_{args.base_run_id}_live.csv"
    base_aug_live_csv = base_dir / f"input_augmented_{args.base_run_id}_live.csv"
    base_status_json = base_dir / f"phone_extract_{args.base_run_id}_live_status.json"
    base_master_log = base_dir / f"phone_extract_{args.base_run_id}.log"

    if not base_results_live_jsonl.exists():
        raise FileNotFoundError(
            f"Missing base live results JSONL: {base_results_live_jsonl}"
        )

    total_rows = _parse_total_rows_from_master_log(base_master_log, fallback=0)
    if total_rows <= 0:
        raise RuntimeError(
            f"Could not determine total_rows from {base_master_log}. "
            "Refusing to proceed (to avoid writing incomplete outputs)."
        )

    # Discover retry runs (same date as base by default, but we keep it flexible)
    retry_dirs: List[Path] = []
    for name in os.listdir(out_data):
        p = out_data / name
        if not p.is_dir():
            continue
        if args.retry_dir_substring not in name:
            continue
        if not name.endswith(args.retry_dir_suffix):
            continue
        retry_dirs.append(p)
    retry_dirs.sort(key=lambda p: p.stat().st_mtime)
    if not retry_dirs:
        raise RuntimeError(f"No retry dirs found in {out_data} matching {args.retry_dir_substring}*{args.retry_dir_suffix}")

    print(f"Base run: {args.base_run_id}")
    print(f"Total rows expected: {total_rows}")
    print(f"Retry dirs found: {len(retry_dirs)}")

    # Load base partial rows
    results_map = _load_jsonl_by_row_1based(base_results_live_jsonl)
    base_live_rows = len(results_map)
    print(f"Base live rows present: {base_live_rows}")

    # Merge retry outputs
    used_retries: List[str] = []
    for rd in retry_dirs:
        merged_results = next(rd.glob("phone_extraction_results_*_merged.jsonl"), None)
        if merged_results is None:
            continue
        radd = _load_jsonl_by_row_1based(merged_results)
        if not radd:
            continue
        results_map.update(radd)
        used_retries.append(rd.name)

    missing = [i for i in range(1, total_rows + 1) if i not in results_map]
    if missing:
        raise RuntimeError(f"Repair incomplete: still missing {len(missing)} rows (first 25): {missing[:25]}")

    ordered_results = [results_map[i] for i in range(1, total_rows + 1)]

    # Determine headers and delimiters from the base live CSVs (preferred for stable column order).
    results_delim = _detect_delimiter_from_header_line(base_results_live_csv, default=",")
    aug_delim = _detect_delimiter_from_header_line(base_aug_live_csv, default=";")
    if base_results_live_csv.exists():
        results_header = _read_csv_header(base_results_live_csv, delimiter=results_delim)
    else:
        results_header = []

    # Ensure headers include any new keys found in JSONLs
    if not results_header:
        results_header = sorted({k for r in ordered_results for k in r.keys()})
    else:
        for k in sorted({k for r in ordered_results for k in r.keys()}):
            if k not in results_header:
                results_header.append(k)

    # Write standard final artifacts into base run directory
    merged_csv = base_dir / f"phone_extraction_results_{args.base_run_id}_merged.csv"
    merged_jsonl = base_dir / f"phone_extraction_results_{args.base_run_id}_merged.jsonl"
    final_csv = base_dir / f"phone_extraction_results_{args.base_run_id}.csv"
    final_jsonl = base_dir / f"phone_extraction_results_{args.base_run_id}.jsonl"
    aug_csv = base_dir / f"input_augmented_{args.base_run_id}.csv"
    aug_jsonl = base_dir / f"input_augmented_{args.base_run_id}.jsonl"

    print("Writing repaired outputs...")
    _write_csv(merged_csv, ordered_results, results_header, delimiter=results_delim)
    _write_jsonl(merged_jsonl, ordered_results)
    # For repaired runs, we set the stable final to be identical to the merged results.
    shutil.copyfile(merged_csv, final_csv)
    shutil.copyfile(merged_jsonl, final_jsonl)
    # Rebuild augmented output from the ORIGINAL input CSV + repaired results (do not rely on live/retry augmented JSONLs;
    # they intentionally omit __source_row_1based).
    input_path = Path(args.input_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    input_sep = _detect_csv_delimiter(input_path)
    # Robust read (input has multiline quoted descriptions).
    import pandas as pd

    original_df = pd.read_csv(
        input_path,
        sep=input_sep,
        engine="python",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    original_df["__source_row_1based"] = list(range(1, len(original_df) + 1))
    results_df = pd.DataFrame(ordered_results)
    # Keep a strict join key as int
    results_df["__source_row_1based"] = results_df["__source_row_1based"].astype(int)
    merged_aug = original_df.merge(results_df, how="left", on="__source_row_1based", suffixes=("", "_results"))

    # Derive legacy summary columns expected by downstream tooling
    phone_found: List[str] = []
    phone_type: List[str] = []
    phone_src: List[str] = []
    for _, r in results_df.sort_values("__source_row_1based").iterrows():
        num, typ, src = _pick_best_callable(r.to_dict())
        phone_found.append("" if num is None else str(num))
        phone_type.append("" if typ is None else str(typ))
        phone_src.append("" if src is None else str(src))
    # Align to original row order
    merged_aug = merged_aug.sort_values("__source_row_1based").reset_index(drop=True)
    merged_aug["PhoneNumber_Found"] = phone_found
    merged_aug["PhoneType_Found"] = phone_type
    merged_aug["PhoneSources_Found"] = phone_src
    merged_aug["PhoneExtract_Outcome"] = merged_aug.get("Final_Row_Outcome_Reason", "")
    merged_aug["PhoneExtract_FaultCategory"] = merged_aug.get("Determined_Fault_Category", "")

    # Decide output column order: original input columns first, then appended phone columns (stable, readable).
    input_cols = [c for c in original_df.columns if c != "__source_row_1based"]
    appended_cols = [
        "PhoneNumber_Found",
        "PhoneType_Found",
        "PhoneSources_Found",
        "PhoneExtract_Outcome",
        "PhoneExtract_FaultCategory",
    ]
    # Include all repaired results columns too (so augmented contains the full phone output set).
    for c in results_df.columns:
        if c not in merged_aug.columns:
            continue
        if c in input_cols:
            continue
        if c in appended_cols:
            continue
        appended_cols.append(c)

    aug_out_cols = input_cols + appended_cols
    merged_aug.to_csv(aug_csv, index=False, encoding="utf-8", sep=input_sep, columns=aug_out_cols)
    # JSONL companion
    with aug_jsonl.open("w", encoding="utf-8") as f:
        for _, row in merged_aug[aug_out_cols].iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    summary = base_dir / "REPAIR_SUMMARY.md"
    summary.write_text(
        "# Repair summary\n\n"
        f"- Base run: `{args.base_run_id}`\n"
        f"- Total rows expected: {total_rows}\n"
        f"- Base live rows present before repair: {base_live_rows}\n"
        f"- Retry runs merged: {len(used_retries)}\n"
        + "".join([f"  - `{d}`\n" for d in used_retries])
        + "\nRepaired artifacts written:\n"
        f"- `{merged_csv.name}`\n"
        f"- `{merged_jsonl.name}`\n"
        f"- `{final_csv.name}`\n"
        f"- `{final_jsonl.name}`\n"
        f"- `{aug_csv.name}`\n"
        f"- `{aug_jsonl.name}`\n",
        encoding="utf-8",
    )

    required = [merged_csv, merged_jsonl, final_csv, final_jsonl, aug_csv, aug_jsonl]
    if not all(_nonempty(p) for p in required):
        raise RuntimeError("Refusing to delete live files: repaired outputs are missing/empty.")

    # Cleanup stale live files (best-effort)
    for p in [base_results_live_csv, base_results_live_jsonl, base_aug_live_csv, base_aug_live_jsonl, base_status_json]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    print("OK repaired:", args.base_run_id)
    print("Repaired files:", final_csv.name, final_jsonl.name, aug_csv.name, aug_jsonl.name)


if __name__ == "__main__":
    main()

