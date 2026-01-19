"""
Batch runner for the "5-30 Apollo new runs" dataset.

Goal:
- Keep input folders read-only (never write into data/).
- Combine the 3 input CSVs into a single combined CSV under output_data/.
- Run phone extraction with N workers (default 10) on the combined input.
- Run the full pipeline once on the enriched phone output to generate ONE combined SalesOutreach report.

Notes:
- phone_extract.py supports --workers (multiprocessing). main_pipeline.py is currently single-process.
- We intentionally run phone extraction first (parallel), then sales pitch generation (single-process).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


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


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _combine_inputs(input_files: List[Path], out_csv: Path) -> None:
    dfs: List[pd.DataFrame] = []
    for p in input_files:
        sep = _detect_csv_delimiter(p)
        df = pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False, na_filter=False)
        df["__source_file"] = p.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write semicolon CSV (matches your Apollo exports and opens nicely in DE Excel locales).
    combined.to_csv(out_csv, sep=";", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"data\5-30apollo-new-runs",
        help="Directory containing source_rows_augmented*.csv files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Worker processes for phone extraction (phone_extract.py).",
    )
    parser.add_argument(
        "--row-range",
        type=str,
        default="",
        help="Optional row range like '1-200' for a smaller test run. Empty = all rows.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="5_30_batch",
        help="Suffix for run IDs / output folders.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = (repo_root / args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    input_files = [
        input_dir / "source_rows_augmented.csv",
        input_dir / "source_rows_augmented2.csv",
        input_dir / "source_rows_augmented3.csv",
    ]
    missing = [str(p) for p in input_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    out_base = repo_root / "output_data" / f"{_timestamp_id()}_{args.suffix}"
    combined_input = out_base / "combined_input_5_30.csv"
    _combine_inputs(input_files, combined_input)

    # 1) Phone extraction (parallel)
    phone_suffix = f"{args.suffix}_phone{args.workers}"
    phone_cmd = [
        "python",
        "phone_extract.py",
        "-i",
        str(combined_input),
        "-s",
        phone_suffix,
        "--workers",
        str(args.workers),
        "--input-profile",
        "company_semicolon_phone_extract",
    ]
    if args.row_range:
        phone_cmd.extend(["-r", args.row_range])

    print("Running phone_extract:", " ".join(phone_cmd))
    env = os.environ.copy()
    # Force German outputs for the full pipeline summarization + attribute extraction.
    env.setdefault("PROMPT_PATH_WEBSITE_SUMMARIZER", "prompts/website_summarizer_prompt_de.txt")
    env.setdefault("PROMPT_PATH_ATTRIBUTE_EXTRACTOR", "prompts/attribute_extractor_prompt_de.txt")
    env.setdefault("SALES_PROMPT_LANGUAGE", "de")
    subprocess.check_call(phone_cmd, cwd=str(repo_root), env=env)

    # Find the phone_extract output dir by suffix (latest match).
    output_data_dir = repo_root / "output_data"
    candidates = sorted(
        [p for p in output_data_dir.iterdir() if p.is_dir() and str(p.name).endswith(phone_suffix)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"Could not locate phone_extract output dir ending with suffix: {phone_suffix}")
    phone_run_dir = candidates[0]

    # Prefer the augmented CSV if available.
    augmented_candidates = sorted(phone_run_dir.glob("input_augmented_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not augmented_candidates:
        raise RuntimeError(f"No input_augmented_*.csv found in phone run dir: {phone_run_dir}")
    augmented_input = augmented_candidates[0]

    # 2) Full pipeline (single-process), using enriched phone column
    full_cmd = [
        "python",
        "main_pipeline.py",
        "-i",
        str(augmented_input),
        "-s",
        f"{args.suffix}_full",
        "--input-profile",
        "company_semicolon_phone_found",
        "--skip-prequalification",
        "--skip-phone-retrieval",
    ]
    # Run same slice if user passed row-range (handy for smoke runs)
    if args.row_range:
        full_cmd.extend(["-r", args.row_range])

    print("Running main_pipeline:", " ".join(full_cmd))
    subprocess.check_call(full_cmd, cwd=str(repo_root), env=env)

    print("Done.")
    print("Combined input:", combined_input)
    print("Phone-enriched input:", augmented_input)


if __name__ == "__main__":
    main()

