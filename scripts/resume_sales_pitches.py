import argparse
import csv
import os
import sys
from urllib.parse import urlparse
from typing import List, Tuple, Optional, Set

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import main_pipeline


def _detect_csv_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
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


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_set = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_set:
            return cols_set[cand.lower()]
    return None


def _normalize_key_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    # Ensure parseable
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
        host = (parsed.netloc or "").lower().strip()
        host = host.split("@")[-1].split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        path = (parsed.path or "").rstrip("/")
        # Keep path because some sources store pathful URLs
        return f"{host}{path}".strip()
    except Exception:
        return raw.lower().rstrip("/")


def _build_done_keys(resume_df: pd.DataFrame) -> Set[Tuple[str, str]]:
    cols = list(resume_df.columns)
    company_col = _pick_col(cols, ["CompanyName", "Company"])
    url_col = _pick_col(cols, ["CanonicalEntryURL", "GivenURL", "Website", "URL"])
    pitch_col = _pick_col(cols, ["sales_pitch", "Sales Pitch", "Pitch"])
    if not company_col or not url_col:
        raise ValueError(f"Could not find CompanyName/Company and GivenURL/Website columns in resume file. Found columns: {cols}")
    if not pitch_col:
        # If no pitch column, treat all rows in resume as "done"
        pitch_col = None

    done: Set[Tuple[str, str]] = set()
    for _, row in resume_df.iterrows():
        company = str(row.get(company_col, "")).strip().lower()
        url = _normalize_key_url(str(row.get(url_col, "")))
        if not company or not url:
            continue
        if pitch_col is None:
            done.add((company, url))
            continue
        pitch = str(row.get(pitch_col, "")).strip()
        if pitch and pitch.lower() not in {"nan", "none", "null"}:
            done.add((company, url))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume sales pitch generation by skipping rows already containing a sales_pitch in an existing report.")
    ap.add_argument("--input", required=True, help="Augmented input CSV (ideally output of phone_extract) to generate sales pitches for.")
    ap.add_argument("--input-profile", default="company_semicolon_phone_found", help="Input profile for the augmented input.")
    ap.add_argument("--resume-from", required=True, help="Existing live/final SalesOutreachReport CSV used to determine which rows are already done.")
    ap.add_argument("--workers", type=int, default=20, help="Number of workers for end-to-end full pipeline.")
    ap.add_argument("--suffix", default="resume_pitches", help="Run id suffix for the new resume run.")
    ap.add_argument("--reuse-scraped-content-from", action="append", default=[], help="Path(s) to phone_extract run dir or scraped_content dir(s). Can be repeated.")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit to first N remaining rows (useful for smoke testing). 0 means no limit.")
    ap.add_argument("--run-prequalification", action="store_true", default=False, help="Run B2B/capacity checks (by default, resume runs skip prequalification).")
    ap.add_argument("--pitch-from-description", action="store_true", default=False, help="Build pitch from description only (no scraping).")
    args = ap.parse_args()

    input_path = os.path.normpath(args.input)
    resume_path = os.path.normpath(args.resume_from)
    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")
    if not os.path.exists(resume_path):
        raise SystemExit(f"Resume file not found: {resume_path}")

    in_sep = _detect_csv_delimiter(input_path)
    resume_sep = _detect_csv_delimiter(resume_path)

    input_df = pd.read_csv(input_path, sep=in_sep, engine="python", dtype=str, keep_default_na=False, na_filter=False)
    resume_df = pd.read_csv(resume_path, sep=resume_sep, engine="python", dtype=str, keep_default_na=False, na_filter=False)

    done_keys = _build_done_keys(resume_df)

    cols = list(input_df.columns)
    company_col = _pick_col(cols, ["CompanyName", "Company"])
    url_col = _pick_col(cols, ["CanonicalEntryURL", "GivenURL", "Website", "URL"])
    if not company_col or not url_col:
        raise SystemExit(f"Could not find CompanyName/Company and GivenURL/Website columns in input file. Found columns: {cols}")

    mask = []
    for _, row in input_df.iterrows():
        company = str(row.get(company_col, "")).strip().lower()
        url = _normalize_key_url(str(row.get(url_col, "")))
        if not company or not url:
            mask.append(True)  # keep (let pipeline log failures)
            continue
        mask.append((company, url) not in done_keys)

    remaining_df = input_df.loc[mask].copy()
    if args.limit and int(args.limit) > 0:
        remaining_df = remaining_df.head(int(args.limit)).copy()
    if remaining_df.empty:
        print("No remaining rows to process (all rows already have a sales_pitch in the resume file). Exiting.")
        return

    # Write filtered input into output_data (never touch the input data directory)
    out_dir = os.path.join("output_data", f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.suffix}_input")
    os.makedirs(out_dir, exist_ok=True)
    resume_input_path = os.path.join(out_dir, "resume_input.csv")
    remaining_df.to_csv(resume_input_path, index=False, sep=in_sep, encoding="utf-8-sig")

    # Call main_pipeline in-process with parallel workers and scrape reuse
    ns = argparse.Namespace(
        input_file=resume_input_path,
        range="",
        suffix=args.suffix,
        test=False,
        input_profile=args.input_profile,
        skip_prequalification=not bool(args.run_prequalification),
        pitch_from_description=args.pitch_from_description,
        force_phone_extraction=False,
        skip_phone_retrieval=True,
        workers=int(args.workers),
        reuse_scraped_content_from=list(args.reuse_scraped_content_from or []),
    )
    main_pipeline.main(ns)


if __name__ == "__main__":
    main()

