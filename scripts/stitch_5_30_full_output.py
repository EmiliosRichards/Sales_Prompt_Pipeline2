"""
Stitch together the 5-30 Apollo "phone extraction augmented" CSV with the
SalesOutreachReport CSVs produced by the (partial) original run + the resume run.

Why this exists:
- The phone extraction run (`*_phone50`) produced `input_augmented_*.csv` for *all* inputs (semicolon-delimited).
- The original full pipeline run produced some `SalesOutreachReport_*_live.csv` rows (comma-delimited).
- The resume run produced additional `SalesOutreachReport_*.csv` rows (comma-delimited).

This script produces:
- A single combined SalesOutreachReport CSV + JSONL (comma-delimited CSV).
- A single combined input_augmented CSV + JSONL (semicolon-delimited CSV) by overlaying
  SalesOutreach fields (e.g., `sales_pitch`, matching fields, extracted attributes) onto the base rows.

Usage (PowerShell):
  python scripts\\stitch_5_30_full_output.py `
    --base-augmented \"output_data\\20260119_133439_5_30_FULL_RUN_phone50\\input_augmented_20260119_133439_5_30_FULL_RUN_phone50.csv\" `
    --sales-report \"output_data\\20260119_135323_5_30_FULL_RUN_full\\SalesOutreachReport_20260119_135323_5_30_FULL_RUN_full_live.csv\" `
    --sales-report \"output_data\\20260119_153829_resume_FULL_w50\\SalesOutreachReport_20260119_153829_resume_FULL_w50.csv\"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


def _norm_url_key(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z]+://", url):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        path = (parsed.path or "").rstrip("/")
        return f"{host}{path}"
    except Exception:
        return url.strip().lower().rstrip("/")


def _norm_host_key(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z]+://", url):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return url.strip().lower().rstrip("/")


def _format_avg_leads_per_day(raw: str) -> str:
    """
    Convert values like '8.0' -> '8', '8,5' -> '8.5', leave non-numeric strings as-is.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    # Handle german decimal comma.
    s2 = s.replace(",", ".")
    try:
        v = float(s2)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        # Keep up to 2 decimals, trimmed.
        out = f"{v:.2f}".rstrip("0").rstrip(".")
        return out
    except Exception:
        return s


_PLACEHOLDER_TOKEN = "{programmatic placeholder}"


def _fill_programmatic_placeholder(text: str, avg_leads_per_day_raw: str) -> str:
    t = text or ""
    if _PLACEHOLDER_TOKEN not in t:
        return t
    formatted = _format_avg_leads_per_day(avg_leads_per_day_raw)
    if not formatted:
        return t
    return t.replace(_PLACEHOLDER_TOKEN, formatted)


def _company_name_from_row(row: Dict[str, str]) -> str:
    return ((row.get("CompanyName") or "") or (row.get("Company") or "")).strip()


def _pick_sales_url(row: Dict[str, str]) -> str:
    return (row.get("CanonicalEntryURL") or row.get("GivenURL") or "").strip()


@dataclass
class LoadedCsv:
    header: List[str]
    rows: List[Dict[str, str]]


def _read_csv_dicts(path: Path, delimiter: str) -> LoadedCsv:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        raw_header = list(reader.fieldnames or [])
        # Normalize BOM-prefixed headers (common when CSVs are written with a UTF-8 BOM).
        bom_renames: Dict[str, str] = {}
        for h in raw_header:
            if h.startswith("\ufeff"):
                cleaned = h.lstrip("\ufeff")
                if cleaned and cleaned not in raw_header:
                    bom_renames[h] = cleaned

        header: List[str] = []
        for h in raw_header:
            if h in bom_renames:
                if bom_renames[h] not in header:
                    header.append(bom_renames[h])
            else:
                header.append(h)

        rows: List[Dict[str, str]] = []
        for r in reader:
            d = dict(r)
            for old, new in bom_renames.items():
                if old in d and new not in d:
                    d[new] = d.get(old, "")
                if old in d:
                    del d[old]
            rows.append(d)
    return LoadedCsv(header=header, rows=rows)


def _write_csv_dicts(path: Path, header: List[str], rows: Iterable[Dict[str, str]], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _merge_headers(primary: List[str], extra: List[str]) -> List[str]:
    seen = set(primary)
    out = list(primary)
    for c in extra:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _build_sales_map(
    sales_reports: List[LoadedCsv],
    prefer_later: bool = True,
) -> Tuple[List[str], Dict[str, Dict[str, str]], Dict[str, Optional[Dict[str, str]]], List[Dict[str, str]]]:
    """
    Returns:
      - merged_header
      - map from normalized URL key -> row dict (later reports override earlier if prefer_later)
      - map from normalized HOST key -> row dict IF SAFE (None indicates host collision / ambiguous)
      - list of rows with no usable URL key (orphans)
    """
    merged_header: List[str] = []
    sales_by_key: Dict[str, Dict[str, str]] = {}
    sales_by_host: Dict[str, Optional[Dict[str, str]]] = {}
    orphans: List[Dict[str, str]] = []

    reports = sales_reports if prefer_later else list(reversed(sales_reports))
    for rep in reports:
        merged_header = _merge_headers(merged_header, rep.header)
        for row in rep.rows:
            # Post-process old runs: fill any remaining {programmatic placeholder} using Avg Leads Per Day.
            avg_raw = (row.get("Avg Leads Per Day") or row.get("avg_leads_per_day") or "").strip()
            # Excel (especially DE locale) can misread strings like "8.0" as "80".
            # Normalize the stored value to a safe string (e.g. "8" instead of "8.0").
            avg_formatted = _format_avg_leads_per_day(avg_raw)
            if avg_formatted:
                if "Avg Leads Per Day" in row:
                    row["Avg Leads Per Day"] = avg_formatted
                if "avg_leads_per_day" in row:
                    row["avg_leads_per_day"] = avg_formatted
            if "sales_pitch" in row:
                row["sales_pitch"] = _fill_programmatic_placeholder(row.get("sales_pitch", ""), avg_formatted or avg_raw)
            if "phone_sales_line" in row:
                row["phone_sales_line"] = _fill_programmatic_placeholder(row.get("phone_sales_line", ""), avg_formatted or avg_raw)

            url = _pick_sales_url(row)
            key = _norm_url_key(url)
            if not key:
                orphans.append(row)
                continue
            sales_by_key[key] = row

            hkey = _norm_host_key(url)
            if hkey:
                # Mark host ambiguous if we ever see it with different company names across rows.
                existing = sales_by_host.get(hkey)
                if existing is None:
                    continue
                if existing is not None:
                    if _company_name_from_row(existing).strip().lower() != _company_name_from_row(row).strip().lower():
                        sales_by_host[hkey] = None
                    else:
                        # Same company name: keep the newer (current precedence order).
                        sales_by_host[hkey] = row
                else:
                    sales_by_host[hkey] = row

    return merged_header, sales_by_key, sales_by_host, orphans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-augmented", required=True, help="Semicolon-delimited input_augmented CSV from the phone extraction run")
    ap.add_argument(
        "--sales-report",
        action="append",
        required=True,
        help="Comma-delimited SalesOutreachReport CSV(s). Provide multiple times; later wins by default.",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output folder (default: output_data/<timestamp>_5_30_STITCHED)",
    )
    ap.add_argument(
        "--host-fallback",
        action="store_true",
        help="If exact URL+path key doesn't match, fall back to matching by host (domain) when unambiguous.",
    )
    args = ap.parse_args()

    base_path = Path(args.base_augmented)
    sales_paths = [Path(p) for p in (args.sales_report or [])]
    if len(sales_paths) < 1:
        raise SystemExit("Need at least one --sales-report")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("output_data") / f"{stamp}_5_30_STITCHED"

    # Load base augmented (semicolon).
    base = _read_csv_dicts(base_path, delimiter=";")
    base_hosts = {_norm_host_key((r.get("Website") or "").strip()) for r in base.rows}
    base_hosts.discard("")
    base_url_keys = {_norm_url_key((r.get("Website") or "").strip()) for r in base.rows}
    base_url_keys.discard("")

    # Load all sales reports (comma).
    sales_reports = [_read_csv_dicts(p, delimiter=",") for p in sales_paths]

    sales_header, sales_by_key, sales_by_host, sales_orphans = _build_sales_map(sales_reports, prefer_later=True)

    # --- Write combined SalesOutreachReport ---
    combined_sales_rows = list(sales_by_key.values())
    combined_sales_csv = out_dir / "SalesOutreachReport_5_30_STITCHED.csv"
    combined_sales_jsonl = out_dir / "SalesOutreachReport_5_30_STITCHED.jsonl"
    _write_csv_dicts(combined_sales_csv, header=sales_header, rows=combined_sales_rows, delimiter=",")
    _write_jsonl(combined_sales_jsonl, combined_sales_rows)

    # Also write a version that is strictly limited to rows that belong to the base input set.
    # This avoids confusion when older runs produced SalesOutreach rows for other datasets.
    combined_sales_rows_only_inputs: List[Dict[str, str]] = []
    for r in combined_sales_rows:
        u = _pick_sales_url(r)
        k = _norm_url_key(u)
        h = _norm_host_key(u)
        if (k and k in base_url_keys) or (h and h in base_hosts):
            combined_sales_rows_only_inputs.append(r)

    combined_sales_only_inputs_csv = out_dir / "SalesOutreachReport_5_30_STITCHED_only_inputs.csv"
    combined_sales_only_inputs_jsonl = out_dir / "SalesOutreachReport_5_30_STITCHED_only_inputs.jsonl"
    _write_csv_dicts(combined_sales_only_inputs_csv, header=sales_header, rows=combined_sales_rows_only_inputs, delimiter=",")
    _write_jsonl(combined_sales_only_inputs_jsonl, combined_sales_rows_only_inputs)

    # --- Build combined augmented by overlaying sales fields onto base rows ---
    base_header = base.header
    base_header_set = set(base_header)
    augmented_header = _merge_headers(base_header, sales_header)
    augmented_rows: List[Dict[str, str]] = []
    unmatched_base = 0
    for row in base.rows:
        website = (row.get("Website") or "").strip()
        key = _norm_url_key(website)
        sales_row = sales_by_key.get(key)
        if sales_row is None and args.host_fallback:
            hkey = _norm_host_key(website)
            sales_row = sales_by_host.get(hkey) if hkey else None
        merged = dict(row)
        if sales_row:
            # Overlay sales columns WITHOUT clobbering original/base columns.
            for k in sales_header:
                sv = sales_row.get(k, "")
                if k in base_header_set:
                    # Preserve base value; only fill if base is empty and sales has value.
                    bv = (merged.get(k, "") or "").strip()
                    if (not bv) and sv != "":
                        merged[k] = sv
                    else:
                        merged.setdefault(k, merged.get(k, ""))
                else:
                    if sv != "":
                        merged[k] = sv
                    else:
                        merged.setdefault(k, "")
        else:
            unmatched_base += 1
            for k in sales_header:
                merged.setdefault(k, "")
        augmented_rows.append(merged)

    combined_aug_csv = out_dir / "input_augmented_5_30_STITCHED.csv"
    combined_aug_jsonl = out_dir / "input_augmented_5_30_STITCHED.jsonl"
    _write_csv_dicts(combined_aug_csv, header=augmented_header, rows=augmented_rows, delimiter=";")
    _write_jsonl(combined_aug_jsonl, augmented_rows)

    # --- Write a small summary + orphans for debugging ---
    summary = {
        "base_augmented_rows": len(base.rows),
        "sales_reports": [str(p) for p in sales_paths],
        "combined_sales_rows": len(combined_sales_rows),
        "combined_sales_rows_only_inputs": len(combined_sales_rows_only_inputs),
        "base_rows_without_sales_match": unmatched_base,
        "sales_rows_without_base_match": len(
            [k for k in sales_by_key.keys() if k not in {_norm_url_key((r.get('Website') or '').strip()) for r in base.rows}]
        ),
        "sales_orphan_rows_no_url": len(sales_orphans),
    }
    (out_dir / "STITCH_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if sales_orphans:
        _write_csv_dicts(out_dir / "SalesOutreachReport_orphans_no_url.csv", header=sales_header, rows=sales_orphans, delimiter=",")

    print(f"Wrote: {combined_sales_csv}")
    print(f"Wrote: {combined_sales_jsonl}")
    print(f"Wrote: {combined_aug_csv}")
    print(f"Wrote: {combined_aug_jsonl}")
    print(f"Wrote: {out_dir / 'STITCH_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

