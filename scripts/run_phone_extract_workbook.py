"""
Prepare a multi-sheet workbook for phone extraction, optionally run it, and
optionally rebuild a multi-sheet workbook from the augmented results.

This is useful for Apollo-style Excel exports where each sheet has the same
schema (`Company Name`, `Website`, `Company Phone`, etc.) but the current
phone pipeline only processes one sheet/file at a time.

Workflow:
1. Read every sheet (or a selected subset).
2. Add provenance columns so rows can be traced back to workbook + sheet.
3. Optionally drop rows with blank Website values.
4. Write one combined CSV.
5. Optionally invoke `phone_extract.py` on that combined CSV.
6. Optionally write a new workbook with rows merged back into their
   original sheets and the phone output columns appended.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


PHONE_WRITEBACK_COLUMNS = [
    "PhoneNumber_Found",
    "PhoneType_Found",
    "PhoneSources_Found",
    "PhoneExtract_Outcome",
    "PhoneExtract_FaultCategory",
    "Top_Number_1",
    "Top_Type_1",
    "Top_SourceURL_1",
    "Top_Number_2",
    "Top_Type_2",
    "Top_SourceURL_2",
    "Top_Number_3",
    "Top_Type_3",
    "Top_SourceURL_3",
    "CanonicalEntryURL",
    "Final_Row_Outcome_Reason",
    "HttpFallbackAttempted",
    "HttpFallbackUsed",
    "HttpFallbackResult",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine workbook sheets into one CSV for phone extraction."
    )
    parser.add_argument(
        "--input-xlsx",
        required=True,
        help="Path to the source XLSX workbook.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Path for the combined CSV. Defaults to data/prepared/<stem>__all_sheets.csv",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        default=[],
        help="Optional sheet name to include. Repeat to include multiple sheets.",
    )
    parser.add_argument(
        "--keep-blank-url-rows",
        action="store_true",
        help="Keep rows where Website is blank. Default is to drop them.",
    )
    parser.add_argument(
        "--run-phone-extract",
        action="store_true",
        help="After writing the combined CSV, run phone_extract.py on it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker count for phone_extract.py when --run-phone-extract is used.",
    )
    parser.add_argument(
        "--suffix",
        default="workbook_phone",
        help="Run suffix for phone_extract.py when --run-phone-extract is used.",
    )
    parser.add_argument(
        "--input-profile",
        default="apollo_shopware_partner_germany",
        help="Input profile for phone_extract.py.",
    )
    parser.add_argument(
        "--write-back-workbook",
        action="store_true",
        help="Write a new multi-sheet workbook with phone results merged back into the original sheets.",
    )
    parser.add_argument(
        "--augmented-csv",
        default="",
        help="Existing input_augmented CSV to use for workbook write-back. If omitted after --run-phone-extract, the script auto-detects the run output.",
    )
    parser.add_argument(
        "--output-xlsx",
        default="",
        help="Path for the rebuilt workbook. Defaults to data/prepared/<stem>__phone_augmented.xlsx",
    )
    return parser.parse_args()


def _default_output_path(input_xlsx: Path) -> Path:
    safe_stem = input_xlsx.stem.replace(" ", "_")
    return input_xlsx.parent / "prepared" / f"{safe_stem}__all_sheets.csv"


def _default_workbook_output_path(input_xlsx: Path) -> Path:
    safe_stem = input_xlsx.stem.replace(" ", "_")
    return input_xlsx.parent / "prepared" / f"{safe_stem}__phone_augmented.xlsx"


def _selected_sheet_names(all_sheet_names: List[str], requested: Iterable[str]) -> List[str]:
    requested = list(requested)
    if not requested:
        return all_sheet_names

    missing = [name for name in requested if name not in all_sheet_names]
    if missing:
        raise ValueError(f"Requested sheet(s) not found: {missing}")
    return requested


def _combine_workbook(input_xlsx: Path, sheet_names: List[str], keep_blank_url_rows: bool) -> tuple[pd.DataFrame, List[dict]]:
    xls = pd.ExcelFile(input_xlsx)
    frames: List[pd.DataFrame] = []
    summary: List[dict] = []

    for sheet_name in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, keep_default_na=False)
        original_rows = len(df)

        # Preserve provenance so augmented outputs can always be mapped back.
        df.insert(0, "SourceWorkbook", input_xlsx.name)
        df.insert(1, "SourceSheet", sheet_name)
        df.insert(2, "SourceSheetRowNumber", range(2, len(df) + 2))

        website_series = df["Website"].astype(str).str.strip() if "Website" in df.columns else pd.Series([""] * len(df))
        if keep_blank_url_rows:
            kept_df = df
        else:
            kept_df = df.loc[website_series != ""].copy()

        frames.append(kept_df)
        summary.append(
            {
                "sheet_name": sheet_name,
                "rows_total": original_rows,
                "rows_kept": len(kept_df),
                "rows_dropped_blank_website": original_rows - len(kept_df),
            }
        )

    if not frames:
        return pd.DataFrame(), summary

    combined = pd.concat(frames, ignore_index=True)
    return combined, summary


def _run_phone_extract(output_csv: Path, workers: int, suffix: str, input_profile: str) -> int:
    cmd = [
        sys.executable,
        "phone_extract.py",
        "-i",
        str(output_csv),
        "-s",
        suffix,
        "--input-profile",
        input_profile,
        "--workers",
        str(max(1, workers)),
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode or 0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_augmented_csv_for_input(input_csv: Path) -> Optional[Path]:
    output_data_dir = _repo_root() / "output_data"
    if not output_data_dir.exists():
        return None

    input_csv_str = str(input_csv.resolve())
    manifest_paths = sorted(output_data_dir.glob("*/run_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        if manifest.get("pipeline_name") != "phone_extract":
            continue
        if manifest.get("status") != "completed":
            continue
        input_meta = manifest.get("input_file") or {}
        if not isinstance(input_meta, dict) or input_meta.get("path") != input_csv_str:
            continue

        run_id = manifest.get("run_id")
        if not run_id:
            continue

        candidate = manifest_path.parent / f"input_augmented_{run_id}.csv"
        if candidate.exists():
            return candidate
    return None


def _read_csv_auto(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=None, engine="python", keep_default_na=False, na_filter=False)


def _write_back_workbook(input_xlsx: Path, augmented_csv: Path, output_xlsx: Path) -> None:
    augmented_df = _read_csv_auto(augmented_csv)
    required_cols = {"SourceSheet", "SourceSheetRowNumber"}
    missing_required = [col for col in required_cols if col not in augmented_df.columns]
    if missing_required:
        raise ValueError(
            f"Augmented CSV is missing required provenance columns for workbook reconstruction: {missing_required}"
        )

    available_phone_cols = [col for col in PHONE_WRITEBACK_COLUMNS if col in augmented_df.columns]
    xls = pd.ExcelFile(input_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            original_df = pd.read_excel(xls, sheet_name=sheet_name, keep_default_na=False)
            original_df = original_df.copy()
            original_df["__sheet_row_number__"] = range(2, len(original_df) + 2)

            sheet_aug = augmented_df.loc[augmented_df["SourceSheet"] == sheet_name].copy()
            if "SourceSheetRowNumber" in sheet_aug.columns:
                sheet_aug["SourceSheetRowNumber"] = pd.to_numeric(sheet_aug["SourceSheetRowNumber"], errors="coerce")

            merge_cols = ["SourceSheetRowNumber"] + available_phone_cols
            merged_df = original_df.merge(
                sheet_aug.loc[:, merge_cols],
                how="left",
                left_on="__sheet_row_number__",
                right_on="SourceSheetRowNumber",
            )
            merged_df.drop(columns=["__sheet_row_number__", "SourceSheetRowNumber"], inplace=True, errors="ignore")
            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Wrote reconstructed workbook: {output_xlsx}")
    print(f"Phone columns appended: {', '.join(available_phone_cols) if available_phone_cols else '(none found in augmented CSV)'}")


def main() -> int:
    args = parse_args()
    input_xlsx = Path(args.input_xlsx).resolve()
    if not input_xlsx.exists():
        print(f"Input workbook not found: {input_xlsx}", file=sys.stderr)
        return 1

    output_csv = Path(args.output_csv).resolve() if args.output_csv else _default_output_path(input_xlsx).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(input_xlsx)
    sheet_names = _selected_sheet_names(xls.sheet_names, args.sheet)

    combined_df, summary = _combine_workbook(
        input_xlsx=input_xlsx,
        sheet_names=sheet_names,
        keep_blank_url_rows=bool(args.keep_blank_url_rows),
    )

    combined_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Wrote combined CSV: {output_csv}")
    print(f"Rows written: {len(combined_df)}")
    for item in summary:
        print(
            f"{item['sheet_name']}\t"
            f"rows_total={item['rows_total']}\t"
            f"rows_kept={item['rows_kept']}\t"
            f"dropped_blank_website={item['rows_dropped_blank_website']}"
        )

    if not args.run_phone_extract:
        if args.write_back_workbook:
            if not args.augmented_csv:
                print("--write-back-workbook requires --augmented-csv when not running phone_extract.", file=sys.stderr)
                return 1
            output_xlsx = Path(args.output_xlsx).resolve() if args.output_xlsx else _default_workbook_output_path(input_xlsx).resolve()
            _write_back_workbook(
                input_xlsx=input_xlsx,
                augmented_csv=Path(args.augmented_csv).resolve(),
                output_xlsx=output_xlsx,
            )
        return 0

    exit_code = _run_phone_extract(
        output_csv=output_csv,
        workers=args.workers,
        suffix=args.suffix,
        input_profile=args.input_profile,
    )
    if exit_code != 0:
        return exit_code

    if args.write_back_workbook:
        augmented_csv = Path(args.augmented_csv).resolve() if args.augmented_csv else _find_augmented_csv_for_input(output_csv)
        if augmented_csv is None or not augmented_csv.exists():
            print(
                "Phone extraction finished, but I could not auto-detect the resulting input_augmented CSV. "
                "Re-run with --augmented-csv <path> to build the workbook.",
                file=sys.stderr,
            )
            return 1
        output_xlsx = Path(args.output_xlsx).resolve() if args.output_xlsx else _default_workbook_output_path(input_xlsx).resolve()
        _write_back_workbook(
            input_xlsx=input_xlsx,
            augmented_csv=augmented_csv,
            output_xlsx=output_xlsx,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
