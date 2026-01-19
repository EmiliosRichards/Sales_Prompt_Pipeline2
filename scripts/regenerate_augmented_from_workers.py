import argparse
import glob
import os
import re
import sys

import pandas as pd


def _worker_index_from_path(path: str) -> int:
    # Expected: .../workers/w{idx}of{N}/phone_extraction_results_*.csv
    m = re.search(r"workers[\\/]+w(\d+)of", path)
    return int(m.group(1)) if m else 10**9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate merged + augmented outputs from existing worker CSVs (no re-scrape / no LLM calls)."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing workers/ (e.g. output_data/20260113_143902_full25).",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Original input CSV used for the run (for augmented output).",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based start row used for the run (default: 1).",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="FIXED",
        help="Suffix appended to output filenames (default: FIXED).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    input_csv = args.input_csv
    start_row_1based = int(args.start_row)
    out_suffix = args.out_suffix.strip() or "FIXED"

    workers_glob = os.path.join(run_dir, "workers", "w*of*", "phone_extraction_results_*.csv")
    worker_paths = glob.glob(workers_glob)
    if not worker_paths:
        raise SystemExit(f"No worker outputs found under: {workers_glob}")

    worker_paths_sorted = sorted(worker_paths, key=_worker_index_from_path)

    merged_df = pd.concat(
        [pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False) for p in worker_paths_sorted],
        ignore_index=True,
    )

    merged_out = os.path.join(run_dir, f"phone_extraction_results_{out_suffix}_merged.csv")
    merged_df.to_csv(merged_out, index=False, encoding="utf-8")

    # Reuse the same augmented-writer used by phone_extract.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import phone_extract

    augmented_out = os.path.join(run_dir, f"input_augmented_{out_suffix}.csv")
    phone_extract._write_augmented_csv(os.path.abspath(input_csv), start_row_1based, merged_df, augmented_out)

    print(f"Wrote merged: {merged_out}")
    print(f"Wrote augmented: {augmented_out}")


if __name__ == "__main__":
    main()

