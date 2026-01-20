"""
Phone-only pipeline entrypoint: scrape → regex candidates → phone LLM classify → consolidate → write outputs.

Key design points:
- Supports `--workers` on Windows via `ProcessPoolExecutor` + `multiprocessing.Manager().Queue()` to stream
  per-row results back to a single master writer (no concurrent file writes).
- Always produces **live** outputs (`*_live.csv`, `*_live.jsonl`, `*_live_status.json`) so you can monitor progress.
- Also copies live outputs into stable final outputs after completion.
- When input is CSV, the augmented CSV preserves the input delimiter (Apollo exports are often `;`).

See:
- `docs/SYSTEM_OVERVIEW.md`
- `docs/OUTPUT_FILES.md`
- `docs/PHONE_NUMBER_FIELDS.md`
"""

import argparse
import csv
import json
import logging
import os
import time
import queue as pyqueue
import threading
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any, Dict

import pandas as pd
from dotenv import load_dotenv

from src.core.config import AppConfig
from src.core.logging_config import setup_logging
from src.phone_retrieval.data_handling.normalizer import normalize_phone_number
from src.phone_retrieval.data_handling.loader import load_and_preprocess_data
from src.phone_retrieval.extractors.llm_extractor import GeminiLLMExtractor
from src.phone_retrieval.processing.pipeline_flow import execute_pipeline_flow
from src.phone_retrieval.reporting.metrics_manager import write_run_metrics
from src.phone_retrieval.utils.helpers import (
    generate_run_id,
    initialize_dataframe_columns,
    initialize_run_metrics,
    resolve_path,
)

# Load .env but do NOT override environment variables already set by the shell.
# This allows easy per-run overrides (e.g., prompt paths) without editing .env.
load_dotenv(override=False)

logger = logging.getLogger(__name__)
BASE_FILE_PATH_FOR_RESOLVE = __file__

def _detect_csv_delimiter(file_path: str) -> str:
    """Detect delimiter for CSV inputs; default to comma if unsure."""
    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the phone extraction pipeline in isolation (no sales prompts)."
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default=r"data\Apollo Shopware Partner Germany 1(Sheet1).csv",
        help="Path to input CSV/XLSX.",
    )
    parser.add_argument(
        "-r",
        "--range",
        type=str,
        default="",
        help="Row range to process (e.g., '1-200', '500-', '-100', '50').",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        default="phone_only",
        help="Suffix appended to run_id for easier identification.",
    )
    parser.add_argument(
        "--input-profile",
        type=str,
        default="apollo_shopware_partner_germany",
        help="Input profile name from AppConfig.INPUT_COLUMN_PROFILES.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Optional explicit output CSV path. If omitted, writes into the run output dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to run row chunks in parallel. Use small values (2-4) to avoid rate limits.",
    )
    return parser.parse_args()


def _setup_run_dirs(app_config: AppConfig, run_id: str) -> Tuple[str, str]:
    """Create run output dir + llm_context dir."""
    output_base_dir_abs = app_config.output_base_dir
    if not os.path.isabs(output_base_dir_abs):
        output_base_dir_abs = os.path.join(os.path.dirname(os.path.abspath(BASE_FILE_PATH_FOR_RESOLVE)), output_base_dir_abs)
    run_output_dir = os.path.join(output_base_dir_abs, run_id)
    llm_context_dir = os.path.join(run_output_dir, app_config.llm_context_subdir)
    os.makedirs(llm_context_dir, exist_ok=True)
    return run_output_dir, llm_context_dir


def _write_failure_header(failure_writer: csv.writer) -> None:
    failure_writer.writerow(
        [
            "log_timestamp",
            "input_row_identifier",
            "CompanyName",
            "GivenURL",
            "stage_of_failure",
            "error_reason",
            "error_details",
            "Associated_Pathful_Canonical_URL",
        ]
    )


@dataclass(frozen=True)
class WorkerJob:
    input_file: str
    input_profile: str
    row_range: str
    run_id: str
    suffix: str
    log_level: str
    console_log_level: str
    output_base_dir: str


def _run_worker(job: WorkerJob) -> str:
    """
    Process a slice of rows in-process and return the path to that worker's results CSV.

    This path is used both in single-worker mode and as a worker sub-task in `--workers` mode.
    """
    pipeline_start_time = time.time()

    app_config = AppConfig(
        input_file_override=job.input_file,
        row_range_override=job.row_range or None,
        run_id_suffix_override=job.suffix,
        test_mode=False,
    )
    app_config.input_file_profile_name = job.input_profile
    app_config.log_level = job.log_level
    app_config.console_log_level = job.console_log_level
    app_config.output_base_dir = job.output_base_dir

    run_metrics = initialize_run_metrics(job.run_id)
    run_output_dir, llm_context_dir = _setup_run_dirs(app_config, job.run_id)
    log_file_path = os.path.join(run_output_dir, f"phone_extract_{job.run_id}.log")
    file_log_level_int = getattr(logging, app_config.log_level.upper(), logging.INFO)
    console_log_level_int = getattr(logging, app_config.console_log_level.upper(), logging.WARNING)
    setup_logging(file_log_level=file_log_level_int, console_log_level=console_log_level_int, log_file_path=log_file_path)

    input_file_path_abs = resolve_path(app_config.input_excel_file_path, BASE_FILE_PATH_FOR_RESOLVE)
    df, original_phone_col_name, _ = load_and_preprocess_data(input_file_path_abs, app_config_instance=app_config)
    if df is None:
        raise RuntimeError(f"[{job.run_id}] Failed to load input; DataFrame is None.")

    run_metrics["data_processing_stats"]["input_rows_count"] = len(df)
    df = initialize_dataframe_columns(df)

    failure_log_csv_path = os.path.join(run_output_dir, f"failed_rows_{job.run_id}.csv")
    llm_extractor = GeminiLLMExtractor(config=app_config)

    failure_log_file_handle = None
    attrition_data_list = []
    canonical_domain_journey_data = {}

    try:
        failure_log_file_handle = open(failure_log_csv_path, "w", newline="", encoding="utf-8")
        failure_writer = csv.writer(failure_log_file_handle)
        _write_failure_header(failure_writer)

        df_processed, attrition_data_list, canonical_domain_journey_data, *_ = execute_pipeline_flow(
            df=df,
            app_config=app_config,
            llm_extractor=llm_extractor,
            run_output_dir=run_output_dir,
            llm_context_dir=llm_context_dir,
            run_id=job.run_id,
            failure_writer=failure_writer,
            run_metrics=run_metrics,
            original_phone_col_name_for_profile=original_phone_col_name,
        )

        output_path = os.path.join(run_output_dir, f"phone_extraction_results_{job.run_id}.csv")
        df_processed.to_csv(output_path, index=False, encoding="utf-8")
        # JSONL companion (one object per row) for downstream tooling.
        try:
            jsonl_path = os.path.splitext(output_path)[0] + ".jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for _, r in df_processed.iterrows():
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass
    finally:
        if failure_log_file_handle:
            try:
                failure_log_file_handle.close()
            except Exception:
                pass

    run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
    write_run_metrics(
        metrics=run_metrics,
        output_dir=run_output_dir,
        run_id=job.run_id,
        pipeline_start_time=pipeline_start_time,
        attrition_data_list_for_metrics=attrition_data_list,
        canonical_domain_journey_data=canonical_domain_journey_data,
        logger=logging.getLogger(__name__),
    )

    return output_path


def _parse_start_1based(row_range: str) -> int:
    rr = (row_range or "").strip()
    if not rr:
        return 1
    if "-" in rr:
        a = rr.split("-", 1)[0].strip()
        return int(a) if a.isdigit() else 1
    return 1


def _run_worker_streaming(job: WorkerJob, queue_obj: Any) -> str:
    """
    Worker wrapper used in `--workers` mode.

    Streams finalized per-row outputs to the master writer via `queue_obj`.
    The master writer is responsible for writing `*_live.csv` + `*_live.jsonl`.
    """
    try:
        queue_obj.put({"type": "job_start", "job_id": job.run_id, "row_range": job.row_range, "ts": datetime.now().isoformat()})
    except Exception:
        pass

    pipeline_start_time = time.time()
    app_cfg = AppConfig(
        input_file_override=job.input_file,
        row_range_override=job.row_range or None,
        run_id_suffix_override=job.suffix,
        test_mode=False,
    )
    app_cfg.input_file_profile_name = job.input_profile
    app_cfg.log_level = job.log_level
    app_cfg.console_log_level = job.console_log_level
    app_cfg.output_base_dir = job.output_base_dir

    run_metrics = initialize_run_metrics(job.run_id)
    run_output_dir_w, llm_context_dir_w = _setup_run_dirs(app_cfg, job.run_id)
    log_file_path_w = os.path.join(run_output_dir_w, f"phone_extract_{job.run_id}.log")
    file_log_level_int = getattr(logging, app_cfg.log_level.upper(), logging.INFO)
    console_log_level_int = getattr(logging, app_cfg.console_log_level.upper(), logging.WARNING)
    setup_logging(file_log_level=file_log_level_int, console_log_level=console_log_level_int, log_file_path=log_file_path_w)

    df, original_phone_col_name, _ = load_and_preprocess_data(job.input_file, app_config_instance=app_cfg)
    if df is None:
        raise RuntimeError(f"[{job.run_id}] Failed to load input; DataFrame is None.")

    # Attach absolute row number for stable ordering/debug (best-effort).
    start_1based = _parse_start_1based(job.row_range)
    df["__source_row_1based"] = list(range(int(start_1based), int(start_1based) + len(df)))

    run_metrics["data_processing_stats"]["input_rows_count"] = len(df)
    df = initialize_dataframe_columns(df)

    failure_log_csv_path = os.path.join(run_output_dir_w, f"failed_rows_{job.run_id}.csv")
    llm_extractor = GeminiLLMExtractor(config=app_cfg)
    attrition_data_list = []
    canonical_domain_journey_data = {}

    with open(failure_log_csv_path, "w", newline="", encoding="utf-8") as fh_fail:
        failure_writer = csv.writer(fh_fail)
        _write_failure_header(failure_writer)
        df_processed, attrition_data_list, canonical_domain_journey_data, *_ = execute_pipeline_flow(
            df=df,
            app_config=app_cfg,
            llm_extractor=llm_extractor,
            run_output_dir=run_output_dir_w,
            llm_context_dir=llm_context_dir_w,
            run_id=job.run_id,
            failure_writer=failure_writer,
            run_metrics=run_metrics,
            original_phone_col_name_for_profile=original_phone_col_name,
            row_queue=queue_obj,
            row_queue_meta={"job_id": job.run_id, "row_range": job.row_range},
        )

    output_path = os.path.join(run_output_dir_w, f"phone_extraction_results_{job.run_id}.csv")
    df_processed.to_csv(output_path, index=False, encoding="utf-8")

    run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
    write_run_metrics(
        metrics=run_metrics,
        output_dir=run_output_dir_w,
        run_id=job.run_id,
        pipeline_start_time=pipeline_start_time,
        attrition_data_list_for_metrics=attrition_data_list,
        canonical_domain_journey_data=canonical_domain_journey_data,
        logger=logging.getLogger(__name__),
    )
    try:
        queue_obj.put({"type": "job_done", "job_id": job.run_id, "row_range": job.row_range, "ts": datetime.now().isoformat()})
    except Exception:
        pass
    return output_path


def _write_status_json(path: str, status_obj: dict) -> None:
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status_obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _stringify_for_csv(v: object) -> object:
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return v


def _build_augmented_row_from_processed(
    processed_row: dict,
    input_header: List[str],
    profile_mapping: dict,
) -> dict:
    """
    Build an "augmented input" row:
    - preserves the original input columns (input_header)
    - adds enrichment columns from processed_row
    """
    out: dict = {}
    # Fill original columns by mapping to canonical keys where needed.
    for col in input_header:
        if col in processed_row:
            out[col] = processed_row.get(col)
            continue
        mapped = profile_mapping.get(col)
        if mapped and mapped in processed_row:
            out[col] = processed_row.get(mapped)
        else:
            out[col] = ""

    # Append/overlay enrichment columns (keep them even if they didn't exist in the original input).
    enrichment_keys = [
        "PhoneNumber_Found",
        "PhoneType_Found",
        "PhoneSources_Found",
        "PhoneExtract_Outcome",
        "PhoneExtract_FaultCategory",
        "HttpFallbackAttempted",
        "HttpFallbackUsed",
        "HttpFallbackResult",
        # Rich phone retrieval diagnostics
        "GivenPhoneNumber",
        "NormalizedGivenPhoneNumber",
        "ScrapingStatus",
        "Overall_VerificationStatus",
        "Original_Number_Status",
        "Primary_Number_1", "Primary_Type_1", "Primary_SourceURL_1",
        "Secondary_Number_1", "Secondary_Type_1", "Secondary_SourceURL_1",
        "Secondary_Number_2", "Secondary_Type_2", "Secondary_SourceURL_2",
        "RegexCandidateSnippets",
        "BestMatchedPhoneNumbers",
        "OtherRelevantNumbers",
        "ConfidenceScore",
        "LLMExtractedNumbers",
        "LLMContextPath",
        "Notes",
        "Top_Number_1", "Top_Type_1", "Top_SourceURL_1",
        "Top_Number_2", "Top_Type_2", "Top_SourceURL_2",
        "Top_Number_3", "Top_Type_3", "Top_SourceURL_3",
        "Final_Row_Outcome_Reason",
        "Determined_Fault_Category",
        "RunID",
        "TargetCountryCodes",
    ]

    # Derive the old "PhoneNumber_Found/Type/Sources/Outcome/FaultCategory" fields if missing.
    # This keeps compatibility with earlier augmented CSVs.
    top1 = (processed_row.get("Top_Number_1") or "").strip() if isinstance(processed_row.get("Top_Number_1"), str) else processed_row.get("Top_Number_1")
    if processed_row.get("PhoneNumber_Found") in (None, "", "nan", "NaN") and top1:
        out["PhoneNumber_Found"] = top1
    else:
        out["PhoneNumber_Found"] = processed_row.get("PhoneNumber_Found", out.get("PhoneNumber_Found", ""))
    out["PhoneType_Found"] = processed_row.get("PhoneType_Found", processed_row.get("Top_Type_1", ""))
    out["PhoneSources_Found"] = processed_row.get("PhoneSources_Found", processed_row.get("Top_SourceURL_1", ""))
    out["PhoneExtract_Outcome"] = processed_row.get("PhoneExtract_Outcome", processed_row.get("Final_Row_Outcome_Reason", ""))
    out["PhoneExtract_FaultCategory"] = processed_row.get("PhoneExtract_FaultCategory", processed_row.get("Determined_Fault_Category", ""))
    out["HttpFallbackAttempted"] = processed_row.get("HttpFallbackAttempted", "")
    out["HttpFallbackUsed"] = processed_row.get("HttpFallbackUsed", "")
    out["HttpFallbackResult"] = processed_row.get("HttpFallbackResult", "")

    for k in enrichment_keys:
        if k in out:
            # Already set above
            continue
        out[k] = processed_row.get(k, "")

    return out


def _writer_thread_phone_extract(
    queue_obj: object,
    results_live_csv: str,
    results_live_jsonl: str,
    results_header: List[str],
    augmented_live_csv: str,
    augmented_live_jsonl: str,
    augmented_header: List[str],
    input_header: List[str],
    profile_mapping: dict,
    augmented_delimiter: str,
    expected_jobs: int,
    status_path: str,
) -> None:
    """
    Master writer thread for phone_extract.

    Receives messages from workers:
    - job_start/job_done
    - row (finalized row payload)
    - stop

    Writes:
    - results live CSV/JSONL (comma-delimited)
    - augmented live CSV/JSONL (delimiter matches input CSV)
    - live status JSON
    """
    os.makedirs(os.path.dirname(results_live_csv), exist_ok=True)
    with (
        open(results_live_csv, "w", newline="", encoding="utf-8-sig") as f_csv,
        open(results_live_jsonl, "w", encoding="utf-8") as f_jsonl,
        open(augmented_live_csv, "w", newline="", encoding="utf-8-sig") as f_aug_csv,
        open(augmented_live_jsonl, "w", encoding="utf-8") as f_aug_jsonl,
    ):
        res_w = csv.DictWriter(f_csv, fieldnames=results_header, extrasaction="ignore")
        res_w.writeheader()
        f_csv.flush()
        aug_w = csv.DictWriter(f_aug_csv, fieldnames=augmented_header, extrasaction="ignore", delimiter=augmented_delimiter or ",")
        aug_w.writeheader()
        f_aug_csv.flush()

        started = {}
        done = {}
        rows_written = 0
        last_status_write = 0.0
        stop_received = False
        stop_received_ts: Optional[float] = None

        def _emit_status(force: bool = False) -> None:
            nonlocal last_status_write
            now = time.time()
            if (not force) and (now - last_status_write <= 2):
                return
            status_obj = {
                "ts": datetime.now().isoformat(),
                "rows_written": rows_written,
                "jobs_expected": expected_jobs,
                "jobs_started": len(started),
                "jobs_done": len(done),
                "jobs_in_flight": max(0, len(started) - len(done)),
            }
            try:
                _write_status_json(status_path, status_obj)
            except Exception:
                pass
            last_status_write = now

        while True:
            try:
                msg = queue_obj.get(timeout=1)
            except pyqueue.Empty:
                if stop_received and len(done) >= expected_jobs:
                    break
                if stop_received and stop_received_ts and (time.time() - stop_received_ts) > 30:
                    break
                _emit_status()
                continue

            mtype = (msg or {}).get("type")
            if mtype == "stop":
                stop_received = True
                stop_received_ts = time.time()
                _emit_status(force=True)
                continue
            if mtype == "job_start":
                jid = msg.get("job_id")
                started[jid] = msg
                _emit_status()
                continue
            if mtype == "job_done":
                jid = msg.get("job_id")
                done[jid] = msg
                _emit_status(force=True)
                continue
            if mtype != "row":
                continue

            data = msg.get("data", {}) or {}
            # Results CSV row (stringify complex types)
            out_row = {}
            for k in results_header:
                out_row[k] = _stringify_for_csv(data.get(k, ""))
            res_w.writerow(out_row)
            f_csv.flush()
            f_jsonl.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_jsonl.flush()

            # Augmented row (preserve original input columns + enrich)
            aug_obj = _build_augmented_row_from_processed(
                processed_row=data,
                input_header=input_header,
                profile_mapping=profile_mapping,
            )
            aug_row = {}
            for k in augmented_header:
                aug_row[k] = _stringify_for_csv(aug_obj.get(k, ""))
            aug_w.writerow(aug_row)
            f_aug_csv.flush()
            f_aug_jsonl.write(json.dumps(aug_obj, ensure_ascii=False) + "\n")
            f_aug_jsonl.flush()

            rows_written += 1
            _emit_status()

        _emit_status(force=True)


def _infer_total_rows(input_path: str) -> int:
    if input_path.lower().endswith(".csv"):
        sep = _detect_csv_delimiter(input_path)
        # Use python engine for robustness with multiline quoted fields.
        return len(pd.read_csv(input_path, sep=sep, engine="python", dtype=str, keep_default_na=False, na_filter=False))
    if input_path.lower().endswith((".xls", ".xlsx")):
        return len(pd.read_excel(input_path))
    raise ValueError(f"Unsupported input type: {input_path}")


def _build_chunks(row_range: str, total_rows: int, workers: int) -> List[str]:
    # If user provided an explicit finite range like "1-200", chunk that.
    start = 1
    end = total_rows
    if row_range:
        rr = row_range.strip()
        if "-" in rr:
            a, b = rr.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a.isdigit():
                start = int(a)
            if b.isdigit():
                end = int(b)
        elif rr.isdigit():
            start = 1
            end = int(rr)
    if start < 1:
        start = 1
    if end > total_rows:
        end = total_rows
    if end < start:
        end = start

    n = max(1, workers)
    span = end - start + 1
    chunk_size = max(1, (span + n - 1) // n)
    chunks: List[str] = []
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + chunk_size - 1)
        chunks.append(f"{cur}-{chunk_end}")
        cur = chunk_end + 1
    return chunks


def _parse_range_bounds(row_range: str, total_rows: int) -> Tuple[int, int]:
    """Return 1-based inclusive bounds for a row_range string."""
    start = 1
    end = total_rows
    if row_range:
        rr = row_range.strip()
        if "-" in rr:
            a, b = rr.split("-", 1)
            a = a.strip()
            b = b.strip()
            if a.isdigit():
                start = int(a)
            if b.isdigit():
                end = int(b)
        elif rr.isdigit():
            start = 1
            end = int(rr)
    if start < 1:
        start = 1
    if end > total_rows:
        end = total_rows
    if end < start:
        end = start
    return start, end


def _write_augmented_csv(
    input_file_path_abs: str,
    original_start_row_1based: int,
    processed_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Write an 'augmented input' CSV: original columns + appended found phone fields for the processed slice."""
    if not input_file_path_abs.lower().endswith(".csv"):
        # Keep scope simple for now; the results CSV is still produced for XLSX inputs.
        return

    # Preserve the input delimiter (many Apollo exports are semicolon-delimited).
    detected_sep = _detect_csv_delimiter(input_file_path_abs)

    original_df = pd.read_csv(input_file_path_abs, sep=detected_sep, keep_default_na=False, na_filter=False)

    # Slice original rows to match what we processed.
    start0 = max(0, original_start_row_1based - 1)
    end0 = start0 + len(processed_df)
    original_slice = original_df.iloc[start0:end0].copy().reset_index(drop=True)

    excel_prefix = os.getenv("AUGMENTED_PHONE_TEXT_PREFIX", "").strip()
    # Recommended for Excel-opened CSVs is a leading apostrophe to keep "+49..." as text:
    #   AUGMENTED_PHONE_TEXT_PREFIX="'"

    def _as_str_or_none(val: object) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
            return None
        # If something got coerced into a float-like string, undo common ".0" artifact
        if s.endswith(".0") and s.replace(".", "", 1).isdigit():
            s = s[:-2]
        return s

    def _parse_target_regions(raw: object) -> List[str]:
        # Prefer TargetCountryCodes from processed_df when available; otherwise default to DACH.
        if raw is None:
            return ["DE", "AT", "CH"]
        if isinstance(raw, list):
            return [str(x).upper() for x in raw if str(x).strip()]
        s = _as_str_or_none(raw)
        if not s:
            return ["DE", "AT", "CH"]
        # Common representation in this pipeline: "['DE', 'AT', 'CH']"
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list) and parsed:
                    return [str(x).upper() for x in parsed if str(x).strip()]
            except Exception:
                pass
        # Comma-separated fallback
        parts = [p.strip().upper() for p in s.split(",") if p.strip()]
        return parts if parts else ["DE", "AT", "CH"]

    def _normalize_found(raw_number: object, target_regions: List[str]) -> Optional[str]:
        s = _as_str_or_none(raw_number)
        if not s:
            return None
        # If it looks like an E164 country code without '+', try fixing common cases.
        digits = "".join(ch for ch in s if ch.isdigit())
        if s.startswith("00") and digits:
            # 0049... -> +49...
            s = "+" + digits[2:]
        elif not s.startswith("+") and digits and digits.startswith("49") and len(digits) >= 10:
            s = "+" + digits

        # Prefer strict E.164 normalization; try region hints for national-format numbers.
        norm = normalize_phone_number(s, region=None)
        if norm and norm != "InvalidFormat":
            return norm
        for region in target_regions:
            norm2 = normalize_phone_number(s, region=region)
            if norm2 and norm2 != "InvalidFormat":
                return norm2
        # If it doesn't normalize, keep the cleaned raw string.
        return s

    # Grab fields from processed_df (which may have been merged from workers).
    top1 = processed_df["Top_Number_1"] if "Top_Number_1" in processed_df.columns else pd.Series([None] * len(processed_df))
    top1_type = processed_df["Top_Type_1"] if "Top_Type_1" in processed_df.columns else pd.Series([None] * len(processed_df))
    top1_src = processed_df["Top_SourceURL_1"] if "Top_SourceURL_1" in processed_df.columns else pd.Series([None] * len(processed_df))
    outcome = processed_df["Final_Row_Outcome_Reason"] if "Final_Row_Outcome_Reason" in processed_df.columns else pd.Series([None] * len(processed_df))
    fault = processed_df["Determined_Fault_Category"] if "Determined_Fault_Category" in processed_df.columns else pd.Series([None] * len(processed_df))
    http_fb_attempted = processed_df["HttpFallbackAttempted"] if "HttpFallbackAttempted" in processed_df.columns else pd.Series([None] * len(processed_df))
    http_fb = processed_df["HttpFallbackUsed"] if "HttpFallbackUsed" in processed_df.columns else pd.Series([None] * len(processed_df))
    http_fb_result = processed_df["HttpFallbackResult"] if "HttpFallbackResult" in processed_df.columns else pd.Series([None] * len(processed_df))
    tgt = processed_df["TargetCountryCodes"] if "TargetCountryCodes" in processed_df.columns else pd.Series([None] * len(processed_df))

    normalized_numbers: List[Optional[str]] = []
    for raw_num, raw_regions in zip(top1.tolist(), tgt.tolist()):
        regions = _parse_target_regions(raw_regions)
        normalized_numbers.append(_normalize_found(raw_num, regions))

    if excel_prefix:
        normalized_numbers = [f"{excel_prefix}{n}" if n else n for n in normalized_numbers]

    original_slice["PhoneNumber_Found"] = normalized_numbers
    original_slice["PhoneType_Found"] = [(_as_str_or_none(v) or "") for v in top1_type.tolist()]
    original_slice["PhoneSources_Found"] = [(_as_str_or_none(v) or "") for v in top1_src.tolist()]
    original_slice["PhoneExtract_Outcome"] = [(_as_str_or_none(v) or "") for v in outcome.tolist()]
    original_slice["PhoneExtract_FaultCategory"] = [(_as_str_or_none(v) or "") for v in fault.tolist()]
    original_slice["HttpFallbackAttempted"] = [(_as_str_or_none(v) or "") for v in http_fb_attempted.tolist()]
    original_slice["HttpFallbackUsed"] = [(_as_str_or_none(v) or "") for v in http_fb.tolist()]
    original_slice["HttpFallbackResult"] = [(_as_str_or_none(v) or "") for v in http_fb_result.tolist()]

    # Also persist the richer phone-retrieval diagnostics + selected numbers so downstream
    # full-pipeline / resume runs can reuse them without rerunning extraction.
    passthrough_cols = [
        "GivenPhoneNumber",
        "NormalizedGivenPhoneNumber",
        "ScrapingStatus",
        "Overall_VerificationStatus",
        "Original_Number_Status",
        "Primary_Number_1", "Primary_Type_1", "Primary_SourceURL_1",
        "Secondary_Number_1", "Secondary_Type_1", "Secondary_SourceURL_1",
        "Secondary_Number_2", "Secondary_Type_2", "Secondary_SourceURL_2",
        "RegexCandidateSnippets",
        "BestMatchedPhoneNumbers",
        "OtherRelevantNumbers",
        "ConfidenceScore",
        "LLMExtractedNumbers",
        "LLMContextPath",
        "Notes",
        "Top_Number_1", "Top_Type_1", "Top_SourceURL_1",
        "Top_Number_2", "Top_Type_2", "Top_SourceURL_2",
        "Top_Number_3", "Top_Type_3", "Top_SourceURL_3",
        "Final_Row_Outcome_Reason",
        "Determined_Fault_Category",
        "RunID",
        "TargetCountryCodes",
    ]
    for c in passthrough_cols:
        if c in processed_df.columns and c not in original_slice.columns:
            original_slice[c] = processed_df[c].tolist()

    original_slice.to_csv(output_path, index=False, encoding="utf-8", sep=detected_sep)
    # JSONL companion for easier downstream parsing.
    try:
        jsonl_path = os.path.splitext(output_path)[0] + ".jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, r in original_slice.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    except Exception:
        pass


def main() -> None:
    args = _parse_args()
    pipeline_start_time = time.time()

    app_config = AppConfig(
        input_file_override=args.input_file,
        row_range_override=args.range or None,
        run_id_suffix_override=args.suffix,
        test_mode=False,
    )
    if getattr(args, "input_profile", None):
        app_config.input_file_profile_name = args.input_profile

    master_run_id = generate_run_id()
    if app_config.run_id_suffix:
        master_run_id = f"{master_run_id}_{app_config.run_id_suffix}"

    # Master run lives at: <OUTPUT_BASE_DIR>/<master_run_id>/
    run_output_dir, _ = _setup_run_dirs(app_config, master_run_id)
    log_file_path = os.path.join(run_output_dir, f"phone_extract_{master_run_id}.log")
    file_log_level_int = getattr(logging, app_config.log_level.upper(), logging.INFO)
    console_log_level_int = getattr(logging, app_config.console_log_level.upper(), logging.WARNING)
    setup_logging(
        file_log_level=file_log_level_int,
        console_log_level=console_log_level_int,
        log_file_path=log_file_path,
    )

    input_file_path_abs = resolve_path(app_config.input_excel_file_path, BASE_FILE_PATH_FOR_RESOLVE)
    if not os.path.exists(input_file_path_abs):
        raise FileNotFoundError(f"Input file not found: {input_file_path_abs}")

    workers = max(1, int(args.workers or 1))

    # Determine input header + profile mapping for consistent augmented output.
    input_header: List[str] = []
    if input_file_path_abs.lower().endswith(".csv"):
        detected_sep = _detect_csv_delimiter(input_file_path_abs)
        try:
            input_header = list(pd.read_csv(input_file_path_abs, sep=detected_sep, nrows=0).columns)
        except Exception:
            input_header = []
    elif input_file_path_abs.lower().endswith((".xls", ".xlsx")):
        try:
            input_header = list(pd.read_excel(input_file_path_abs, nrows=0).columns)
        except Exception:
            input_header = []

    profile_mapping = (AppConfig.INPUT_COLUMN_PROFILES.get(app_config.input_file_profile_name) or {}).copy()
    # Remove meta keys
    profile_mapping = {k: v for k, v in profile_mapping.items() if not str(k).startswith("_")}

    # Results header: original header with profile renames applied (+ extra required cols).
    results_header = [profile_mapping.get(c, c) for c in input_header] if input_header else []
    required_cols = [
        "__source_row_1based",
        "CompanyName",
        "GivenURL",
        "GivenPhoneNumber",
        "NormalizedGivenPhoneNumber",
        "ScrapingStatus",
        "Overall_VerificationStatus",
        "Original_Number_Status",
        "Primary_Number_1", "Primary_Type_1", "Primary_SourceURL_1",
        "Secondary_Number_1", "Secondary_Type_1", "Secondary_SourceURL_1",
        "Secondary_Number_2", "Secondary_Type_2", "Secondary_SourceURL_2",
        "RegexCandidateSnippets",
        "BestMatchedPhoneNumbers",
        "OtherRelevantNumbers",
        "ConfidenceScore",
        "LLMExtractedNumbers",
        "LLMContextPath",
        "Notes",
        "Top_Number_1", "Top_Type_1", "Top_SourceURL_1",
        "Top_Number_2", "Top_Type_2", "Top_SourceURL_2",
        "Top_Number_3", "Top_Type_3", "Top_SourceURL_3",
        "Final_Row_Outcome_Reason",
        "Determined_Fault_Category",
        "RunID",
        "TargetCountryCodes",
        "HttpFallbackAttempted",
        "HttpFallbackUsed",
        "HttpFallbackResult",
    ]
    seen = set()
    results_header = [c for c in results_header if not (c in seen or seen.add(c))]
    for c in required_cols:
        if c not in seen:
            results_header.append(c)
            seen.add(c)

    # Augmented header: original input columns + appended enrichment columns
    augmented_header = list(input_header) if input_header else []
    for c in [
        "__source_row_1based",
        "PhoneNumber_Found",
        "PhoneType_Found",
        "PhoneSources_Found",
        "PhoneExtract_Outcome",
        "PhoneExtract_FaultCategory",
        "HttpFallbackAttempted",
        "HttpFallbackUsed",
        "HttpFallbackResult",
        # Rich diagnostics (also include for convenience in augmented)
        "GivenPhoneNumber",
        "NormalizedGivenPhoneNumber",
        "ScrapingStatus",
        "Overall_VerificationStatus",
        "Original_Number_Status",
        "Primary_Number_1", "Primary_Type_1", "Primary_SourceURL_1",
        "Secondary_Number_1", "Secondary_Type_1", "Secondary_SourceURL_1",
        "Secondary_Number_2", "Secondary_Type_2", "Secondary_SourceURL_2",
        "RegexCandidateSnippets",
        "BestMatchedPhoneNumbers",
        "OtherRelevantNumbers",
        "ConfidenceScore",
        "LLMExtractedNumbers",
        "LLMContextPath",
        "Notes",
        "Top_Number_1", "Top_Type_1", "Top_SourceURL_1",
        "Top_Number_2", "Top_Type_2", "Top_SourceURL_2",
        "Top_Number_3", "Top_Type_3", "Top_SourceURL_3",
        "Final_Row_Outcome_Reason",
        "Determined_Fault_Category",
        "RunID",
        "TargetCountryCodes",
    ]:
        if c not in augmented_header:
            augmented_header.append(c)

    # Live output files (always)
    live_results_csv = os.path.join(run_output_dir, f"phone_extraction_results_{master_run_id}_live.csv")
    live_results_jsonl = os.path.join(run_output_dir, f"phone_extraction_results_{master_run_id}_live.jsonl")
    live_aug_csv = os.path.join(run_output_dir, f"input_augmented_{master_run_id}_live.csv")
    live_aug_jsonl = os.path.join(run_output_dir, f"input_augmented_{master_run_id}_live.jsonl")
    live_status_path = os.path.join(run_output_dir, f"phone_extract_{master_run_id}_live_status.json")

    augmented_delim = detected_sep if input_file_path_abs.lower().endswith(".csv") else ","

    if workers == 1:
        q = pyqueue.Queue()
        wt = threading.Thread(
            target=_writer_thread_phone_extract,
            kwargs=dict(
                queue_obj=q,
                results_live_csv=live_results_csv,
                results_live_jsonl=live_results_jsonl,
                results_header=results_header,
                augmented_live_csv=live_aug_csv,
                augmented_live_jsonl=live_aug_jsonl,
                augmented_header=augmented_header,
                input_header=input_header,
                profile_mapping=profile_mapping,
                augmented_delimiter=augmented_delim,
                expected_jobs=1,
                status_path=live_status_path,
            ),
            daemon=True,
        )
        wt.start()

        # Run single worker but stream per-row to the writer thread.
        job = WorkerJob(
            input_file=input_file_path_abs,
            input_profile=app_config.input_file_profile_name,
            row_range=args.range or "",
            run_id=master_run_id,
            suffix=args.suffix,
            log_level=app_config.log_level,
            console_log_level=app_config.console_log_level,
            output_base_dir=app_config.output_base_dir,
        )
        try:
            q.put({"type": "job_start", "job_id": job.run_id, "ts": datetime.now().isoformat()})
        except Exception:
            pass
        # Run the worker inline but pass queue into execute_pipeline_flow by temporarily calling _run_worker and then replaying.
        # For simplicity, re-run the pipeline here with streaming enabled.
        pipeline_start_time = time.time()
        app_config_single = AppConfig(
            input_file_override=job.input_file,
            row_range_override=job.row_range or None,
            run_id_suffix_override=job.suffix,
            test_mode=False,
        )
        app_config_single.input_file_profile_name = job.input_profile
        app_config_single.log_level = job.log_level
        app_config_single.console_log_level = job.console_log_level
        app_config_single.output_base_dir = job.output_base_dir
        run_metrics = initialize_run_metrics(job.run_id)
        run_output_dir_single, llm_context_dir_single = _setup_run_dirs(app_config_single, job.run_id)
        failure_log_csv_path = os.path.join(run_output_dir_single, f"failed_rows_{job.run_id}.csv")
        llm_extractor = GeminiLLMExtractor(config=app_config_single)
        df, original_phone_col_name, _ = load_and_preprocess_data(input_file_path_abs, app_config_instance=app_config_single)
        if df is None:
            raise RuntimeError(f"[{job.run_id}] Failed to load input; DataFrame is None.")
        run_metrics["data_processing_stats"]["input_rows_count"] = len(df)
        df = initialize_dataframe_columns(df)
        with open(failure_log_csv_path, "w", newline="", encoding="utf-8") as fh_fail:
            failure_writer = csv.writer(fh_fail)
            _write_failure_header(failure_writer)
            df_processed, attrition_data_list, canonical_domain_journey_data, *_ = execute_pipeline_flow(
                df=df,
                app_config=app_config_single,
                llm_extractor=llm_extractor,
                run_output_dir=run_output_dir_single,
                llm_context_dir=llm_context_dir_single,
                run_id=job.run_id,
                failure_writer=failure_writer,
                run_metrics=run_metrics,
                original_phone_col_name_for_profile=original_phone_col_name,
                row_queue=q,
                row_queue_meta={"job_id": job.run_id},
            )
        output_path = os.path.join(run_output_dir_single, f"phone_extraction_results_{job.run_id}.csv")
        df_processed.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Wrote phone extraction results CSV: {output_path}")
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(
            metrics=run_metrics,
            output_dir=run_output_dir_single,
            run_id=job.run_id,
            pipeline_start_time=pipeline_start_time,
            attrition_data_list_for_metrics=attrition_data_list,
            canonical_domain_journey_data=canonical_domain_journey_data,
            logger=logging.getLogger(__name__),
        )
        try:
            q.put({"type": "job_done", "job_id": job.run_id, "ts": datetime.now().isoformat()})
            q.put({"type": "stop"})
        except Exception:
            pass
        wt.join(timeout=60)

        # Copy live outputs to stable final copies
        try:
            import shutil
            shutil.copyfile(live_results_csv, os.path.join(run_output_dir, f"phone_extraction_results_{master_run_id}.csv"))
            shutil.copyfile(live_results_jsonl, os.path.join(run_output_dir, f"phone_extraction_results_{master_run_id}.jsonl"))
            shutil.copyfile(live_aug_csv, os.path.join(run_output_dir, f"input_augmented_{master_run_id}.csv"))
            shutil.copyfile(live_aug_jsonl, os.path.join(run_output_dir, f"input_augmented_{master_run_id}.jsonl"))
        except Exception:
            pass

        # Also write an "augmented input" CSV for CSV inputs.
        total_rows = _infer_total_rows(input_file_path_abs)
        start_1based, _ = _parse_range_bounds(args.range or "", total_rows=total_rows)
        augmented_path = os.path.join(run_output_dir, f"input_augmented_{master_run_id}.csv")
        processed_df = pd.read_csv(output_path, dtype=str, keep_default_na=False, na_filter=False)
        _write_augmented_csv(input_file_path_abs, start_1based, processed_df, augmented_path)
        if os.path.exists(augmented_path):
            logger.info(f"Wrote augmented input CSV: {augmented_path}")
        return

    total_rows = _infer_total_rows(input_file_path_abs)
    chunks = _build_chunks(args.range or "", total_rows=total_rows, workers=workers)
    logger.info(f"Parallel mode enabled: workers={workers}, total_rows={total_rows}, chunks={chunks}")

    # Worker runs are contained within the master run directory to avoid spilling into OUTPUT_BASE_DIR.
    # Layout:
    #   <OUTPUT_BASE_DIR>/<master_run_id>/workers/<worker_run_id>/
    worker_base_dir = os.path.join(run_output_dir, "workers")
    os.makedirs(worker_base_dir, exist_ok=True)

    jobs: List[WorkerJob] = []
    for i, rr in enumerate(chunks):
        worker_run_id = f"w{i+1}of{len(chunks)}"
        jobs.append(
            WorkerJob(
                input_file=input_file_path_abs,
                input_profile=app_config.input_file_profile_name,
                row_range=rr,
                run_id=worker_run_id,
                suffix=args.suffix,
                log_level=app_config.log_level,
                console_log_level=app_config.console_log_level,
                output_base_dir=worker_base_dir,
            )
        )

    # Live streaming writer (master)
    mgr = mp.Manager()
    q = mgr.Queue()
    wt = threading.Thread(
        target=_writer_thread_phone_extract,
        kwargs=dict(
            queue_obj=q,
            results_live_csv=live_results_csv,
            results_live_jsonl=live_results_jsonl,
            results_header=results_header,
            augmented_live_csv=live_aug_csv,
            augmented_live_jsonl=live_aug_jsonl,
            augmented_header=augmented_header,
            input_header=input_header,
            profile_mapping=profile_mapping,
            augmented_delimiter=augmented_delim,
            expected_jobs=len(jobs),
            status_path=live_status_path,
        ),
        daemon=True,
    )
    wt.start()

    worker_outputs: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for job in jobs:
            logger.info(f"Worker start: {job.run_id} range={job.row_range}")
            futures[ex.submit(_run_worker_streaming, job, q)] = job
        for fut in as_completed(futures):
            job = futures[fut]
            out_path = fut.result()
            logger.info(f"Worker complete: {job.run_id} range={job.row_range} -> {out_path}")
            worker_outputs.append(out_path)

    try:
        q.put({"type": "stop"})
    except Exception:
        pass
    wt.join(timeout=120)

    # Copy live outputs to stable final copies
    try:
        import shutil
        shutil.copyfile(live_results_csv, os.path.join(run_output_dir, f"phone_extraction_results_{master_run_id}.csv"))
        shutil.copyfile(live_results_jsonl, os.path.join(run_output_dir, f"phone_extraction_results_{master_run_id}.jsonl"))
        shutil.copyfile(live_aug_csv, os.path.join(run_output_dir, f"input_augmented_{master_run_id}.csv"))
        shutil.copyfile(live_aug_jsonl, os.path.join(run_output_dir, f"input_augmented_{master_run_id}.jsonl"))
    except Exception:
        pass

    # Merge outputs into a single CSV under the master run dir
    # IMPORTANT: preserve the original row order. Worker outputs must be concatenated in chunk order
    # (by the start row of each chunk), not lexicographic path order (e.g. w10 < w2).
    worker_outputs_with_start: List[Tuple[int, str]] = []
    for p in worker_outputs:
        # We can recover the associated WorkerJob by matching on the output path; to avoid brittle matching,
        # derive the worker id from the path and map it back to the chunk start.
        worker_outputs_with_start.append((10**9, p))
    # Build a map run_id -> start row for stable ordering
    run_id_to_start: dict = {}
    for job in jobs:
        try:
            start_1based, _ = _parse_range_bounds(job.row_range or "", total_rows=total_rows)
            run_id_to_start[job.run_id] = start_1based
        except Exception:
            run_id_to_start[job.run_id] = 10**9

    def _extract_worker_run_id(path_str: str) -> str:
        # Expected: .../workers/<run_id>/phone_extraction_results_<run_id>.csv
        base = os.path.basename(path_str)
        # Prefer parsing from filename first:
        if base.startswith("phone_extraction_results_") and base.endswith(".csv"):
            return base[len("phone_extraction_results_"):-len(".csv")]
        # Fallback: parent directory name
        return os.path.basename(os.path.dirname(path_str))

    worker_outputs_with_start = []
    for p in worker_outputs:
        rid = _extract_worker_run_id(p)
        worker_outputs_with_start.append((int(run_id_to_start.get(rid, 10**9)), p))

    worker_paths_sorted = [p for _, p in sorted(worker_outputs_with_start, key=lambda t: t[0])]
    merged_df = pd.concat(
        [pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False) for p in worker_paths_sorted],
        ignore_index=True
    )
    merged_output_path = args.output_file.strip() if args.output_file else os.path.join(
        run_output_dir, f"phone_extraction_results_{master_run_id}_merged.csv"
    )
    merged_df.to_csv(merged_output_path, index=False, encoding="utf-8")
    # JSONL companion for merged results.
    try:
        jsonl_path = os.path.splitext(merged_output_path)[0] + ".jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, r in merged_df.iterrows():
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    except Exception:
        pass
    logger.info(f"Merged output written: {merged_output_path}")

    # Also write augmented CSV for the processed slice (CSV inputs only).
    start_1based, _ = _parse_range_bounds(args.range or "", total_rows=total_rows)
    augmented_path = os.path.join(run_output_dir, f"input_augmented_{master_run_id}.csv")
    _write_augmented_csv(input_file_path_abs, start_1based, merged_df, augmented_path)
    if os.path.exists(augmented_path):
        logger.info(f"Wrote augmented input CSV: {augmented_path}")
    logger.info(f"Total duration: {time.time() - pipeline_start_time:.2f}s")


if __name__ == "__main__":
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO)
    main()

