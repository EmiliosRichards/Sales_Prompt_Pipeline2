"""
Parallel full-pipeline orchestration.

Design:
- A master process launches many small "jobs" (row ranges) into a ProcessPoolExecutor.
- Worker processes run the *full* pipeline flow for their slice (scrape + LLM + phone + pitch).
- Workers stream per-row results back to the master via a multiprocessing.Manager().Queue().
- The master appends rows as they arrive into:
  - SalesOutreachReport_<run_id>_live.csv
  - SalesOutreachReport_<run_id>_live.jsonl

This keeps progress visibility strong and avoids concurrent file writes from multiple processes.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue as pyqueue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.config import AppConfig
from src.core.logging_config import setup_logging
from src.data_handling.loader import load_and_preprocess_data
from src.llm_clients.gemini_client import GeminiClient
from src.processing.pipeline_flow import execute_pipeline_flow
from src.utils.helpers import (
    initialize_dataframe_columns,
    initialize_run_metrics,
    precompute_input_duplicate_stats,
)

logger = logging.getLogger(__name__)

_PROCESS_LOGGING_CONFIGURED = False


@dataclass(frozen=True)
class ParallelJob:
    job_id: str
    row_range: str


def _detect_csv_delimiter(file_path: str) -> str:
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


def _infer_total_rows_csv(input_path: str) -> int:
    sep = _detect_csv_delimiter(input_path)
    # Use python engine to safely handle quoted multiline descriptions.
    df = pd.read_csv(input_path, sep=sep, engine="python", usecols=[0], dtype=str, keep_default_na=False, na_filter=False)
    return len(df)

def _infer_total_rows_excel(input_path: str) -> int:
    # Robust but potentially slower; acceptable for orchestration.
    df = pd.read_excel(input_path, usecols=[0])
    return len(df)


def _parse_range_bounds(row_range: str, total_rows: int) -> Tuple[int, int]:
    """Return 1-based inclusive bounds."""
    start = 1
    end = total_rows
    rr = (row_range or "").strip()
    if rr:
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
    start = max(1, start)
    end = min(total_rows, end)
    if end < start:
        end = start
    return start, end


def _build_jobs(row_range: str, total_rows: int, workers: int, chunk_multiplier: int = 4) -> List[ParallelJob]:
    """Create many smaller jobs for better load balancing."""
    start, end = _parse_range_bounds(row_range, total_rows=total_rows)
    span = end - start + 1
    workers = max(1, int(workers))
    chunks_target = max(workers, workers * max(1, int(chunk_multiplier)))
    chunk_size = max(1, (span + chunks_target - 1) // chunks_target)

    jobs: List[ParallelJob] = []
    cur = start
    i = 1
    while cur <= end:
        chunk_end = min(end, cur + chunk_size - 1)
        jobs.append(ParallelJob(job_id=f"job{i:04d}", row_range=f"{cur}-{chunk_end}"))
        cur = chunk_end + 1
        i += 1
    return jobs


def _write_status(path: str, status_obj: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status_obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _writer_thread(
    queue: Any,
    live_csv_path: str,
    live_jsonl_path: str,
    header: List[str],
    augmented_csv_path: str,
    augmented_jsonl_path: str,
    augmented_header: List[str],
    expected_jobs: int,
    status_path: str,
) -> None:
    os.makedirs(os.path.dirname(live_csv_path), exist_ok=True)
    with (
        open(live_csv_path, "w", newline="", encoding="utf-8-sig") as f_csv,
        open(live_jsonl_path, "w", encoding="utf-8") as f_jsonl,
        open(augmented_csv_path, "w", newline="", encoding="utf-8-sig") as f_aug_csv,
        open(augmented_jsonl_path, "w", encoding="utf-8") as f_aug_jsonl,
    ):
        csv_writer = csv.DictWriter(f_csv, fieldnames=header, extrasaction="ignore")
        csv_writer.writeheader()
        f_csv.flush()
        aug_writer = csv.DictWriter(f_aug_csv, fieldnames=augmented_header, extrasaction="ignore")
        aug_writer.writeheader()
        f_aug_csv.flush()

        started: Dict[str, Any] = {}
        done: Dict[str, Any] = {}
        rows_written = 0

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
                _write_status(status_path, status_obj)
            except Exception:
                pass
            last_status_write = now

        last_status_write = 0.0
        stop_received = False
        stop_received_ts: Optional[float] = None
        while True:
            try:
                msg = queue.get(timeout=1)
            except pyqueue.Empty:
                # If we've been told to stop, only exit once all jobs are done and the queue is drained.
                if stop_received and len(done) >= expected_jobs:
                    break
                # Also, after a grace period, allow exit even if some job_done messages never arrive.
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
            elif mtype == "job_done":
                jid = msg.get("job_id")
                done[jid] = msg
                _emit_status(force=True)
            elif mtype == "row":
                data = msg.get("data", {}) or {}
                # Normalize non-scalar values for CSV.
                row_out: Dict[str, Any] = {}
                for k in header:
                    v = data.get(k, "")
                    if isinstance(v, (dict, list)):
                        row_out[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row_out[k] = v
                csv_writer.writerow(row_out)
                f_csv.flush()
                f_jsonl.write(json.dumps(data, ensure_ascii=False) + "\n")
                f_jsonl.flush()

                aug_row: Dict[str, Any] = {}
                for k in augmented_header:
                    v = data.get(k, "")
                    if isinstance(v, (dict, list)):
                        aug_row[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        aug_row[k] = v
                aug_writer.writerow(aug_row)
                f_aug_csv.flush()
                f_aug_jsonl.write(json.dumps(data, ensure_ascii=False) + "\n")
                f_aug_jsonl.flush()
                rows_written += 1
                _emit_status()

        # Final status snapshot (always)
        _emit_status(force=True)


def _worker_run_job(
    input_file_path_abs: str,
    input_profile: str,
    job: ParallelJob,
    run_id: str,
    run_output_dir: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    golden_partner_summaries: List[Dict[str, Any]],
    queue: Any,
    skip_prequalification: bool,
    pitch_from_description: bool,
    enable_phone_retrieval_in_full_pipeline: bool,
    force_phone_extraction: bool,
    reuse_scraped_content_if_available: bool,
    scraped_content_cache_dirs: List[str],
) -> Dict[str, Any]:
    global _PROCESS_LOGGING_CONFIGURED
    # Worker-local AppConfig / GeminiClient
    app_config = AppConfig(
        input_file_override=input_file_path_abs,
        row_range_override=job.row_range,
        run_id_suffix_override=None,
        test_mode=False,
    )
    app_config.input_file_profile_name = input_profile
    app_config.enable_phone_retrieval_in_full_pipeline = enable_phone_retrieval_in_full_pipeline
    app_config.force_phone_extraction = force_phone_extraction
    try:
        app_config.reuse_scraped_content_if_available = bool(reuse_scraped_content_if_available)
        app_config.scraped_content_cache_dirs = list(scraped_content_cache_dirs or [])
    except Exception:
        pass

    # Each job writes artifacts into its own folder under run_output_dir/jobs/<job_id>/
    job_out = os.path.join(run_output_dir, "jobs", job.job_id)
    job_ctx = os.path.join(job_out, "llm_context")
    job_req = os.path.join(job_out, "llm_requests")
    os.makedirs(job_ctx, exist_ok=True)
    os.makedirs(job_req, exist_ok=True)

    # ProcessPoolExecutor reuses processes across jobs; configure logging once per process.
    if not _PROCESS_LOGGING_CONFIGURED:
        pid = os.getpid()
        log_dir = os.path.join(run_output_dir, "jobs", "_worker_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"worker_pid{pid}.log")
        setup_logging(
            file_log_level=getattr(logging, app_config.log_level.upper(), logging.INFO),
            console_log_level=getattr(logging, app_config.console_log_level.upper(), logging.WARNING),
            log_file_path=log_path,
        )
        _PROCESS_LOGGING_CONFIGURED = True

    queue.put({"type": "job_start", "job_id": job.job_id, "row_range": job.row_range, "ts": datetime.now().isoformat()})

    gemini_client = GeminiClient(config=app_config)

    df = load_and_preprocess_data(input_file_path_abs, app_config_instance=app_config)
    if df is None:
        queue.put({"type": "job_done", "job_id": job.job_id, "row_range": job.row_range, "ok": False, "reason": "df_none"})
        return {"job_id": job.job_id, "ok": False}

    df = initialize_dataframe_columns(df)
    run_metrics = initialize_run_metrics(run_id)
    run_metrics["data_processing_stats"]["input_rows_count"] = len(df)
    # Duplicate stats are not very meaningful per slice, but harmless.
    df = precompute_input_duplicate_stats(df, app_config, run_metrics)

    # Failure log for this job
    failure_csv_path = os.path.join(job_out, f"failed_rows_{job.job_id}.csv")
    import csv as _csv
    with open(failure_csv_path, "w", newline="", encoding="utf-8") as fh:
        fw = _csv.writer(fh)
        fw.writerow(["log_timestamp", "input_row_identifier", "CompanyName", "GivenURL", "stage_of_failure", "error_reason", "error_details", "Associated_Pathful_Canonical_URL"])

        # Compute global row base for monitoring (cheap: parse from job.row_range)
        start_1based, _ = _parse_range_bounds(job.row_range, total_rows=10**9)

        execute_pipeline_flow(
            df=df,
            app_config=app_config,
            gemini_client=gemini_client,
            run_output_dir=job_out,
            llm_context_dir=job_ctx,
            llm_requests_dir=job_req,
            run_id=run_id,
            failure_writer=fw,
            run_metrics=run_metrics,
            golden_partner_summaries=golden_partner_summaries,
            skip_prequalification=skip_prequalification,
            pitch_from_description=pitch_from_description,
            live_reporter=None,
            row_queue=queue,
            row_queue_meta={"job_id": job.job_id, "row_range": job.row_range, "global_row_start_1based": start_1based},
        )

    queue.put({"type": "job_done", "job_id": job.job_id, "row_range": job.row_range, "ok": True, "ts": datetime.now().isoformat()})
    return {"job_id": job.job_id, "ok": True}


def run_parallel_full_pipeline(
    *,
    input_file_path_abs: str,
    input_profile: str,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    golden_partner_summaries: List[Dict[str, Any]],
    workers: int,
    row_range: str,
    skip_prequalification: bool,
    pitch_from_description: bool,
) -> None:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if input_file_path_abs.lower().endswith(".csv"):
        total_rows = _infer_total_rows_csv(input_file_path_abs)
    else:
        total_rows = _infer_total_rows_excel(input_file_path_abs)
    jobs = _build_jobs(row_range=row_range, total_rows=total_rows, workers=workers, chunk_multiplier=4)

    live_csv_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_live.csv")
    live_jsonl_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_live.jsonl")
    status_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_live_status.json")
    augmented_csv_path = os.path.join(run_output_dir, f"input_augmented_{run_id}_live.csv")
    augmented_jsonl_path = os.path.join(run_output_dir, f"input_augmented_{run_id}_live.jsonl")

    # Header must match what workers emit: use the canonicalized DataFrame columns
    # (after profile-based renaming), just like the single-process pipeline does.
    header_probe_cfg = AppConfig(
        input_file_override=input_file_path_abs,
        row_range_override="1",  # read 1 row to keep it fast
        run_id_suffix_override=None,
        test_mode=False,
    )
    header_probe_cfg.input_file_profile_name = input_profile
    df_head = load_and_preprocess_data(input_file_path_abs, app_config_instance=header_probe_cfg)
    base_cols = list(df_head.columns) if df_head is not None else []

    extra_cols = [
        'CanonicalEntryURL', 'found_number', 'PhoneNumber_Status', 'is_b2b', 'serves_1000',
        'is_b2b_reason', 'serves_1000_reason', 'description', 'Industry',
        'Products/Services Offered', 'USP/Key Selling Points', 'Customer Target Segments',
        'Business Model', 'Company Size Inferred', 'Innovation Level Indicators',
        'Website Clarity Notes', 'B2B Indicator', 'Phone Outreach Suitability',
        'Target Group Size Assessment', 'sales_pitch', 'matched_golden_partner',
        'match_reasoning', 'Matched Partner Description', 'Avg Leads Per Day', 'Rank',
        '__meta_job_id', '__meta_row_range', '__meta_global_row_start_1based',
    ]

    # Deduplicate while preserving order
    header: List[str] = []
    seen = set()
    for col in list(base_cols) + list(extra_cols):
        if col in seen:
            continue
        seen.add(col)
        header.append(col)

    augmented_extra_cols = [
        'CanonicalEntryURL', 'found_number', 'PhoneNumber_Status',
        'description', 'Industry',
        'Products/Services Offered', 'USP/Key Selling Points', 'Customer Target Segments',
        'Business Model', 'Company Size Inferred', 'Innovation Level Indicators',
        'Website Clarity Notes', 'B2B Indicator', 'Phone Outreach Suitability',
        'Target Group Size Assessment',
        'matched_golden_partner', 'match_reasoning',
        'sales_pitch',
    ]
    augmented_header: List[str] = []
    seen2 = set()
    for col in list(base_cols) + list(augmented_extra_cols):
        if col in seen2:
            continue
        seen2.add(col)
        augmented_header.append(col)

    mp_ctx = mp.get_context("spawn")
    manager = mp_ctx.Manager()
    queue = manager.Queue()

    writer = threading.Thread(
        target=_writer_thread,
        args=(queue, live_csv_path, live_jsonl_path, header, augmented_csv_path, augmented_jsonl_path, augmented_header, len(jobs), status_path),
        daemon=True,
    )
    writer.start()

    # Dispatch jobs
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = []
        for job in jobs:
            futures.append(
                ex.submit(
                    _worker_run_job,
                    input_file_path_abs,
                    input_profile,
                    job,
                    run_id,
                    run_output_dir,
                    llm_context_dir,
                    llm_requests_dir,
                    golden_partner_summaries,
                    queue,
                    skip_prequalification,
                    pitch_from_description,
                    # propagate phone flags
                    bool(getattr(app_config, "enable_phone_retrieval_in_full_pipeline", True)),
                    bool(getattr(app_config, "force_phone_extraction", False)),
                    # propagate scrape reuse flags
                    bool(getattr(app_config, "reuse_scraped_content_if_available", False)),
                    list(getattr(app_config, "scraped_content_cache_dirs", []) or []),
                )
            )
        for fut in as_completed(futures):
            _ = fut.result()

    # Stop writer and produce final outputs (CSV + XLSX)
    queue.put({"type": "stop"})
    writer.join(timeout=600)

    final_csv = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}.csv")
    final_xlsx = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}.xlsx")
    final_aug_csv = os.path.join(run_output_dir, f"input_augmented_{run_id}.csv")
    final_jsonl = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}.jsonl")
    final_aug_jsonl = os.path.join(run_output_dir, f"input_augmented_{run_id}.jsonl")
    try:
        df_live = pd.read_csv(live_csv_path, dtype=str, keep_default_na=False, na_filter=False)
        df_live.to_csv(final_csv, index=False, encoding="utf-8-sig")
        df_live.to_excel(final_xlsx, index=False)
        logger.info(f"Wrote final combined report: {final_csv} and {final_xlsx}")
    except Exception as e:
        logger.error(f"Failed to write final combined report: {e}", exc_info=True)

    # Final JSONL copies (end-to-end rows)
    try:
        import shutil
        if os.path.exists(live_jsonl_path):
            shutil.copyfile(live_jsonl_path, final_jsonl)
            logger.info(f"Wrote final JSONL report: {final_jsonl}")
    except Exception as e:
        logger.error(f"Failed to write final JSONL report: {e}", exc_info=True)

    try:
        df_aug = pd.read_csv(augmented_csv_path, dtype=str, keep_default_na=False, na_filter=False)
        df_aug.to_csv(final_aug_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Wrote final augmented input: {final_aug_csv}")
    except Exception as e:
        logger.error(f"Failed to write final augmented input: {e}", exc_info=True)

    try:
        import shutil
        if os.path.exists(augmented_jsonl_path):
            shutil.copyfile(augmented_jsonl_path, final_aug_jsonl)
            logger.info(f"Wrote final augmented JSONL: {final_aug_jsonl}")
    except Exception as e:
        logger.error(f"Failed to write final augmented JSONL: {e}", exc_info=True)

