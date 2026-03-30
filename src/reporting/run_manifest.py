"""
Run manifest + run history indexing.

Goal: make pipeline runs auditable and easy to reproduce by persisting:
- CLI args / run command
- effective config snapshot (sanitized)
- input file metadata
- output file inventory + optional row counts
- runtime stats + git versioning (commit/branch/dirty)

Artifacts:
- Per-run:   output_data/<run_id>/run_manifest.json
- Global:    output_data/_run_history.jsonl   (one JSON object per run; append-only)

This module is intentionally best-effort: it should never crash a run.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


MANIFEST_FILENAME = "run_manifest.json"
GLOBAL_HISTORY_FILENAME = "_run_history.jsonl"
MANIFEST_SCHEMA_VERSION = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mkdir(path: Union[str, Path]) -> None:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _safe_json_write(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    """
    Atomic JSON write (best-effort).
    """
    try:
        p = Path(path)
        _safe_mkdir(p.parent)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        # never fail a run due to manifest issues
        pass


def _safe_json_read(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _redact_value(key: str, value: Any) -> Any:
    k = (key or "").lower()
    if any(tok in k for tok in ["api_key", "apikey", "token", "secret", "password", "passwd", "key"]):
        # Keep the fact that it existed but never persist the value.
        if value in (None, "", False):
            return value
        return "***REDACTED***"
    return value


def _snapshot_config(app_config: Any) -> Dict[str, Any]:
    """
    Snapshot a subset of AppConfig that is helpful for reproducing runs.
    We keep this intentionally curated to reduce noise and avoid secrets.
    """
    allow: List[str] = [
        # input/output
        "input_excel_file_path",
        "input_file_profile_name",
        "skip_rows_config",
        "nrows_config",
        "consecutive_empty_rows_to_stop",
        "output_base_dir",
        # logging
        "log_level",
        "console_log_level",
        # LLM
        "full_llm_provider",
        "llm_model_name",
        "llm_model_name_sales_insights",
               "phone_llm_provider",
               "openai_model_name",
        "openai_model_name_sales_insights",
               "openai_service_tier",
               "openai_timeout_seconds",
               "openai_flex_max_retries",
               "openai_flex_fallback_to_auto",
               "openai_prompt_cache",
               "openai_prompt_cache_retention",
               "openai_reasoning_effort",
        "llm_temperature_default",
        "llm_temperature",
        "llm_temperature_extraction",
        "llm_temperature_creative",
        "llm_max_tokens",
        "llm_chunk_processor_max_tokens",
        "llm_max_chunks_per_url",
        "llm_top_k",
        "llm_top_p",
        "LLM_MAX_INPUT_CHARS_FOR_SUMMARY",
        "PROMPT_PATH_WEBSITE_SUMMARIZER",
        "PROMPT_PATH_ATTRIBUTE_EXTRACTOR",
        "PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION",
        "PROMPT_PATH_GERMAN_PARTNER_MATCHING",
        "PROMPT_PATH_GERMAN_SALES_PITCH_GENERATION",
        "SALES_PROMPT_LANGUAGE",
        "MAX_GOLDEN_PARTNERS_IN_PROMPT",
        "partner_match_sparse_top_k",
        "partner_match_dense_top_k",
        "partner_match_fused_top_k",
        "partner_match_rrf_k",
        "PATH_TO_GOLDEN_PARTNERS_DATA",
        "provider_max_inflight_default",
        "provider_max_inflight_openai",
        "provider_max_inflight_gemini",
        "provider_backpressure_cooldown_seconds",
        "provider_backpressure_error_burst_threshold",
        # scraping
        "scraper_max_pages_per_domain",
        "max_depth_internal_links",
        "respect_robots_txt",
        "scraper_http_fallback_enabled",
        "scraper_http_fallback_timeout_seconds",
        "scraper_http_fallback_max_bytes",
        "scraper_http_fallback_min_text_chars",
        # phone settings
        "target_country_codes",
        "default_region_code",
        "force_phone_extraction",
        "enable_phone_retrieval_in_full_pipeline",
        "phone_llm_max_candidates_total",
        "phone_llm_prefer_url_path_keywords",
        "phone_llm_prefer_snippet_keywords",
        "enable_phone_llm_rerank",
        "phone_llm_rerank_max_candidates",
        # caching/reuse
        "reuse_scraped_content_if_available",
        "scraped_content_cache_dirs",
        "scraped_content_cache_min_chars",
        "reuse_phone_results_if_available",
        "phone_results_cache_dir",
    ]

    out: Dict[str, Any] = {}
    for k in allow:
        try:
            if hasattr(app_config, k):
                v = getattr(app_config, k)
                out[k] = _redact_value(k, v)
        except Exception:
            continue
    return out


def _git_info(repo_root: Union[str, Path]) -> Dict[str, Any]:
    """
    Best-effort git version snapshot.
    """
    root = str(Path(repo_root).resolve())

    def _run(args: List[str]) -> Optional[str]:
        try:
            cp = subprocess.run(
                args,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            s = (cp.stdout or "").strip()
            return s if s else None
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = None
    try:
        status = _run(["git", "status", "--porcelain"])
        if status is None:
            dirty = None
        else:
            dirty = bool(status.strip())
    except Exception:
        dirty = None

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def _sys_info() -> Dict[str, Any]:
    try:
        return {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    except Exception:
        return {}


def _file_meta(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    try:
        st = p.stat()
        return {
            "path": str(p),
            "exists": True,
            "size_bytes": int(st.st_size),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        }
    except Exception:
        return {"path": str(p), "exists": False}


def _collect_output_files(run_output_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    base = Path(run_output_dir)
    out: List[Dict[str, Any]] = []
    try:
        if not base.exists():
            return out
        for p in base.rglob("*"):
            try:
                if p.is_dir():
                    continue
                # Skip enormous internal worker artifacts? Keep everything; caller can filter.
                st = p.stat()
                out.append(
                    {
                        "relative_path": str(p.relative_to(base)).replace("\\", "/"),
                        "size_bytes": int(st.st_size),
                        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                    }
                )
            except Exception:
                continue
        # Stable ordering for diffs
        out.sort(key=lambda d: d.get("relative_path", ""))
    except Exception:
        return out
    return out


def _detect_csv_delimiter(path: Union[str, Path]) -> str:
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
        if not sample:
            return ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|:")
            return dialect.delimiter or ","
        except Exception:
            counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|", ":"]}
            best = max(counts.items(), key=lambda kv: kv[1])
            return best[0] if best[1] > 0 else ","
    except Exception:
        return ","


def _count_csv_rows_robust(path: Union[str, Path], max_bytes: int = 200_000_000) -> Optional[int]:
    """
    Robust CSV row count that tolerates quoted multiline fields.
    Returns number of data rows (excludes header). Best-effort.
    """
    try:
        p = Path(path)
        if not p.exists():
            return None
        if p.stat().st_size > max_bytes:
            return None
        delim = _detect_csv_delimiter(p)
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f, delimiter=delim)
            _ = next(r, None)  # header
            return sum(1 for _ in r)
    except Exception:
        return None


def _should_count_rows() -> bool:
    raw = (os.getenv("RUN_MANIFEST_COUNT_ROWS", "True") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _repo_root_from_run_dir(run_output_dir: Union[str, Path]) -> str:
    """
    Infer repo root as parent of output_data when possible.
    run_output_dir is typically: <repo_root>/output_data/<run_id>
    """
    p = Path(run_output_dir).resolve()
    try:
        # output_data/<run_id>
        if p.parent.name.lower() == "output_data":
            return str(p.parent.parent)
    except Exception:
        pass
    # fallback: cwd
    return str(Path(".").resolve())


def _global_history_path(run_output_dir: Union[str, Path]) -> Path:
    p = Path(run_output_dir).resolve()
    out_data = p.parent if p.parent.name.lower() == "output_data" else p
    return out_data / GLOBAL_HISTORY_FILENAME


def _append_global_history(run_output_dir: Union[str, Path], summary_obj: Dict[str, Any]) -> None:
    try:
        hist = _global_history_path(run_output_dir)
        _safe_mkdir(hist.parent)
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary_obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


@dataclass
class ManifestHandle:
    run_output_dir: str
    run_id: str


def start_run_manifest(
    *,
    run_output_dir: str,
    run_id: str,
    pipeline_name: str,
    argv: Optional[List[str]] = None,
    run_command: Optional[str] = None,
    args_obj: Optional[Any] = None,
    app_config: Optional[Any] = None,
    input_file_abs: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ManifestHandle:
    """
    Create or update a run manifest at the start of a run.
    """
    try:
        repo_root = _repo_root_from_run_dir(run_output_dir)
        manifest_path = os.path.join(run_output_dir, MANIFEST_FILENAME)
        base = _safe_json_read(manifest_path) or {}
        # Track an epoch start time for reliable duration calculation.
        if "_started_epoch" not in base or not isinstance(base.get("_started_epoch"), (int, float)):
            base["_started_epoch"] = time.time()

        args_dict: Optional[Dict[str, Any]] = None
        if args_obj is not None:
            try:
                # argparse.Namespace supports vars()
                args_dict = dict(vars(args_obj))
            except Exception:
                args_dict = None

        obj: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "status": "running",
            "started_at_utc": base.get("started_at_utc") or _now_iso(),
            "finished_at_utc": None,
            "duration_seconds": None,
            "run_command": run_command or " ".join(argv or sys.argv),
            "argv": list(argv or sys.argv),
            "args": args_dict,
            "git": _git_info(repo_root),
            "system": _sys_info(),
            "input_file": _file_meta(input_file_abs) if input_file_abs else None,
            "config_snapshot": _snapshot_config(app_config) if app_config is not None else None,
            "extra": extra or {},
            "outputs": None,
            "output_row_counts": None,
            "metrics_summary": None,
            "errors": [],
        }

        # Merge in any previous data without losing fields (start is conservative)
        merged = dict(base)
        merged.update({k: v for k, v in obj.items() if v is not None or k not in merged})

        _safe_json_write(manifest_path, merged)
    except Exception:
        pass

    return ManifestHandle(run_output_dir=run_output_dir, run_id=run_id)


def finalize_run_manifest(
    *,
    handle: ManifestHandle,
    status: str,
    run_metrics: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
) -> None:
    """
    Finalize a manifest: set completion status + collect outputs + append global history.
    """
    try:
        run_output_dir = handle.run_output_dir
        run_id = handle.run_id
        manifest_path = os.path.join(run_output_dir, MANIFEST_FILENAME)
        base = _safe_json_read(manifest_path) or {}

        started = base.get("started_at_utc")
        finished = _now_iso()
        dur = None
        try:
            # If we have an epoch start, use that; otherwise parse ISO.
            start_ts = base.get("_started_epoch")
            if isinstance(start_ts, (int, float)):
                dur = time.time() - float(start_ts)
        except Exception:
            dur = None

        # Always inventory outputs at the end
        outputs = _collect_output_files(run_output_dir)

        # Optional: row counts for key CSV outputs
        row_counts: Dict[str, Optional[int]] = {}
        if _should_count_rows():
            try:
                base_dir = Path(run_output_dir)
                # Heuristic: count stable finals if present, else count live.
                candidates = list(base_dir.glob("SalesOutreachReport_*.csv")) + list(base_dir.glob("input_augmented_*.csv")) + list(base_dir.glob("phone_extraction_results_*.csv"))
                # exclude worker slice files by path
                for p in candidates:
                    try:
                        rel = str(p.relative_to(base_dir)).replace("\\", "/")
                        if rel.startswith("workers/") or rel.startswith("jobs/"):
                            continue
                        row_counts[rel] = _count_csv_rows_robust(p)
                    except Exception:
                        continue
            except Exception:
                pass

        worker_concurrency = None
        try:
            extra = base.get("extra") if isinstance(base.get("extra"), dict) else {}
            if isinstance(extra, dict):
                candidate = extra.get("worker_concurrency")
                if isinstance(candidate, dict):
                    worker_concurrency = {
                        "requested_workers": candidate.get("requested_workers"),
                        "effective_workers": candidate.get("effective_workers"),
                        "provider": candidate.get("provider"),
                        "provider_ceiling": candidate.get("provider_ceiling"),
                    }
        except Exception:
            worker_concurrency = None

        metrics_summary: Optional[Dict[str, Any]] = None
        if isinstance(run_metrics, dict):
            # Keep a compact summary for the manifest + global history
            metrics_summary = {
                "input_rows_count": (run_metrics.get("data_processing_stats") or {}).get("input_rows_count"),
                "rows_successfully_processed": (run_metrics.get("data_processing_stats") or {}).get("rows_successfully_processed_main_flow")
                or (run_metrics.get("data_processing_stats") or {}).get("rows_successfully_processed_pass1"),
                "rows_failed": (run_metrics.get("data_processing_stats") or {}).get("rows_failed_main_flow")
                or (run_metrics.get("data_processing_stats") or {}).get("rows_failed_pass1"),
                "total_duration_seconds": run_metrics.get("total_duration_seconds"),
            }
        if worker_concurrency:
            metrics_summary = dict(metrics_summary or {})
            metrics_summary["worker_concurrency"] = worker_concurrency

        # Update manifest
        out = dict(base)
        out["status"] = status
        out["finished_at_utc"] = finished
        out["duration_seconds"] = float(dur) if isinstance(dur, (int, float)) else out.get("duration_seconds")
        out["outputs"] = outputs
        out["output_row_counts"] = row_counts if row_counts else out.get("output_row_counts")
        out["metrics_summary"] = metrics_summary
        if errors:
            out.setdefault("errors", [])
            try:
                out["errors"].extend([str(e) for e in errors if str(e)])
            except Exception:
                pass

        _safe_json_write(manifest_path, out)

        # Append global history summary line
        summary = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "pipeline_name": out.get("pipeline_name"),
            "status": status,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "duration_seconds": out.get("duration_seconds"),
            "run_command": out.get("run_command"),
            "input_file": (out.get("input_file") or {}).get("path") if isinstance(out.get("input_file"), dict) else None,
            "git": out.get("git"),
            "metrics_summary": metrics_summary,
            "output_row_counts": row_counts if row_counts else None,
        }
        _append_global_history(run_output_dir, summary)
    except Exception:
        pass

