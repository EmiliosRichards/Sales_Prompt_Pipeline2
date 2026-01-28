"""
One-off: generate a short German description (<=100 words) for each row in a CSV.

This is intended to run independently of the main pipeline.

Input blob (same as --pitch-from-description):
- Short Description (fallback: Description, Combined_Description)
- Keywords (fallback: keywords)
- reasoning (fallback: Reasoning)

Output:
- Writes a new CSV with an appended column: "Short German Description"
- Supports resume if the output file already exists (continues from last written row)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import google.generativeai.types as genai_types

# Ensure repo root is on sys.path so `import src...` works when running from scripts/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.core.config import AppConfig
from src.llm_clients.gemini_client import GeminiClient
from src.utils.llm_processing_helpers import extract_json_from_text, load_prompt_template


OUT_COL = "Short German Description"


def _normalize_cell(val: Optional[str]) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _build_blob(row: Dict[str, str]) -> str:
    sd = _normalize_cell(row.get("Short Description")) or _normalize_cell(row.get("Description")) or _normalize_cell(row.get("Combined_Description"))
    kw = _normalize_cell(row.get("Keywords")) or _normalize_cell(row.get("keywords"))
    rs = _normalize_cell(row.get("reasoning")) or _normalize_cell(row.get("Reasoning"))

    parts = []
    if sd:
        parts.append(f"Short description:\n{sd}")
    if kw:
        parts.append(f"Keywords:\n{kw}")
    if rs:
        parts.append(f"Reasoning:\n{rs}")
    return "\n\n".join(parts).strip()


def _truncate_100_words(text: str) -> str:
    words = [w for w in (text or "").strip().split() if w]
    if len(words) <= 100:
        return (text or "").strip()
    return " ".join(words[:100]).strip()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class CacheEntry:
    summary: str


def _load_cache(cache_path: str) -> Dict[str, CacheEntry]:
    cache: Dict[str, CacheEntry] = {}
    if not cache_path or not os.path.exists(cache_path):
        return cache
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    k = obj.get("key")
                    v = obj.get("summary")
                    if isinstance(k, str) and k and isinstance(v, str):
                        cache[k] = CacheEntry(summary=v)
                except Exception:
                    continue
    except Exception:
        pass
    return cache


def _append_cache(cache_path: str, key: str, summary: str) -> None:
    if not cache_path:
        return
    try:
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "summary": summary}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _count_written_rows(output_csv: str) -> int:
    """
    Return number of data rows already written (excluding header).
    If file doesn't exist or is empty, returns 0.
    """
    if not output_csv or not os.path.exists(output_csv):
        return 0
    try:
        # IMPORTANT: rows may contain embedded newlines inside quoted fields, so we must
        # count via the CSV parser, not by counting file lines.
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            _ = next(r, None)  # header
            return sum(1 for _ in r)
    except Exception:
        return 0


def _call_llm_short_german_summary(
    gemini_client: GeminiClient,
    prompt_template: str,
    blob: str,
    cfg: AppConfig,
) -> Tuple[Optional[str], Optional[str], Dict[str, int]]:
    """
    Returns (summary, error, token_stats)
    """
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not blob.strip():
        return "", None, token_stats

    formatted_prompt = prompt_template.replace("{{INPUT_DESCRIPTION_PLACEHOLDER}}", blob)

    max_out = int(getattr(cfg, "llm_max_tokens_description_de_summary", 256) or 256)
    generation_config_dict = {
        "response_mime_type": "text/plain",
        "candidate_count": 1,
        "max_output_tokens": max_out,
        "temperature": getattr(cfg, "llm_temperature_extraction", 0.2),
    }
    if getattr(cfg, "llm_top_k", None) is not None:
        generation_config_dict["top_k"] = cfg.llm_top_k
    if getattr(cfg, "llm_top_p", None) is not None:
        generation_config_dict["top_p"] = cfg.llm_top_p
    generation_config = genai_types.GenerationConfig(**generation_config_dict)

    system_instruction_text = (
        "You are a summarization/translation assistant. Your entire response MUST be a single, valid JSON formatted string. "
        "Do NOT include any explanations, markdown formatting (like ```json), or any other text outside of this JSON string. "
        "Return JSON matching: {\"german_summary\": \"...\"}."
    )

    contents_for_api = [{"role": "user", "parts": [{"text": formatted_prompt}]}]
    try:
        resp = gemini_client.generate_content_with_retry(
            contents=contents_for_api,
            generation_config=generation_config,
            system_instruction=system_instruction_text,
            file_identifier_prefix="GERMAN_SHORT_DESC_ONEOFF",
            triggering_input_row_id="N/A",
            triggering_company_name="N/A",
        )
        if not resp:
            return None, "No response from LLM client.", token_stats
        raw = getattr(resp, "text", "") or ""
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            token_stats["prompt_tokens"] = resp.usage_metadata.prompt_token_count or 0
            token_stats["completion_tokens"] = resp.usage_metadata.candidates_token_count or 0
            token_stats["total_tokens"] = resp.usage_metadata.total_token_count or 0
        json_str = extract_json_from_text(raw)
        if not json_str:
            return None, f"Failed to extract JSON. Raw head: {raw[:200]}", token_stats
        obj = json.loads(json_str)
        summary = obj.get("german_summary")
        if not isinstance(summary, str):
            return None, "JSON missing german_summary string.", token_stats
        return _truncate_100_words(summary), None, token_stats
    except Exception as e:
        return None, f"Exception calling/parsing LLM: {type(e).__name__}: {e}", token_stats


_THREAD_LOCAL = threading.local()


def _get_thread_local_client(cfg: AppConfig) -> GeminiClient:
    c = getattr(_THREAD_LOCAL, "client", None)
    if c is None:
        c = GeminiClient(config=cfg)
        _THREAD_LOCAL.client = c
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV (comma-delimited).")
    ap.add_argument("--output", default="", help="Path to output CSV. Defaults to <input>_with_short_german_desc.csv")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads for LLM calls (1 = sequential).")
    ap.add_argument("--max-rows", type=int, default=0, help="For testing: process only first N rows (0 = all).")
    ap.add_argument("--resume", action="store_true", default=True, help="Resume if output exists (default: true).")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume behavior and overwrite output.")
    ap.add_argument("--cache-jsonl", default="", help="Optional JSONL cache path (hash -> summary).")
    ap.add_argument("--errors-jsonl", default="", help="Optional JSONL file to log per-row errors.")
    args = ap.parse_args()

    input_csv = os.path.normpath(args.input)
    output_csv = os.path.normpath(args.output) if args.output else os.path.splitext(input_csv)[0] + "_with_short_german_desc.csv"
    resume = bool(args.resume) and not bool(args.no_resume)

    # Default cache/errors paths next to output
    cache_path = os.path.normpath(args.cache_jsonl) if args.cache_jsonl else (output_csv + ".cache.jsonl")
    errors_path = os.path.normpath(args.errors_jsonl) if args.errors_jsonl else (output_csv + ".errors.jsonl")

    cfg = AppConfig()
    # Initialize once to validate key/config; worker threads may create their own clients.
    _ = GeminiClient(config=cfg)

    prompt_path = getattr(cfg, "PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION", "")
    if not prompt_path:
        raise SystemExit("PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION not configured in AppConfig.")
    prompt_template = load_prompt_template(prompt_path)

    cache = _load_cache(cache_path)

    already_written = _count_written_rows(output_csv) if (resume and os.path.exists(output_csv)) else 0
    if not resume and os.path.exists(output_csv):
        os.remove(output_csv)
        already_written = 0

    t0 = time.time()
    processed = 0
    succeeded = 0
    failed = 0

    # Open input
    with open(input_csv, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header.")
        in_fields = list(reader.fieldnames)
        if OUT_COL in in_fields:
            # If already present in input, we still append to the end in output (no duplicates),
            # but prefer the existing column value unless it's blank.
            pass
        out_fields = [c for c in in_fields if c != OUT_COL] + [OUT_COL]

        # Prepare output writer (append if resuming, else write header).
        write_header = True
        mode = "w"
        if resume and os.path.exists(output_csv) and already_written > 0:
            write_header = False
            mode = "a"

        with open(output_csv, mode, encoding="utf-8-sig", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()

            # Skip rows that are already written (resume).
            skipped = 0
            while skipped < already_written:
                try:
                    next(reader)
                    skipped += 1
                except StopIteration:
                    break

            workers = max(1, int(getattr(args, "workers", 1) or 1))
            max_inflight = max(1, workers * 4)

            next_to_write = already_written + 1
            pending: Dict[int, Dict[str, str]] = {}
            inflight: Dict[Future, Tuple[int, Dict[str, str], str]] = {}  # future -> (row_idx, row, cache_key)

            def _log_error(row_idx: int, row: Dict[str, str], err: str) -> None:
                try:
                    with open(errors_path, "a", encoding="utf-8") as ef:
                        ef.write(
                            json.dumps(
                                {
                                    "row_index_1based": row_idx,
                                    "company": row.get("Company") or row.get("CompanyName") or row.get("Company Name"),
                                    "url": row.get("Website") or row.get("GivenURL") or row.get("URL"),
                                    "error": err,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except Exception:
                    pass

            def _maybe_flush() -> None:
                nonlocal next_to_write, succeeded, failed
                while next_to_write in pending:
                    out_row = pending.pop(next_to_write)
                    writer.writerow(out_row)
                    # Treat blank summary as "failed" only if we explicitly logged an error marker.
                    # For empty blobs we write "" but it is not an error.
                    if out_row.get("__error_marker__") == "1":
                        failed += 1
                        out_row.pop("__error_marker__", None)
                    else:
                        succeeded += 1
                        out_row.pop("__error_marker__", None)
                    next_to_write += 1

            def _submit_llm(blob: str) -> Tuple[Optional[str], Optional[str], Dict[str, int]]:
                # Each worker thread keeps its own GeminiClient instance.
                client = _get_thread_local_client(cfg)
                return _call_llm_short_german_summary(client, prompt_template, blob, cfg)

            def _collect_done(block: bool = False) -> None:
                nonlocal processed
                if not inflight:
                    return
                if block:
                    done, _not = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                else:
                    done, _not = wait(set(inflight.keys()), timeout=0, return_when=FIRST_COMPLETED)
                if not done:
                    return
                for fut in list(done):
                    row_idx, row, key = inflight.pop(fut)
                    try:
                        summary, err, _tok = fut.result()
                    except Exception as e:
                        summary, err = None, f"Exception in worker future: {type(e).__name__}: {e}"
                    if summary is None and err:
                        row[OUT_COL] = ""
                        row["__error_marker__"] = "1"
                        _log_error(row_idx, row, err)
                    else:
                        row[OUT_COL] = summary or ""
                        if key:
                            cache[key] = CacheEntry(summary=row[OUT_COL])
                            _append_cache(cache_path, key, row[OUT_COL])
                    pending[row_idx] = row
                _maybe_flush()

            if workers == 1:
                # Sequential mode: still goes through the same buffering to keep behavior identical.
                for row_idx, row in enumerate(reader, start=already_written + 1):
                    if args.max_rows and processed >= args.max_rows:
                        break
                    processed += 1

                    existing_val = _normalize_cell(row.get(OUT_COL))
                    if existing_val:
                        row[OUT_COL] = existing_val
                        pending[row_idx] = row
                        _maybe_flush()
                        continue

                    blob = _build_blob(row)
                    if not blob:
                        row[OUT_COL] = ""
                        pending[row_idx] = row
                        _maybe_flush()
                        continue

                    key = _sha1(blob)
                    if key and key in cache:
                        row[OUT_COL] = cache[key].summary
                        pending[row_idx] = row
                        _maybe_flush()
                        continue

                    summary, err, _tok = _submit_llm(blob)
                    if summary is None and err:
                        row[OUT_COL] = ""
                        row["__error_marker__"] = "1"
                        _log_error(row_idx, row, err)
                    else:
                        row[OUT_COL] = summary or ""
                        if key:
                            cache[key] = CacheEntry(summary=row[OUT_COL])
                            _append_cache(cache_path, key, row[OUT_COL])
                    pending[row_idx] = row
                    _maybe_flush()

                    if processed % 25 == 0:
                        dt = time.time() - t0
                        rate = processed / dt if dt > 0 else 0
                        print(f"[progress] processed={processed} ok={succeeded} fail={failed} rate={rate:.2f} rows/s output={output_csv}")
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for row_idx, row in enumerate(reader, start=already_written + 1):
                        if args.max_rows and processed >= args.max_rows:
                            break
                        processed += 1

                        existing_val = _normalize_cell(row.get(OUT_COL))
                        if existing_val:
                            row[OUT_COL] = existing_val
                            pending[row_idx] = row
                            _maybe_flush()
                            continue

                        blob = _build_blob(row)
                        if not blob:
                            row[OUT_COL] = ""
                            pending[row_idx] = row
                            _maybe_flush()
                            continue

                        key = _sha1(blob)
                        if key and key in cache:
                            row[OUT_COL] = cache[key].summary
                            pending[row_idx] = row
                            _maybe_flush()
                            continue

                        fut = ex.submit(_submit_llm, blob)
                        inflight[fut] = (row_idx, row, key)

                        # Backpressure
                        while len(inflight) >= max_inflight:
                            _collect_done(block=True)

                        # Opportunistically collect completions
                        _collect_done(block=False)

                        if processed % 25 == 0:
                            dt = time.time() - t0
                            rate = processed / dt if dt > 0 else 0
                            print(f"[progress] processed={processed} ok={succeeded} fail={failed} rate={rate:.2f} rows/s output={output_csv}")

                    # Drain remaining futures
                    while inflight:
                        _collect_done(block=True)

                # Final flush (should be empty)
                _maybe_flush()

    dt = time.time() - t0
    print(f"[done] processed={processed} ok={succeeded} fail={failed} seconds={dt:.1f} output={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

