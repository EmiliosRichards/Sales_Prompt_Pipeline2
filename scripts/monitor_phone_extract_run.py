"""
Monitor a phone_extract.py parallel run directory.

Shows:
- How many worker jobs exist vs completed (results CSV present)
- Which workers look "stalled" (no log update for N seconds)
- Last modification times per worker log/result

This is safe to run while the pipeline is running.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class WorkerStatus:
    name: str
    has_results: bool
    log_mtime: Optional[float]
    results_mtime: Optional[float]


def _mtime(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def _fmt_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s2 = divmod(s, 60)
    if m < 60:
        return f"{m}m{s2:02d}s"
    h, m2 = divmod(m, 60)
    return f"{h}h{m2:02d}m"


def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "n/a"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _snapshot(run_dir: Path) -> Tuple[List[WorkerStatus], Optional[str]]:
    workers_dir = run_dir / "workers"
    if not workers_dir.exists():
        return [], f"Missing workers dir: {workers_dir}"

    worker_dirs = sorted([p for p in workers_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    statuses: List[WorkerStatus] = []
    for wd in worker_dirs:
        log_path = wd / f"phone_extract_{wd.name}.log"
        results_path = wd / f"phone_extraction_results_{wd.name}.csv"
        statuses.append(
            WorkerStatus(
                name=wd.name,
                has_results=results_path.exists(),
                log_mtime=_mtime(log_path),
                results_mtime=_mtime(results_path),
            )
        )
    return statuses, None


def _print(statuses: List[WorkerStatus], stale_seconds: int) -> None:
    now = time.time()
    total = len(statuses)
    completed = sum(1 for s in statuses if s.has_results)
    running = total - completed

    # Consider a worker "stalled" if it has no results yet and its log hasn't been updated recently.
    stalled: List[WorkerStatus] = []
    for s in statuses:
        if s.has_results:
            continue
        if s.log_mtime is None:
            stalled.append(s)
            continue
        age = now - s.log_mtime
        if age > stale_seconds:
            stalled.append(s)

    print("")
    print(f"Workers total: {total} | completed: {completed} | running: {running} | stalled(>{stale_seconds}s): {len(stalled)}")

    # Show a short table of the oldest (least recently updated) running workers.
    running_workers = [s for s in statuses if not s.has_results]
    running_workers.sort(key=lambda s: (s.log_mtime or 0.0))

    head = running_workers[:10]
    if head:
        print("")
        print("Oldest-running workers (least recent log update):")
        print("worker\tlog_last_ts\t\tlog_age\tresults?")
        for s in head:
            age = (now - s.log_mtime) if s.log_mtime else None
            print(f"{s.name}\t{_fmt_ts(s.log_mtime)}\t{_fmt_age(age)}\t{str(s.has_results)}")
    else:
        print("")
        print("No running workers (all completed).")

    if stalled:
        print("")
        print("Stalled workers:")
        for s in stalled[:20]:
            age = (now - s.log_mtime) if s.log_mtime else None
            print(f"- {s.name}: last_log={_fmt_ts(s.log_mtime)} age={_fmt_age(age)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to output_data/<run_id>_phone*/ directory")
    ap.add_argument("--stale-seconds", type=int, default=300, help="Seconds without log updates to consider a worker stalled.")
    ap.add_argument("--refresh", type=int, default=0, help="If >0, refresh every N seconds until Ctrl+C.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    while True:
        statuses, err = _snapshot(run_dir)
        if err:
            print(err)
            return
        _print(statuses, stale_seconds=max(1, int(args.stale_seconds)))
        if int(args.refresh) <= 0:
            return
        time.sleep(int(args.refresh))


if __name__ == "__main__":
    main()

