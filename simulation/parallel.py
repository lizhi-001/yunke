"""Lightweight helpers for running independent simulation tasks in parallel.

Uses loky (via joblib) as the default process-pool backend.  loky avoids the
POSIX-semaphore permission issues that cause stdlib ProcessPoolExecutor to fail
in restricted environments (containers, some cloud VMs).  It also reuses worker
processes across calls, cutting fork/spawn overhead.

Fallback chain: loky → ProcessPoolExecutor → ThreadPoolExecutor.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar

try:
    from joblib.externals.loky import get_reusable_executor as _get_loky_executor
    _HAS_LOKY = True
except ImportError:
    _HAS_LOKY = False


T = TypeVar("T")
R = TypeVar("R")


def run_task_map(
    worker: Callable[[T], R],
    tasks: Sequence[T],
    n_jobs: int,
    verbose: bool = False,
    progress_every: int = 10,
    progress_label: str = "Monte Carlo iteration",
) -> List[R]:
    """Map independent tasks either serially or with a process pool.

    Attempts loky first (robust process pool), then falls back to stdlib
    ProcessPoolExecutor, and finally to ThreadPoolExecutor.
    """
    if n_jobs <= 1 or len(tasks) <= 1:
        results: List[R] = []
        for idx, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if verbose and idx % progress_every == 0:
                print(f"{progress_label} {idx}/{len(tasks)}")
        return results

    chunksize = max(1, len(tasks) // (n_jobs * 4))

    # --- loky (preferred) ------------------------------------------------
    if _HAS_LOKY:
        try:
            executor = _get_loky_executor(max_workers=n_jobs)
            results = []
            for idx, result in enumerate(executor.map(worker, tasks, chunksize=chunksize), start=1):
                results.append(result)
                if verbose and idx % progress_every == 0:
                    print(f"{progress_label} {idx}/{len(tasks)}")
            return results
        except Exception:
            pass  # fall through to stdlib

    # --- stdlib ProcessPoolExecutor --------------------------------------
    try:
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for idx, result in enumerate(executor.map(worker, tasks, chunksize=chunksize), start=1):
                results.append(result)
                if verbose and idx % progress_every == 0:
                    print(f"{progress_label} {idx}/{len(tasks)}")
        return results
    except (PermissionError, OSError):
        pass  # fall through to threads

    # --- ThreadPoolExecutor (last resort) --------------------------------
    results = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for idx, result in enumerate(executor.map(worker, tasks), start=1):
            results.append(result)
            if verbose and idx % progress_every == 0:
                print(f"{progress_label} {idx}/{len(tasks)}")
    return results
