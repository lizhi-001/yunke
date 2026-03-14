"""Lightweight helpers for running independent simulation tasks in parallel.

Uses ProcessPoolExecutor as the default process-pool backend (one fresh pool
per call), so that concurrent callers from different threads each get their own
independent worker pool and the full n_jobs budget is honoured.

Fallback chain: ProcessPoolExecutor → ThreadPoolExecutor.

Note: loky's get_reusable_executor is a process-level singleton — sharing it
across threads causes all callers to compete for the same fixed-size pool,
defeating the per-caller n_jobs budget.  We therefore use a fresh
ProcessPoolExecutor per call instead.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, List, Sequence, TypeVar


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

    Each call creates its own ProcessPoolExecutor so concurrent callers
    (e.g. two seed threads) each maintain their own n_jobs workers.
    """
    if n_jobs <= 1 or len(tasks) <= 1:
        results: List[R] = []
        for idx, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if verbose and idx % progress_every == 0:
                print(f"{progress_label} {idx}/{len(tasks)}")
        return results

    chunksize = max(1, len(tasks) // (n_jobs * 4))

    # --- ProcessPoolExecutor (fresh pool per call) -----------------------
    try:
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
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
