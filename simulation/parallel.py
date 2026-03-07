"""Lightweight helpers for running independent simulation tasks in parallel."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, List, Sequence, TypeVar


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
    """Map independent tasks either serially or with a process pool."""
    if n_jobs <= 1 or len(tasks) <= 1:
        results: List[R] = []
        for idx, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if verbose and idx % progress_every == 0:
                print(f"{progress_label} {idx}/{len(tasks)}")
        return results

    chunksize = max(1, len(tasks) // (n_jobs * 4))
    try:
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for idx, result in enumerate(executor.map(worker, tasks, chunksize=chunksize), start=1):
                results.append(result)
                if verbose and idx % progress_every == 0:
                    print(f"{progress_label} {idx}/{len(tasks)}")
        return results
    except (PermissionError, OSError):
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for idx, result in enumerate(executor.map(worker, tasks), start=1):
                results.append(result)
                if verbose and idx % progress_every == 0:
                    print(f"{progress_label} {idx}/{len(tasks)}")
        return results
