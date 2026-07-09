"""Worker module-level functions for multiprocessing jobs.

These functions MUST be at module level (not closures) to be picklable
by multiprocessing.spawn on Windows.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any


def run_genetic_process(
    queue: mp.Queue,
    tickers: list[str],
    period: str,
    generations: int,
    population_size: int,
    workers: int,
    use_wfo: bool,
) -> None:
    """Runs the genetic optimization in a child process.

    Communicates progress and results back to the parent via a Queue.
    This runs in a SEPARATE PROCESS so it has its own GIL and does NOT
    block the FastAPI asyncio event loop in the parent process.
    """
    # Signal: process started
    queue.put({"type": "started"})

    try:
        from portfolio.genetic_optimizer import GeneticOptimizer
        from api.utils import sanitize_for_json

        optimizer = GeneticOptimizer(
            tickers=tickers,
            period=period,
            use_wfo=use_wfo,
        )

        def _progress_cb(gen, total, best_fit, metrics_dict):
            queue.put({
                "type": "progress",
                "current_gen": gen,
                "total_gens": total,
                "pct": round(gen / total * 100, 1) if total > 0 else 0,
                "best_fitness": best_fit,
                "gen_metrics": metrics_dict,
            })

        result = optimizer.run(
            generations=generations,
            population_size=population_size,
            workers=workers,
            progress_callback=_progress_cb,
        )

        # Send the final result (sanitized for JSON)
        queue.put({"type": "completed", "result": sanitize_for_json(result)})

    except Exception as e:
        queue.put({"type": "failed", "error": str(e)})
