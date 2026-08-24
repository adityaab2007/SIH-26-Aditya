"""Generic-runner adapter for the historical Experiment 3 implementation."""
from __future__ import annotations

from backend.app.ml.residual_overrun_experiment import run_residual_overrun_experiment


def run_experiment(training_start: int, training_end: int, test_end: int) -> dict:
    return run_residual_overrun_experiment(
        training_start=training_start,
        training_end=training_end,
        test_end=test_end,
        persist=True,
    )
