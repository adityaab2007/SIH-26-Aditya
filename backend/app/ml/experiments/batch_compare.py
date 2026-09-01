"""Batch-only production-vs-experiment comparison helper.

Research experiment workflows use this helper directly so adapter_exp*.py files do
not need to opt into the interactive lifecycle comparison registry.
"""
from __future__ import annotations

from types import ModuleType

from backend.app.services import lifecycle_retraining_service as retraining
from backend.app.services import lifecycle_simulation_service as simulation


def run_batch_comparison(adapter_module: ModuleType, start_year: int, end_year: int) -> dict:
    """Freshly retrain production and evaluate one explicit experiment module."""
    start_year, end_year = int(start_year), int(end_year)
    data, _identity, _min_year, max_year = retraining._training_data()
    production = retraining.retrain_lifecycle(start_year, end_year)
    production_bundle = simulation._artifact_bundle(start_year, end_year, production.get("run_id"))
    fitted = adapter_module.fit_against_production(
        data=data,
        training_start=start_year,
        training_end=end_year,
        test_end=max_year,
        production_bundle=production_bundle,
        production_receipt=production,
    )
    if not isinstance(fitted, dict):
        raise ValueError("Experiment adapter returned an invalid fit result.")
    experiment = dict(fitted.get("experiment") or {})
    overall = dict(fitted.get("overall_comparison") or {})
    if not experiment.get("run_id"):
        raise ValueError("Experiment adapter did not return an experiment run_id.")
    return {
        "status": "success",
        "production": production,
        "experiment": experiment,
        "overall_comparison": overall,
    }
