"""Experiment 75: reliability-weighted revised-cost correction."""
from __future__ import annotations

from backend.app.ml.experiments.cost_residual_challenger_compat import ChallengerConfig, fit_challenger, filter_rows, predict_row

EXPERIMENT_ID = "exp_75"
EXPERIMENT_SEQUENCE = 75
EXPERIMENT_NAME = "Reliability-Weighted Revised-Cost Correction"
EXPERIMENT_SCOPE = "cost"
CONFIG = ChallengerConfig(EXPERIMENT_ID, EXPERIMENT_SEQUENCE, EXPERIMENT_NAME, "revised_cost_reliability")


def _production_cost_family(bundle: dict) -> str:
    current = bundle["cost"]
    seen: set[int] = set()
    while not hasattr(current, "named_steps"):
        marker = id(current)
        if marker in seen:
            raise ValueError("Cycle while unwrapping production Cost model.")
        seen.add(marker)
        next_model = None
        for attribute in ("model", "base_model", "estimator", "pipeline"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                next_model = candidate
                break
        if next_model is None:
            raise ValueError(f"Unable to unwrap production Cost model {type(current).__name__}.")
        current = next_model
    estimator = current.named_steps["model"]
    lowered = type(estimator).__name__.lower()
    if "lgbm" in lowered or "lightgbm" in lowered:
        return "lightgbm"
    if "xgb" in lowered or "xgboost" in lowered:
        return "xgboost"
    if "extra" in lowered:
        return "extra_trees"
    raise ValueError(f"Unable to identify production Cost estimator family from {type(estimator).__name__}.")


def fit_against_production(**kwargs):
    local_kwargs = dict(kwargs)
    bundle = dict(local_kwargs["production_bundle"])
    metadata = dict(bundle.get("metadata") or {})
    selected = dict(metadata.get("selected_algorithms") or {})
    selected["cost"] = _production_cost_family(bundle)
    metadata["selected_algorithms"] = selected
    bundle["metadata"] = metadata
    local_kwargs["production_bundle"] = bundle
    return fit_challenger(CONFIG, **local_kwargs)


def filter_comparable_rows(held, runtime_state):
    return filter_rows(held, runtime_state)


def predict_project(row, runtime_state):
    return predict_row(row, runtime_state)
