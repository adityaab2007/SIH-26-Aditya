"""Experiment 78: cross-window consensus calibration."""
from __future__ import annotations

from backend.app.ml.experiments.cost_residual_challenger_common import (
    ChallengerConfig,
    fit_challenger,
    filter_rows,
    predict_row,
)

EXPERIMENT_ID = "exp_78"
EXPERIMENT_SEQUENCE = 78
EXPERIMENT_NAME = "Cross-Window Consensus Calibration"
EXPERIMENT_SCOPE = "cost"
CONFIG = ChallengerConfig(
    EXPERIMENT_ID,
    EXPERIMENT_SEQUENCE,
    EXPERIMENT_NAME,
    "cross_window_consensus",
)


def _production_cost_family(bundle: dict) -> str:
    """Resolve the raw estimator family through production calibration wrappers."""
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
            raise ValueError(
                f"Unable to unwrap production Cost model {type(current).__name__}."
            )
        current = next_model

    estimator = current.named_steps["model"]
    lowered = type(estimator).__name__.lower()
    if "lgbm" in lowered or "lightgbm" in lowered:
        return "lightgbm"
    if "xgb" in lowered or "xgboost" in lowered:
        return "xgboost"
    if "extra" in lowered:
        return "extra_trees"
    raise ValueError(
        f"Unable to identify production Cost estimator family from {type(estimator).__name__}."
    )


def fit_against_production(**kwargs):
    # Production Exp61 persists Cost as ResidualCalibratedCostModel rather than a
    # raw sklearn Pipeline. The shared Exp75-79 harness still needs the raw
    # estimator family for leakage-safe rolling OOF refits, so provide a temporary
    # metadata view with that family. Production artifacts/models are unchanged.
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
