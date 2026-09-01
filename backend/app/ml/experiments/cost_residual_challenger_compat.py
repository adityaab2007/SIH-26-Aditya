"""Compatibility layer for Cost-only Exp75-79 on current compound production models.

This module preserves each Cost experiment's hypothesis while making the
comparison harness consume the exact current production preparation contract.
The frozen frame is enriched through the canonical Exp34 path-feature pipeline,
the comparison cohort uses the production Cost cohort plus deterministic Exp35
calibration gate, and Delay remains an exact production control.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from backend.app.ml.experiments import cost_residual_challenger_common as legacy
from backend.app.ml.experiments.nextgen_common import _compare as production_comparison_cohort
from backend.app.ml.experiments.nextgen_common import _prepare as prepare_current_production_frame

ChallengerConfig = legacy.ChallengerConfig


def _normalize_feature_missing(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Convert only literal pd.NA values in model features to sklearn-safe np.nan."""
    result = frame.copy()
    for column in dict.fromkeys(features):
        if column not in result.columns:
            continue
        object_series = result[column].astype(object)
        if object_series.map(lambda value: value is pd.NA).any():
            result[column] = object_series.where(object_series.notna(), np.nan)
    return result


def _X(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return _normalize_feature_missing(frame.reindex(columns=list(features)), list(features))


def _require_features(frame: pd.DataFrame, features: list[str], label: str) -> None:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} is missing current production features: {', '.join(missing)}"
        )


def fit_challenger(
    config: ChallengerConfig,
    *,
    data: pd.DataFrame,
    training_start: int,
    training_end: int,
    test_end: int,
    production_bundle: dict,
    production_receipt: dict,
) -> dict:
    # Current production is Exp34 -> Exp35 -> Exp61. Reuse the canonical
    # preparation instead of reindexing absent production features to NaN.
    frame = prepare_current_production_frame(data)
    train, raw_test = legacy.temporal_project_split(
        frame, int(training_start), int(training_end), int(test_end)
    )
    test = production_comparison_cohort(raw_test)

    contract = legacy.target_feature_contract(production_bundle.get("metadata") or {})
    cost_features = list(contract.get("cost") or production_receipt.get("features_used") or [])
    delay_features = list(contract.get("delay") or production_receipt.get("features_used") or [])
    if not cost_features or not delay_features:
        raise ValueError("Production target feature contract unavailable.")

    _require_features(train, cost_features, "training frame")
    _require_features(test, cost_features, "comparison Cost frame")
    _require_features(test, delay_features, "comparison Delay frame")
    algorithm = legacy._algorithm(production_bundle, production_receipt)

    train = _normalize_feature_missing(train, cost_features)
    test = _normalize_feature_missing(test, list(dict.fromkeys(cost_features + delay_features)))

    production_cost = np.asarray(
        production_bundle["cost"].predict(_X(test, cost_features)), dtype=float
    )
    production_delay = np.maximum(
        0.0,
        np.asarray(production_bundle["delay"].predict(_X(test, delay_features)), dtype=float),
    )

    if config.strategy == "uncertainty_shrinkage":
        policy = legacy._fit_exp79(train, cost_features, algorithm)
        disagreement = np.vstack(
            [m.predict(_X(test, cost_features)) for m in policy["models"]]
        ).std(axis=0)
        experiment_cost = production_cost.copy()
        high = disagreement >= policy["threshold"]
        experiment_cost[high] = (
            (1.0 - policy["alpha"]) * experiment_cost[high]
            + policy["alpha"] * policy["target_median"]
        )
        serializable_policy = {k: v for k, v in policy.items() if k != "models"}
    else:
        oof = legacy._rolling_oof(train, cost_features, algorithm, max_folds=3)
        if config.strategy == "revised_cost_reliability":
            policy = legacy._fit_exp75(oof)
            correction = legacy._apply_exp75(test, policy)
        elif config.strategy == "medium_project_specialist":
            policy = legacy._fit_exp76(oof)
            correction = legacy._apply_exp76(test, policy)
        elif config.strategy == "early_financial_surrogate":
            policy = legacy._fit_exp77(oof)
            correction = legacy._apply_exp77(test, policy)
        elif config.strategy == "cross_window_consensus":
            policy = legacy._fit_exp78(oof)
            correction = legacy._apply_exp78(test, policy, production_cost)
        else:
            raise ValueError(f"Unknown challenger strategy: {config.strategy}")
        experiment_cost = production_cost + correction
        serializable_policy = policy

    experiment_delay = production_delay.copy()
    test["production_cost"] = production_cost
    test["experiment_cost"] = experiment_cost
    test["production_delay"] = production_delay
    test["experiment_delay"] = experiment_delay

    prod_cost_metric = legacy._metric(test, legacy.TARGET, "production_cost")
    exp_cost_metric = legacy._metric(test, legacy.TARGET, "experiment_cost")
    prod_delay_metric = legacy._metric(test, legacy.DELAY_TARGET, "production_delay")
    exp_delay_metric = legacy._metric(test, legacy.DELAY_TARGET, "experiment_delay")
    cost_imp = legacy._improvement(prod_cost_metric["MAE"], exp_cost_metric["MAE"])
    delay_imp = legacy._improvement(prod_delay_metric["MAE"], exp_delay_metric["MAE"])

    paired_cost = legacy.paired_project_mae_comparison(
        test,
        actual=legacy.TARGET,
        baseline_prediction="production_cost",
        candidate_prediction="experiment_cost",
    )
    paired_delay = legacy.paired_project_mae_comparison(
        test,
        actual=legacy.DELAY_TARGET,
        baseline_prediction="production_delay",
        candidate_prediction="experiment_delay",
        seed=26104,
    )

    runtime_state = {
        "strategy": config.strategy,
        "policy": policy,
        "serializable_policy": serializable_policy,
        "cost_features": cost_features,
        "delay_features": delay_features,
        "production_cost_model": production_bundle["cost"],
        "production_delay_model": production_bundle["delay"],
    }
    run_id = f"{config.experiment_id}-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    overall = {
        "production_cost_mae": prod_cost_metric["MAE"],
        "experiment_cost_mae": exp_cost_metric["MAE"],
        "cost_improvement_percentage": round(cost_imp, 6) if cost_imp is not None else None,
        "production_delay_mae": prod_delay_metric["MAE"],
        "experiment_delay_mae": exp_delay_metric["MAE"],
        "delay_improvement_percentage": round(delay_imp, 6) if delay_imp is not None else None,
        "comparison_test_projects": int(test.canonical_project_id.nunique()),
        "comparison_test_snapshots": int(len(test)),
        "paired_project_cost_comparison": paired_cost,
        "paired_project_delay_comparison": paired_delay,
        "cost_only_delay_predictions_identical": bool(
            np.array_equal(production_delay, experiment_delay)
        ),
        "production_wrapper_feature_policy": "canonical_current_production_prepare_and_compare",
        "verdict": legacy._verdict(cost_imp, delay_imp),
    }
    experiment = {
        "experiment_id": config.experiment_id,
        "experiment_name": config.name,
        "scope": "cost",
        "run_id": run_id,
        "model_role": "experiment",
        "promotion_allowed": False,
        "strategy": config.strategy,
        "training_only_policy": serializable_policy,
        "future_holdout_used_for_selection": False,
        "delay_mode": "production_control",
    }
    return {
        "experiment": experiment,
        "overall_comparison": overall,
        "runtime_state": runtime_state,
    }


def filter_rows(held: pd.DataFrame, runtime_state: dict) -> pd.DataFrame:
    return held.copy()


def predict_row(row: pd.DataFrame, runtime_state: dict) -> dict:
    if not isinstance(row, pd.DataFrame):
        row = pd.DataFrame([row])
    cost_features = runtime_state["cost_features"]
    delay_features = runtime_state["delay_features"]
    prod_cost = np.asarray(
        runtime_state["production_cost_model"].predict(_X(row, cost_features)), dtype=float
    )
    prod_delay = np.maximum(
        0.0,
        np.asarray(
            runtime_state["production_delay_model"].predict(_X(row, delay_features)),
            dtype=float,
        ),
    )
    strategy = runtime_state["strategy"]
    policy = runtime_state["policy"]
    if strategy == "revised_cost_reliability":
        candidate_cost = prod_cost + legacy._apply_exp75(row, policy)
    elif strategy == "medium_project_specialist":
        candidate_cost = prod_cost + legacy._apply_exp76(row, policy)
    elif strategy == "early_financial_surrogate":
        candidate_cost = prod_cost + legacy._apply_exp77(row, policy)
    elif strategy == "cross_window_consensus":
        candidate_cost = prod_cost + legacy._apply_exp78(row, policy, prod_cost)
    elif strategy == "uncertainty_shrinkage":
        disagreement = np.vstack(
            [m.predict(_X(row, cost_features)) for m in policy["models"]]
        ).std(axis=0)
        candidate_cost = prod_cost.copy()
        high = disagreement >= policy["threshold"]
        candidate_cost[high] = (
            (1.0 - policy["alpha"]) * candidate_cost[high]
            + policy["alpha"] * policy["target_median"]
        )
    else:
        raise ValueError(strategy)
    return {
        "predicted_cost_overrun": float(candidate_cost[0]),
        "predicted_delay_days": float(prod_delay[0]),
    }
