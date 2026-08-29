"""Experiment 13: recency-weighted project training.

Only training influence changes.  The production feature/target contracts,
selected estimator families, temporal split, snapshot eligibility, and
project-balanced evaluation are reused unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import build_experiment_context, experiment_run_directory, new_experiment_manifest
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production, target_feature_contract


EXPERIMENT_ID = "exp_13"
EXPERIMENT_SEQUENCE = 13
EXPERIMENT_NAME = "Recency-Weighted Project Training"
EXPERIMENT_SCOPE = "cost_delay"
CANDIDATE_HALF_LIVES: tuple[int | None, ...] = (None, 15, 10, 7, 5, 3)
# These are the lifecycle trainer's existing seeds.  The challenger changes
# only sample weights; estimator randomness remains controlled.
RANDOM_SEEDS = {"cost": 26203, "delay": 26203}


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _algorithm(bundle: dict, receipt: dict, target: str) -> str:
    name = ((bundle.get("metadata") or {}).get("selected_algorithms") or {}).get(target)
    name = name or (receipt.get("selected_algorithms") or {}).get(target)
    if name in _regressors(1):
        return name
    model = bundle[target].named_steps["model"]
    lowered = type(model).__name__.lower()
    if "lgbm" in lowered:
        return "lightgbm"
    if "xgb" in lowered:
        return "xgboost"
    if "extra" in lowered:
        return "extra_trees"
    raise ValueError(f"Cannot identify production {target} algorithm.")


def compute_project_recency_weights(frame: pd.DataFrame, training_end_year: int, half_life: int | None) -> pd.Series:
    """Return one normalized weight per project using training-side age only."""
    required = {"canonical_project_id", "completion_year"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("recency weighting missing: " + ", ".join(sorted(missing)))
    projects = frame[["canonical_project_id", "completion_year"]].dropna().drop_duplicates("canonical_project_id")
    years = pd.to_numeric(projects["completion_year"], errors="coerce")
    if years.isna().any() or (years > int(training_end_year)).any():
        raise ValueError("recency weights require finite training-side completion years at or before the cutoff")
    if half_life is None:
        raw = pd.Series(1.0, index=projects["canonical_project_id"].astype(str))
    else:
        half_life = float(half_life)
        if not math.isfinite(half_life) or half_life <= 0:
            raise ValueError("half_life must be positive or None for the control")
        raw = pd.Series(
            np.power(0.5, (int(training_end_year) - years.to_numpy(dtype=float)) / half_life),
            index=projects["canonical_project_id"].astype(str),
        )
    if not np.isfinite(raw.to_numpy(dtype=float)).all() or (raw <= 0).any():
        raise ValueError("recency weights must be finite and positive")
    return raw / float(raw.mean())


def apply_project_recency_weights(frame: pd.DataFrame, training_end_year: int, half_life: int | None) -> tuple[pd.DataFrame, pd.Series]:
    """Distribute normalized project influence across retained snapshots."""
    result = frame.copy()
    project_weights = compute_project_recency_weights(result, training_end_year, half_life)
    keys = result["canonical_project_id"].astype(str)
    counts = keys.value_counts()
    result["sample_weight"] = keys.map(project_weights).to_numpy(dtype=float) / keys.map(counts).to_numpy(dtype=float)
    if not np.isfinite(result["sample_weight"].to_numpy(dtype=float)).all() or (result["sample_weight"] <= 0).any():
        raise ValueError("snapshot recency weights must be finite and positive")
    observed = result.groupby(keys, sort=False)["sample_weight"].sum()
    if not np.allclose(observed.to_numpy(), project_weights.loc[observed.index].to_numpy(), rtol=0, atol=1e-10):
        raise AssertionError("snapshot weights do not sum to the normalized project weight")
    return result, project_weights


# Short aliases make the experiment utility easy to exercise in focused tests.
recency_project_weights = compute_project_recency_weights
recency_snapshot_weights = apply_project_recency_weights


def _fit_weighted(train: pd.DataFrame, features: list[str], target: str, algorithm: str, end_year: int, half_life: int | None, seed: int):
    weighted, _ = apply_project_recency_weights(train, end_year, half_life)
    return _fit_pipeline(_regressors(seed)[algorithm], weighted, features, target)


def _candidate_validation(train: pd.DataFrame, features: list[str], target: str, algorithm: str, training_end: int, seed: int) -> tuple[int | None, list[dict[str, Any]]]:
    validation_year = int(pd.to_numeric(train.completion_year, errors="coerce").max())
    fitting = train[train.completion_year.lt(validation_year)].copy()
    validation = train[train.completion_year.eq(validation_year)].copy()
    if fitting.canonical_project_id.nunique() < 5 or validation.canonical_project_id.nunique() < 2:
        raise ValueError(f"Insufficient internal temporal validation data for {target}.")
    validation = assign_project_balanced_weights(validation)
    scores = []
    for candidate_index, half_life in enumerate(CANDIDATE_HALF_LIVES):
        model = _fit_weighted(fitting, features, target, algorithm, validation_year - 1, half_life, seed)
        predicted = np.maximum(0, model.predict(validation[features])) if target == "actual_delay_days" else model.predict(validation[features])
        metrics = _regression_metrics(validation[target], predicted, validation.sample_weight, validation.canonical_project_id)
        scores.append({"half_life": half_life, "cost_mae" if target == "actual_cost_overrun_percentage" else "delay_mae": metrics["MAE"], "RMSE": metrics["RMSE"], "validation_year": validation_year, "validation_projects": int(validation.canonical_project_id.nunique()), "validation_snapshots": int(len(validation))})
    winner = min(scores, key=lambda row: (row["cost_mae"] if "cost_mae" in row else row["delay_mae"], row["RMSE"], CANDIDATE_HALF_LIVES.index(row["half_life"])))
    return winner["half_life"], scores


def _weight_diagnostics(frame: pd.DataFrame, training_end: int, half_life: int | None) -> dict[str, Any]:
    weighted, projects = apply_project_recency_weights(frame, training_end, half_life)
    values = projects.to_numpy(dtype=float)
    grouped = frame.assign(_weight=frame["canonical_project_id"].astype(str).map(projects)).groupby("completion_year")["_weight"].first()
    effective = float(values.sum() ** 2 / np.square(values).sum()) if np.square(values).sum() else None
    examples = {str(int(training_end - age)): round(float(0.5 ** (age / half_life)), 6) if half_life else 1.0 for age in (0, 5, 10, 15) if training_end - age >= int(frame.completion_year.min())}
    return {"selected_half_life": half_life, "minimum_project_weight": round(float(values.min()), 6), "maximum_project_weight": round(float(values.max()), 6), "median_project_weight": round(float(np.median(values)), 6), "mean_project_weight": round(float(values.mean()), 6), "effective_weighted_project_sample_size": round(effective, 4) if effective is not None else None, "total_projects": int(len(projects)), "project_weight_distribution_by_completion_year": {str(int(k)): round(float(v), 6) for k, v in grouped.items()}, "formula_examples": examples, "snapshot_weight_sum_policy": "one normalized project weight divided by retained training snapshot count"}


def _metric(frame: pd.DataFrame, actual: str, prediction: str) -> dict:
    return _regression_metrics(frame[actual], frame[prediction].to_numpy(float), frame.sample_weight, frame.canonical_project_id)


def _window_verdict(cost_improvement: float | None, delay_improvement: float | None) -> str:
    if cost_improvement is not None and delay_improvement is not None and cost_improvement > 0 and delay_improvement > 0:
        return "PROMOTION CANDIDATE"
    if cost_improvement is not None and delay_improvement is not None and cost_improvement <= 0 and delay_improvement <= 0:
        return "REGRESSION / DO NOT PROMOTE"
    return "MIXED / NEEDS REVIEW"


def fit_against_production(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None):
    frozen = enrich_supervised_for_production(data.copy())
    frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce")
    frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    base_train, base_test = temporal_project_split(frozen, int(training_start), int(training_end), int(test_end))
    contract = target_feature_contract(production_bundle.get("metadata") or {})
    contract = {key: list(value or production_receipt.get("features_used") or []) for key, value in contract.items()}
    if any(not features for features in contract.values()):
        raise ValueError("Production target feature contract is unavailable.")
    algorithms = {target: _algorithm(production_bundle, production_receipt, target) for target in ("cost", "delay")}
    selected_half_lives = {}
    candidate_validation = {}
    for target, target_column in (("cost", "actual_cost_overrun_percentage"), ("delay", "actual_delay_days")):
        selected_half_lives[target], candidate_validation[target] = _candidate_validation(base_train, contract[target], target_column, algorithms[target], int(training_end), RANDOM_SEEDS[target])
    models = {
        "cost": _fit_weighted(base_train, contract["cost"], "actual_cost_overrun_percentage", algorithms["cost"], int(training_end), selected_half_lives["cost"], RANDOM_SEEDS["cost"]),
        "delay": _fit_weighted(base_train, contract["delay"], "actual_delay_days", algorithms["delay"], int(training_end), selected_half_lives["delay"], RANDOM_SEEDS["delay"]),
    }
    test = assign_project_balanced_weights(base_test)
    test["production_cost"] = production_bundle["cost"].predict(test[contract["cost"]])
    test["experiment_cost"] = models["cost"].predict(test[contract["cost"]])
    test["production_delay"] = np.maximum(0, production_bundle["delay"].predict(test[contract["delay"]]))
    test["experiment_delay"] = np.maximum(0, models["delay"].predict(test[contract["delay"]]))
    production_cost, experiment_cost = _metric(test, "actual_cost_overrun_percentage", "production_cost"), _metric(test, "actual_cost_overrun_percentage", "experiment_cost")
    production_delay, experiment_delay = _metric(test, "actual_delay_days", "production_delay"), _metric(test, "actual_delay_days", "experiment_delay")
    paired_cost = paired_project_mae_comparison(test, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost", candidate_prediction="experiment_cost")
    paired_delay = paired_project_mae_comparison(test, actual="actual_delay_days", baseline_prediction="production_delay", candidate_prediction="experiment_delay", seed=26104)
    cost_improvement = (production_cost["MAE"] - experiment_cost["MAE"]) / production_cost["MAE"] * 100 if production_cost["MAE"] else None
    delay_improvement = (production_delay["MAE"] - experiment_delay["MAE"]) / production_delay["MAE"] * 100 if production_delay["MAE"] else None
    overall = {"production_cost_mae": production_cost["MAE"], "experiment_cost_mae": experiment_cost["MAE"], "absolute_cost_improvement_pp": round(production_cost["MAE"] - experiment_cost["MAE"], 4), "cost_improvement_percentage": round(cost_improvement, 4) if cost_improvement is not None else None, "improvement_percentage": round(cost_improvement, 4) if cost_improvement is not None else None, "production_delay_mae": production_delay["MAE"], "experiment_delay_mae": experiment_delay["MAE"], "absolute_delay_improvement_days": round(production_delay["MAE"] - experiment_delay["MAE"], 4), "delay_improvement_percentage": round(delay_improvement, 4) if delay_improvement is not None else None, "comparison_test_projects": int(test.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(test)), "paired_project_cost_comparison": paired_cost, "paired_project_delay_comparison": paired_delay, "paired_project_comparison": paired_cost, "selected_cost_half_life": selected_half_lives["cost"], "selected_delay_half_life": selected_half_lives["delay"], "candidate_internal_validation": candidate_validation, "age_weight_diagnostics": {"cost": _weight_diagnostics(base_train, int(training_end), selected_half_lives["cost"]), "delay": _weight_diagnostics(base_train, int(training_end), selected_half_lives["delay"])}, "verdict": _window_verdict(cost_improvement, delay_improvement)}
    all_features = list(dict.fromkeys(contract["cost"] + contract["delay"] + contract["risk"]))
    context = build_experiment_context(experiment_id=EXPERIMENT_ID, full_data=frozen, train=base_train, test=base_test, features=all_features, training_start=training_start, training_end=training_end, testing_end=test_end, weighting_policy="project-balanced snapshots with normalized project-level exponential recency influence")
    manifest = new_experiment_manifest(context=context, name=EXPERIMENT_NAME, changed_dimension="weighting", hypothesis="More recent completed projects may better represent future cost and delay relationships while retaining older projects.")
    manifest.update({"scope": EXPERIMENT_SCOPE, "production_run_id": production_receipt.get("run_id"), "selected_algorithms": algorithms, "candidate_half_lives": list(CANDIDATE_HALF_LIVES), "selected_half_lives": selected_half_lives, "candidate_internal_validation": candidate_validation, "random_seeds": RANDOM_SEEDS, "controlled_variables": ["raw PAIMANA data", "processed lifecycle dataset", "project identities", "features", "targets", "algorithms", "model selection", "missing-value handling", "temporal holdout", "snapshot sampling", "test weighting", "evaluation metrics"], "leakage_policy": "Weights use only each training project's completion year and the training cutoff; final outcomes and future holdout rows never influence weights or half-life selection.", "evaluation_weighting_policy": "Production and experiment use identical project-balanced test weights.", "weighting_formula": "0.5 ** ((training_end_year - project_completion_year) / half_life), normalized to mean one, then divided by retained snapshots per project."})
    run_dir = experiment_run_directory(EXPERIMENT_ID, context.window, manifest["run_id"]); run_dir.mkdir(parents=True, exist_ok=False)
    for target, model in models.items():
        joblib.dump(model, run_dir / f"{target}_model.pkl")
    (run_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n")
    (run_dir / "evaluation_results.json").write_text(json.dumps(_json_safe(overall), indent=2, allow_nan=False) + "\n")
    record_experiment({"experiment_id": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "run_id": manifest["run_id"], "status": "COMPLETED", "decision": "PENDING", "model_role": "experiment", "promotion_allowed": False, "scope": EXPERIMENT_SCOPE, "window": context.window, "created_at": manifest["created_at"], "production_run_id": production_receipt.get("run_id"), "cost_improvement_percentage": overall["cost_improvement_percentage"], "delay_improvement_percentage": overall["delay_improvement_percentage"], "verdict": overall["verdict"]})
    comparable = {(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()) for _, row in test.iterrows()}
    experiment = {"experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "run_id": manifest["run_id"], "model_role": "experiment", "scope": EXPERIMENT_SCOPE, "decision": "PENDING", "promotion_allowed": False, "selected_cost_half_life": selected_half_lives["cost"], "selected_delay_half_life": selected_half_lives["delay"], "selected_algorithms": algorithms, "metrics": {"cost": experiment_cost, "delay": experiment_delay}, "candidate_half_lives": list(CANDIDATE_HALF_LIVES), "age_weight_diagnostics": overall["age_weight_diagnostics"], "verdict": overall["verdict"], "leakage_policy": manifest["leakage_policy"]}
    return {"experiment": experiment, "overall_comparison": overall, "runtime_state": {"models": models, "features": contract, "comparable": comparable}}


def _key(row: pd.Series):
    date = pd.to_datetime(row.get("snapshot_date"), errors="coerce")
    project = row.get("canonical_project_id")
    return None if pd.isna(project) or pd.isna(date) else (str(project), pd.Timestamp(date).isoformat())


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    cost_features = state["features"]["cost"]
    delay_features = state["features"]["delay"]
    cost = float(state["models"]["cost"].predict(row.to_frame().T[cost_features])[0])
    delay = max(0.0, float(state["models"]["delay"].predict(row.to_frame().T[delay_features])[0]))
    return {"predicted_cost_overrun": round(cost, 4), "predicted_delay_days": round(delay, 4), "selected_cost_half_life": state.get("selected_cost_half_life"), "selected_delay_half_life": state.get("selected_delay_half_life")}
