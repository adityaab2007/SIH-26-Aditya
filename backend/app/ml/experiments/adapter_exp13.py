"""Experiment 13: recency-weighted project training on current production.

Current production Delay is a compound Exp61 AFT/prior/calibration/fallback model,
not a single sklearn estimator. Replacing that compound architecture with a
simple tree merely to apply sample weights would change more than the experiment
hypothesis. Therefore Exp13 applies recency weighting to the reconstructible Cost
estimator and keeps current production Delay as an exact control. Both metrics
are still reported on the identical temporal cohort.
"""
from __future__ import annotations

import json
import math
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import build_experiment_context, experiment_run_directory, new_experiment_manifest
from backend.app.ml.experiments.nextgen_common import _compare as production_comparison_cohort
from backend.app.ml.experiments.nextgen_common import _prepare as prepare_current_production_frame
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import target_feature_contract

EXPERIMENT_ID = "exp_13"
EXPERIMENT_SEQUENCE = 13
EXPERIMENT_NAME = "Recency-Weighted Project Training"
EXPERIMENT_SCOPE = "cost_delay"
CANDIDATE_HALF_LIVES: tuple[int | None, ...] = (None, 15, 10, 7, 5, 3)
RANDOM_SEEDS = {"cost": 26203, "delay": 26203}


def _json_safe(value: Any):
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)): value = value.item()
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _normalize_feature_missing(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Convert literal pd.NA values in model features to sklearn-safe np.nan."""
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


def _unwrap_pipeline(model):
    current = model; seen: set[int] = set()
    while not hasattr(current, "named_steps"):
        marker = id(current)
        if marker in seen: raise ValueError("Cycle while unwrapping production model.")
        seen.add(marker)
        next_model = None
        for attribute in ("model", "base_model", "estimator", "pipeline"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                next_model = candidate; break
        if next_model is None: raise ValueError(f"Unable to unwrap production model {type(current).__name__}.")
        current = next_model
    return current


def _algorithm(bundle: dict, receipt: dict, target: str) -> str:
    if target != "cost":
        raise ValueError("Current Exp13 reconstructs only the Cost estimator; Delay is a production control.")
    name = ((bundle.get("metadata") or {}).get("selected_algorithms") or {}).get(target)
    name = name or (receipt.get("selected_algorithms") or {}).get(target)
    if name in _regressors(1): return name
    model = _unwrap_pipeline(bundle[target]).named_steps["model"]
    lowered = type(model).__name__.lower()
    if "lgbm" in lowered or "lightgbm" in lowered: return "lightgbm"
    if "xgb" in lowered or "xgboost" in lowered: return "xgboost"
    if "extra" in lowered: return "extra_trees"
    raise ValueError(f"Cannot identify production {target} algorithm.")


def compute_project_recency_weights(frame: pd.DataFrame, training_end_year: int, half_life: int | None) -> pd.Series:
    required = {"canonical_project_id", "completion_year"}; missing = required - set(frame.columns)
    if missing: raise ValueError("recency weighting missing: " + ", ".join(sorted(missing)))
    projects = frame[["canonical_project_id", "completion_year"]].dropna().drop_duplicates("canonical_project_id")
    years = pd.to_numeric(projects["completion_year"], errors="coerce")
    if years.isna().any() or (years > int(training_end_year)).any():
        raise ValueError("recency weights require finite training-side completion years at or before the cutoff")
    if half_life is None:
        raw = pd.Series(1.0, index=projects["canonical_project_id"].astype(str))
    else:
        half_life = float(half_life)
        if not math.isfinite(half_life) or half_life <= 0: raise ValueError("half_life must be positive or None for the control")
        raw = pd.Series(np.power(0.5, (int(training_end_year) - years.to_numpy(dtype=float)) / half_life), index=projects["canonical_project_id"].astype(str))
    if not np.isfinite(raw.to_numpy(dtype=float)).all() or (raw <= 0).any(): raise ValueError("recency weights must be finite and positive")
    return raw / float(raw.mean())


def apply_project_recency_weights(frame: pd.DataFrame, training_end_year: int, half_life: int | None) -> tuple[pd.DataFrame, pd.Series]:
    result = frame.copy(); project_weights = compute_project_recency_weights(result, training_end_year, half_life)
    keys = result["canonical_project_id"].astype(str); counts = keys.value_counts()
    result["sample_weight"] = keys.map(project_weights).to_numpy(dtype=float) / keys.map(counts).to_numpy(dtype=float)
    if not np.isfinite(result["sample_weight"].to_numpy(dtype=float)).all() or (result["sample_weight"] <= 0).any(): raise ValueError("snapshot recency weights must be finite and positive")
    observed = result.groupby(keys, sort=False)["sample_weight"].sum(); expected = project_weights.loc[observed.index]
    if not np.allclose(observed.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-10): raise AssertionError("snapshot weights do not sum to the normalized project weight")
    return result, project_weights


recency_project_weights = compute_project_recency_weights
recency_snapshot_weights = apply_project_recency_weights


def _fit_weighted(train: pd.DataFrame, features: list[str], target: str, algorithm: str, end_year: int, half_life: int | None, seed: int):
    weighted, _ = apply_project_recency_weights(train, end_year, half_life)
    weighted = _normalize_feature_missing(weighted, features)
    return _fit_pipeline(_regressors(seed)[algorithm], weighted, features, target)


def _candidate_validation(train: pd.DataFrame, features: list[str], target: str, algorithm: str, seed: int) -> tuple[int | None, list[dict[str, Any]]]:
    validation_year = int(pd.to_numeric(train.completion_year, errors="coerce").max())
    fitting = train[train.completion_year.lt(validation_year)].copy(); validation = train[train.completion_year.eq(validation_year)].copy()
    if fitting.canonical_project_id.nunique() < 5 or validation.canonical_project_id.nunique() < 2: raise ValueError(f"Insufficient internal temporal validation data for {target}.")
    validation = assign_project_balanced_weights(validation); scores = []
    for half_life in CANDIDATE_HALF_LIVES:
        model = _fit_weighted(fitting, features, target, algorithm, validation_year - 1, half_life, seed)
        predicted = model.predict(_X(validation, features))
        metrics = _regression_metrics(validation[target], predicted, validation.sample_weight, validation.canonical_project_id)
        scores.append({"half_life": half_life, "cost_mae": metrics["MAE"], "RMSE": metrics["RMSE"], "validation_year": validation_year, "validation_projects": int(validation.canonical_project_id.nunique()), "validation_snapshots": int(len(validation))})
    winner = min(scores, key=lambda row: (row["cost_mae"], row["RMSE"], CANDIDATE_HALF_LIVES.index(row["half_life"])))
    return winner["half_life"], scores


def _weight_diagnostics(frame: pd.DataFrame, training_end: int, half_life: int | None) -> dict[str, Any]:
    _, projects = apply_project_recency_weights(frame, training_end, half_life); values = projects.to_numpy(dtype=float); denominator = np.square(values).sum(); effective = float(values.sum() ** 2 / denominator) if denominator else None
    return {"selected_half_life": half_life, "minimum_project_weight": round(float(values.min()),6), "maximum_project_weight": round(float(values.max()),6), "median_project_weight": round(float(np.median(values)),6), "mean_project_weight": round(float(values.mean()),6), "effective_weighted_project_sample_size": round(effective,4) if effective is not None else None, "total_projects": int(len(projects))}


def _metric(frame: pd.DataFrame, actual: str, prediction: str) -> dict:
    return _regression_metrics(frame[actual], frame[prediction].to_numpy(float), frame.sample_weight, frame.canonical_project_id)


def _window_verdict(cost_improvement: float | None, delay_improvement: float | None) -> str:
    if cost_improvement is None or delay_improvement is None: return "INVALID / INCOMPLETE"
    if cost_improvement > 0 and delay_improvement >= -1e-12: return "PROMOTION CANDIDATE"
    if cost_improvement <= 0 and delay_improvement <= 1e-12: return "REGRESSION / DO NOT PROMOTE"
    return "MIXED / NEEDS REVIEW"


def fit_against_production(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None):
    # Use the exact current-production enrichment chain (including Exp34 causal
    # path features) instead of the older base-only supervised preparation.
    frozen = prepare_current_production_frame(data)
    base_train, base_test = temporal_project_split(frozen, int(training_start), int(training_end), int(test_end))
    contract = {k: list(v or production_receipt.get("features_used") or []) for k, v in target_feature_contract(production_bundle.get("metadata") or {}).items()}
    if any(not features for features in contract.values()): raise ValueError("Production target feature contract is unavailable.")

    cost_algorithm = _algorithm(production_bundle, production_receipt, "cost")
    cost_half_life, cost_validation = _candidate_validation(base_train, contract["cost"], "actual_cost_overrun_percentage", cost_algorithm, RANDOM_SEEDS["cost"])
    cost_model = _fit_weighted(base_train, contract["cost"], "actual_cost_overrun_percentage", cost_algorithm, int(training_end), cost_half_life, RANDOM_SEEDS["cost"])

    # Reuse the canonical current-production comparison cohort. This applies the
    # production Cost cohort filter, deterministic Exp35 calibration gate, and
    # project-balanced test weights before invoking the Exp61 Delay wrapper.
    test = production_comparison_cohort(base_test)
    test["production_cost"] = np.asarray(production_bundle["cost"].predict(_X(test, contract["cost"])), dtype=float)
    test["experiment_cost"] = np.asarray(cost_model.predict(_X(test, contract["cost"])), dtype=float)
    test["production_delay"] = np.maximum(0.0, np.asarray(production_bundle["delay"].predict(_X(test, contract["delay"])), dtype=float))
    test["experiment_delay"] = test["production_delay"].to_numpy(dtype=float).copy()

    production_cost = _metric(test, "actual_cost_overrun_percentage", "production_cost"); experiment_cost = _metric(test, "actual_cost_overrun_percentage", "experiment_cost")
    production_delay = _metric(test, "actual_delay_days", "production_delay"); experiment_delay = _metric(test, "actual_delay_days", "experiment_delay")
    cost_improvement = (production_cost["MAE"] - experiment_cost["MAE"]) / production_cost["MAE"] * 100 if production_cost["MAE"] else None
    delay_improvement = (production_delay["MAE"] - experiment_delay["MAE"]) / production_delay["MAE"] * 100 if production_delay["MAE"] else 0.0
    paired_cost = paired_project_mae_comparison(test, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost", candidate_prediction="experiment_cost")
    paired_delay = paired_project_mae_comparison(test, actual="actual_delay_days", baseline_prediction="production_delay", candidate_prediction="experiment_delay", seed=26104)

    overall = {"production_cost_mae": production_cost["MAE"], "experiment_cost_mae": experiment_cost["MAE"], "absolute_cost_improvement_pp": round(production_cost["MAE"]-experiment_cost["MAE"],4), "cost_improvement_percentage": round(cost_improvement,4) if cost_improvement is not None else None, "production_delay_mae": production_delay["MAE"], "experiment_delay_mae": experiment_delay["MAE"], "absolute_delay_improvement_days": round(production_delay["MAE"]-experiment_delay["MAE"],4), "delay_improvement_percentage": round(delay_improvement,4) if delay_improvement is not None else None, "comparison_test_projects": int(test.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(test)), "paired_project_cost_comparison": paired_cost, "paired_project_delay_comparison": paired_delay, "selected_cost_half_life": cost_half_life, "selected_delay_half_life": None, "candidate_internal_validation": {"cost": cost_validation, "delay": [{"mode":"production_control","reason":"current production Delay is a compound Exp61 wrapper"}]}, "age_weight_diagnostics": {"cost": _weight_diagnostics(base_train, int(training_end), cost_half_life), "delay": {"selected_half_life": None, "mode":"production_control"}}, "delay_predictions_identical_to_production": bool(np.array_equal(test["production_delay"].to_numpy(), test["experiment_delay"].to_numpy())), "verdict": _window_verdict(cost_improvement, delay_improvement)}

    all_features = list(dict.fromkeys(contract["cost"] + contract["delay"] + contract["risk"]))
    context = build_experiment_context(experiment_id=EXPERIMENT_ID, full_data=frozen, train=base_train, test=test, features=all_features, training_start=training_start, training_end=training_end, testing_end=test_end, weighting_policy="Cost: normalized project-level exponential recency influence; Delay: exact production control")
    manifest = new_experiment_manifest(context=context, name=EXPERIMENT_NAME, changed_dimension="weighting", hypothesis="More recent completed projects may better represent future Cost relationships while current compound production Delay remains fixed as a control.")
    manifest.update({"scope":EXPERIMENT_SCOPE,"production_run_id":production_receipt.get("run_id"),"selected_algorithms":{"cost":cost_algorithm,"delay":"production_control"},"candidate_half_lives":list(CANDIDATE_HALF_LIVES),"selected_half_lives":{"cost":cost_half_life,"delay":None},"candidate_internal_validation":overall["candidate_internal_validation"],"random_seeds":RANDOM_SEEDS,"delay_mode":"production_control_current_compound_wrapper","leakage_policy":"Cost weights use only training completion year and cutoff; Delay is copied from fresh production; future holdout never influences selection.","evaluation_weighting_policy":"Production and experiment use identical project-balanced test weights on the canonical current-production comparison cohort."})
    run_dir = experiment_run_directory(EXPERIMENT_ID, context.window, manifest["run_id"]); run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(cost_model, run_dir / "cost_model.pkl"); (run_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, allow_nan=False)+"\n"); (run_dir / "evaluation_results.json").write_text(json.dumps(_json_safe(overall), indent=2, allow_nan=False)+"\n")
    record_experiment({"experiment_id":EXPERIMENT_ID,"name":EXPERIMENT_NAME,"run_id":manifest["run_id"],"status":"COMPLETED","decision":"PENDING","model_role":"experiment","promotion_allowed":False,"scope":EXPERIMENT_SCOPE,"window":context.window,"created_at":manifest["created_at"],"production_run_id":production_receipt.get("run_id"),"cost_improvement_percentage":overall["cost_improvement_percentage"],"delay_improvement_percentage":overall["delay_improvement_percentage"],"verdict":overall["verdict"]})
    comparable = {(str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()) for _, row in test.iterrows()}
    experiment = {"experiment_id":EXPERIMENT_ID,"experiment_name":EXPERIMENT_NAME,"run_id":manifest["run_id"],"model_role":"experiment","scope":EXPERIMENT_SCOPE,"decision":"PENDING","promotion_allowed":False,"selected_cost_half_life":cost_half_life,"selected_delay_half_life":None,"selected_algorithms":{"cost":cost_algorithm,"delay":"production_control"},"metrics":{"cost":experiment_cost,"delay":experiment_delay},"delay_mode":"production_control_current_compound_wrapper","verdict":overall["verdict"],"leakage_policy":manifest["leakage_policy"]}
    return {"experiment":experiment,"overall_comparison":overall,"runtime_state":{"cost_model":cost_model,"production_delay_model":production_bundle["delay"],"features":contract,"comparable":comparable,"selected_cost_half_life":cost_half_life,"selected_delay_half_life":None,"delay_mode":"production_control"}}


def _key(row: pd.Series):
    date=pd.to_datetime(row.get("snapshot_date"),errors="coerce"); project=row.get("canonical_project_id")
    if pd.isna(project) or pd.isna(date): return None
    return str(project),pd.Timestamp(date).isoformat()


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"],axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    cost_features=state["features"]["cost"]; delay_features=state["features"]["delay"]; one=row.to_frame().T
    cost=float(state["cost_model"].predict(_X(one, cost_features))[0]); delay=max(0.0,float(state["production_delay_model"].predict(_X(one, delay_features))[0]))
    return {"predicted_cost_overrun":round(cost,4),"predicted_delay_days":round(delay,4),"selected_cost_half_life":state.get("selected_cost_half_life"),"selected_delay_half_life":None,"delay_mode":"production_control"}
