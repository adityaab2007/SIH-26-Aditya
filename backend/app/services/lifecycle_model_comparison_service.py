"""Retrain production and the latest isolated experiment on one frozen lifecycle dataset.

The comparison flow never promotes an experiment. It retrains the production
model first, fits the selected experiment against the same prepared PAIMANA
frame, and opens a judge session that can score one held-out project through
both models before revealing the single official outcome.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import (
    build_experiment_context,
    experiment_run_directory,
    new_experiment_manifest,
)
from backend.app.ml.experiments.registry import decision_from_improvement, record_experiment
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics
from backend.app.ml.residual_overrun_experiment import (
    CURRENT_OVERRUN,
    FINAL_TARGET,
    RESIDUAL_TARGET,
    _stage_cost_metrics,
    prepare_common_cost_cohort,
    reconstruct_final_overrun,
)
from backend.app.services import lifecycle_retraining_service as retraining
from backend.app.services import lifecycle_simulation_service as simulation

LATEST_EXPERIMENT_ID = "exp_03"
LATEST_EXPERIMENT_NAME = "Remaining-overrun forecasting"
_COMPARISON_SESSIONS: dict[str, dict] = {}
_MAX_COMPARISON_SESSIONS = 20


def _float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fit_exp03_against_production(
    *,
    data: pd.DataFrame,
    training_start: int,
    training_end: int,
    test_end: int,
    production_bundle: dict,
    production_receipt: dict,
) -> tuple[dict, object]:
    """Fit Exp 3 while using the exact fresh production model as its baseline."""
    train, test = prepare_common_cost_cohort(data, training_start, training_end, test_end)
    metadata = production_bundle["metadata"]
    features = list(metadata.get("features_used") or production_receipt.get("features_used") or [])
    if not features:
        raise ValueError("Fresh production run did not publish a feature contract.")
    missing_features = [name for name in features if name not in train.columns or name not in test.columns]
    if missing_features:
        raise ValueError("Experiment 3 cannot reuse the production feature contract: " + ", ".join(missing_features))

    algorithm = (metadata.get("selected_algorithms") or {}).get("cost") or (production_receipt.get("selected_algorithms") or {}).get("cost")
    production_cost_model = production_bundle["cost"]
    if not hasattr(production_cost_model, "named_steps") or "model" not in production_cost_model.named_steps:
        raise ValueError("Fresh production cost artifact does not expose the fitted estimator contract.")

    # Clone the exact estimator configuration selected by the production retrain.
    # Only the target and retained common cohort differ for Experiment 3.
    residual_estimator = clone(production_cost_model.named_steps["model"])
    residual_model = _fit_pipeline(residual_estimator, train, features, RESIDUAL_TARGET)
    production_final = np.asarray(production_cost_model.predict(test[features]), dtype=float)
    residual_remaining = np.asarray(residual_model.predict(test[features]), dtype=float)
    experiment_final = reconstruct_final_overrun(test[CURRENT_OVERRUN], residual_remaining)

    production_metrics = _regression_metrics(test[FINAL_TARGET], production_final, test.sample_weight, test.canonical_project_id)
    experiment_metrics = _regression_metrics(test[FINAL_TARGET], experiment_final, test.sample_weight, test.canonical_project_id)
    residual_metrics = _regression_metrics(test[RESIDUAL_TARGET], residual_remaining, test.sample_weight, test.canonical_project_id)
    production_mae = production_metrics.get("MAE")
    experiment_mae = experiment_metrics.get("MAE")
    improvement = None
    if production_mae not in (None, 0) and experiment_mae is not None:
        improvement = round((float(production_mae) - float(experiment_mae)) / float(production_mae) * 100.0, 3)

    rows = test[[
        "canonical_project_id", "project_name", "snapshot_date", "completion_year", "lifecycle_stage",
        CURRENT_OVERRUN, FINAL_TARGET, RESIDUAL_TARGET, "sample_weight",
    ]].copy()
    rows["production_predicted_final_overrun"] = production_final
    rows["experiment_predicted_remaining_overrun"] = residual_remaining
    rows["experiment_predicted_final_overrun"] = experiment_final
    rows["production_error"] = rows.production_predicted_final_overrun - rows[FINAL_TARGET]
    rows["experiment_error"] = rows.experiment_predicted_final_overrun - rows[FINAL_TARGET]

    context = build_experiment_context(
        experiment_id=LATEST_EXPERIMENT_ID,
        full_data=data,
        train=train,
        test=test,
        features=features,
        training_start=training_start,
        training_end=training_end,
        testing_end=test_end,
        weighting_policy="per-project weights renormalized after common-cohort filtering",
        baseline_name=f"production:{production_receipt.get('run_id')}",
    )
    production_dataset_fingerprint = production_receipt.get("dataset_fingerprint") or metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint")
    if not production_dataset_fingerprint or production_dataset_fingerprint != context.dataset_fingerprint:
        raise RuntimeError("Refusing comparison because production and experiment were not built from the same prepared dataset fingerprint.")
    production_feature_fingerprint = (metadata.get("provenance") or {}).get("feature_schema_fingerprint")
    if production_feature_fingerprint and production_feature_fingerprint != context.feature_schema_fingerprint:
        raise RuntimeError("Refusing comparison because Experiment 3 does not match the fresh production feature schema.")

    manifest = new_experiment_manifest(
        context=context,
        name="remaining_overrun_target",
        changed_dimension="cost_target",
        hypothesis="Predict remaining cost deterioration and reconstruct final overrun instead of predicting final overrun directly.",
    )
    decision = decision_from_improvement(improvement, 10.0)
    manifest.update({
        "decision": decision,
        "production_run_id": production_receipt.get("run_id"),
        "production_model_version": production_receipt.get("model_version"),
        "comparison_mode": "fresh_production_vs_experiment_same_dataset",
        "production_estimator_parameters_reused": True,
    })

    paired = paired_project_mae_comparison(
        rows,
        actual=FINAL_TARGET,
        baseline_prediction="production_predicted_final_overrun",
        candidate_prediction="experiment_predicted_final_overrun",
    )
    report = {
        "experiment": "experiment_3_residual_remaining_overrun",
        "experiment_id": LATEST_EXPERIMENT_ID,
        "experiment_name": LATEST_EXPERIMENT_NAME,
        "model_role": "experiment",
        "run_id": manifest["run_id"],
        "production_run_id": production_receipt.get("run_id"),
        "status": "COMPLETED",
        "decision": decision,
        "promotion_allowed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_period": [training_start, training_end],
        "testing_period": [training_end + 1, test_end],
        "experiment_scope": "cost_only",
        "features_used": features,
        "feature_count": len(features),
        "selected_algorithm": algorithm,
        "comparison_control": {
            "same_prepared_dataset": True,
            "same_training_window": True,
            "same_test_window": True,
            "same_features": True,
            "same_cost_estimator_parameters": True,
            "production_model_is_actual_fresh_retrain": True,
            "dataset_fingerprint": context.dataset_fingerprint,
            "training_fingerprint": context.training_fingerprint,
            "test_fingerprint": context.test_fingerprint,
            "feature_schema_fingerprint": context.feature_schema_fingerprint,
            "weighting_policy": context.weighting_policy,
            "training_rows": int(len(train)),
            "training_projects": int(train.canonical_project_id.nunique()),
            "test_rows": int(len(test)),
            "test_projects": int(test.canonical_project_id.nunique()),
        },
        "production_final_overrun_metrics": production_metrics,
        "experiment_reconstructed_final_overrun_metrics": experiment_metrics,
        "experiment_residual_target_metrics": residual_metrics,
        "final_mae_improvement_percentage": improvement,
        "absolute_mae_improvement_pp": round(float(production_mae) - float(experiment_mae), 4) if production_mae is not None and experiment_mae is not None else None,
        "success_threshold": {"metric": "final-overrun MAE reduction", "minimum_percentage": 10.0, "passed": improvement is not None and improvement >= 10.0},
        "paired_project_comparison": paired,
        "production_lifecycle_stage_metrics": _stage_cost_metrics(rows, "production_predicted_final_overrun"),
        "experiment_lifecycle_stage_metrics": _stage_cost_metrics(rows, "experiment_predicted_final_overrun"),
        "target_definition": {
            "production_target": FINAL_TARGET,
            "experiment_target": f"{FINAL_TARGET} - {CURRENT_OVERRUN}",
            "experiment_final_reconstruction": f"{CURRENT_OVERRUN} + predicted_{RESIDUAL_TARGET}",
        },
    }

    destination = experiment_run_directory(LATEST_EXPERIMENT_ID, context.window, manifest["run_id"])
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump(residual_model, destination / "cost_model.pkl")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (destination / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    rows.to_csv(destination / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
    record_experiment({
        "experiment_id": LATEST_EXPERIMENT_ID,
        "name": "remaining_overrun_target",
        "model_role": "experiment",
        "run_id": manifest["run_id"],
        "production_run_id": production_receipt.get("run_id"),
        "status": "COMPLETED",
        "decision": decision,
        "promotion_allowed": False,
        "changed_dimension": "cost_target",
        "training_period": report["training_period"],
        "testing_period": report["testing_period"],
        "baseline_final_cost_mae": production_mae,
        "candidate_final_cost_mae": experiment_mae,
        "improvement_percentage": improvement,
        "dataset_fingerprint": context.dataset_fingerprint,
        "training_fingerprint": context.training_fingerprint,
        "test_fingerprint": context.test_fingerprint,
        "feature_schema_fingerprint": context.feature_schema_fingerprint,
        "created_at": report["generated_at"],
    })
    report["artifact_directory"] = str(destination)
    return report, residual_model


def _open_production_session_from_frozen_data(
    *,
    data: pd.DataFrame,
    start_year: int,
    end_year: int,
    production_bundle: dict,
    production_receipt: dict,
) -> dict:
    """Open the judge session from the exact dataframe used by retrain/compare."""
    frame = data.copy()
    frame["completion_year"] = pd.to_numeric(frame["completion_year"], errors="coerce")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    if "completion_date" in frame:
        frame["completion_date"] = pd.to_datetime(frame["completion_date"], errors="coerce")

    metadata = production_bundle["metadata"]
    features = list(metadata.get("features_used") or production_receipt.get("features_used") or [])
    train_projects = set(frame.loc[frame.completion_year.between(start_year, end_year), "canonical_project_id"].dropna())
    held_all = frame[frame.completion_year.gt(end_year)].copy()
    overlap = train_projects & set(held_all.canonical_project_id.dropna())
    if overlap:
        raise ValueError(f"Project-group leakage across comparison judge split: {len(overlap)} project(s)")
    if held_all.empty:
        raise ValueError("No later lifecycle projects are available for a leakage-free comparison test.")

    held_all = held_all.sort_values(["canonical_project_id", "snapshot_date"])
    held_latest = held_all.drop_duplicates("canonical_project_id", keep="last").reset_index(drop=True)
    held_latest["record_index"] = np.arange(len(held_latest), dtype=int)
    history_counts = held_all.groupby("canonical_project_id").size().to_dict()

    session_id = uuid.uuid4().hex[:16]
    simulation._CUSTOM_SESSIONS[session_id] = {
        "training_start": start_year,
        "training_end": end_year,
        "run_id": production_receipt.get("run_id"),
        "dataset_fingerprint": production_receipt.get("dataset_fingerprint"),
        "features": features,
        "models": production_bundle,
        "held_out": held_latest,
        "history_counts": history_counts,
        "predictions": {},
    }
    while len(simulation._CUSTOM_SESSIONS) > simulation._MAX_CUSTOM_SESSIONS:
        simulation._CUSTOM_SESSIONS.pop(next(iter(simulation._CUSTOM_SESSIONS)), None)

    return {
        "session_id": session_id,
        "run_id": production_receipt.get("run_id"),
        "dataset_fingerprint": production_receipt.get("dataset_fingerprint"),
        "model_version": production_receipt.get("model_version"),
        "training_start": start_year,
        "training_end": end_year,
        "leakage_guard": f"Both comparison models use one frozen PAIMANA frame; only projects completed in {start_year}-{end_year} can contribute to fitting and all offered judge projects complete after {end_year}.",
        "actual_outcomes_sent_to_browser": False,
    }


def retrain_and_compare(start_year: int, end_year: int, experiment_id: str = LATEST_EXPERIMENT_ID) -> dict:
    """Freshly retrain production and the latest experiment, then open one judge session."""
    if experiment_id != LATEST_EXPERIMENT_ID:
        raise ValueError(f"Unsupported comparison experiment '{experiment_id}'. Latest is {LATEST_EXPERIMENT_ID}.")
    start_year, end_year = int(start_year), int(end_year)
    data, _identity, _min_year, max_year = retraining._training_data()

    production = retraining.retrain_lifecycle(start_year, end_year)
    production_bundle = simulation._artifact_bundle(start_year, end_year, production.get("run_id"))
    report, candidate_model = _fit_exp03_against_production(
        data=data,
        training_start=start_year,
        training_end=end_year,
        test_end=max_year,
        production_bundle=production_bundle,
        production_receipt=production,
    )
    production_session = _open_production_session_from_frozen_data(
        data=data,
        start_year=start_year,
        end_year=end_year,
        production_bundle=production_bundle,
        production_receipt=production,
    )
    underlying = simulation._session(production_session["session_id"])
    held = underlying["held_out"]
    comparable = held[pd.to_numeric(held[CURRENT_OVERRUN], errors="coerce").notna()].copy()
    if comparable.empty:
        raise ValueError("No held-out projects have current cost escalation required by Experiment 3.")
    comparable_indices = set(int(value) for value in comparable.record_index.tolist())
    counts = comparable.groupby("completion_year").size().sort_index()

    comparison_session_id = uuid.uuid4().hex[:16]
    _COMPARISON_SESSIONS[comparison_session_id] = {
        "production_session_id": production_session["session_id"],
        "production_run_id": production.get("run_id"),
        "dataset_fingerprint": production.get("dataset_fingerprint"),
        "experiment_id": LATEST_EXPERIMENT_ID,
        "experiment_run_id": report["run_id"],
        "experiment_model": candidate_model,
        "features": report["features_used"],
        "comparable_indices": comparable_indices,
        "overall": report,
        "candidate_predictions": {},
    }
    while len(_COMPARISON_SESSIONS) > _MAX_COMPARISON_SESSIONS:
        _COMPARISON_SESSIONS.pop(next(iter(_COMPARISON_SESSIONS)), None)

    session = {
        "session_id": comparison_session_id,
        "comparison_session_id": comparison_session_id,
        "production_session_id": production_session["session_id"],
        "run_id": production.get("run_id"),
        "production_run_id": production.get("run_id"),
        "experiment_id": LATEST_EXPERIMENT_ID,
        "experiment_name": LATEST_EXPERIMENT_NAME,
        "experiment_run_id": report["run_id"],
        "dataset_fingerprint": production.get("dataset_fingerprint"),
        "training_start": start_year,
        "training_end": end_year,
        "eligible_test_years": [{"year": int(year), "projects": int(count)} for year, count in counts.items()],
        "actual_outcomes_sent_to_browser": False,
        "leakage_guard": production_session.get("leakage_guard"),
    }
    return {
        "status": "success",
        "production": production,
        "experiment": {
            "experiment_id": report["experiment_id"],
            "experiment_name": report["experiment_name"],
            "run_id": report["run_id"],
            "scope": report["experiment_scope"],
            "selected_algorithm": report["selected_algorithm"],
            "decision": report["decision"],
            "promotion_allowed": False,
        },
        "overall_comparison": {
            "production_cost_mae": report["production_final_overrun_metrics"].get("MAE"),
            "experiment_cost_mae": report["experiment_reconstructed_final_overrun_metrics"].get("MAE"),
            "absolute_mae_improvement_pp": report.get("absolute_mae_improvement_pp"),
            "improvement_percentage": report.get("final_mae_improvement_percentage"),
            "candidate_better": report.get("final_mae_improvement_percentage") is not None and report["final_mae_improvement_percentage"] > 0,
            "success_threshold_passed": report["success_threshold"]["passed"],
            "paired_project_comparison": report["paired_project_comparison"],
            "production_stage_metrics": report["production_lifecycle_stage_metrics"],
            "experiment_stage_metrics": report["experiment_lifecycle_stage_metrics"],
            "comparison_test_projects": report["comparison_control"]["test_projects"],
            "comparison_test_snapshots": report["comparison_control"]["test_rows"],
        },
        "session": session,
    }


def _comparison_session(session_id: str) -> dict:
    if session_id not in _COMPARISON_SESSIONS:
        raise KeyError(session_id)
    return _COMPARISON_SESSIONS[session_id]


def comparison_projects(session_id: str, year: int) -> dict:
    session = _comparison_session(session_id)
    response = simulation.custom_projects(session["production_session_id"], int(year))
    response["items"] = [row for row in response["items"] if int(row["record_index"]) in session["comparable_indices"]]
    if not response["items"]:
        raise ValueError(f"No comparable production/Experiment 3 projects are available for {year}.")
    response.update({
        "session_id": session_id,
        "comparison_session_id": session_id,
        "experiment_id": session["experiment_id"],
        "experiment_run_id": session["experiment_run_id"],
        "note": "Only projects that can be scored by both the fresh production model and Experiment 3 are shown; actual outcomes remain hidden until reveal.",
    })
    return response


def predict_comparison(session_id: str, record_index: int) -> dict:
    session = _comparison_session(session_id)
    record_index = int(record_index)
    if record_index not in session["comparable_indices"]:
        raise ValueError("Selected project is not comparable under the active experiment contract.")
    production = simulation.predict_custom(session["production_session_id"], record_index)
    underlying = simulation._session(session["production_session_id"])
    row = simulation._session_row(underlying, record_index)
    features = session["features"]
    remaining = float(session["experiment_model"].predict(row.to_frame().T[features])[0])
    current = _float(row.get(CURRENT_OVERRUN))
    if current is None:
        raise ValueError("Selected project has no current cost escalation required by Experiment 3.")
    experiment_final = current + remaining
    experiment_payload = {
        "experiment_id": session["experiment_id"],
        "experiment_run_id": session["experiment_run_id"],
        "experiment_name": LATEST_EXPERIMENT_NAME,
        "predicted_remaining_cost_overrun": round(remaining, 4),
        "current_observed_cost_escalation": round(current, 4),
        "predicted_cost_overrun": round(experiment_final, 4),
        "scope": "cost_only",
    }
    session["candidate_predictions"][record_index] = experiment_payload
    production["comparison"] = {
        "production": {"predicted_cost_overrun": production["predicted_cost_overrun"], "run_id": session["production_run_id"]},
        "experiment": experiment_payload,
        "prediction_difference_pp": round(experiment_final - float(production["predicted_cost_overrun"]), 4),
        "actual_outcome_sent_to_browser": False,
    }
    production["comparison_session_id"] = session_id
    return production


def reveal_comparison(session_id: str, record_index: int) -> dict:
    session = _comparison_session(session_id)
    record_index = int(record_index)
    candidate = session["candidate_predictions"].get(record_index)
    if candidate is None:
        raise ValueError("Generate the production and experiment predictions before revealing the actual outcome.")
    actual = simulation.reveal_custom(session["production_session_id"], record_index)
    actual_cost = float(actual["actual_cost_overrun"])
    production_error = float(actual["cost_error_absolute_pp"])
    experiment_error = abs(float(candidate["predicted_cost_overrun"]) - actual_cost)
    improvement = None
    if production_error > 0:
        improvement = (production_error - experiment_error) / production_error * 100.0
    elif experiment_error == 0:
        improvement = 0.0
    actual["comparison"] = {
        "production_cost_error_absolute_pp": round(production_error, 4),
        "experiment_cost_error_absolute_pp": round(experiment_error, 4),
        "individual_error_improvement_percentage": round(improvement, 3) if improvement is not None else None,
        "experiment_better_for_project": experiment_error < production_error,
        "production_predicted_cost_overrun": underlying_production_prediction(session, record_index),
        "experiment_predicted_cost_overrun": candidate["predicted_cost_overrun"],
        "experiment_id": session["experiment_id"],
        "experiment_run_id": session["experiment_run_id"],
    }
    actual["comparison_session_id"] = session_id
    return actual


def underlying_production_prediction(session: dict, record_index: int) -> float:
    underlying = simulation._session(session["production_session_id"])
    prediction = underlying["predictions"].get(int(record_index)) or {}
    return float(prediction.get("predicted_cost_overrun"))
