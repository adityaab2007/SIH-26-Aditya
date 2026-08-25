"""Experiment 3 adapter for the generic production-vs-experiment harness."""
from __future__ import annotations

from datetime import datetime, timezone
import json

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

EXPERIMENT_ID = "exp_03"
EXPERIMENT_SEQUENCE = 3
EXPERIMENT_NAME = "Remaining-overrun forecasting"
EXPERIMENT_SCOPE = "cost_only"


def fit_against_production(
    *,
    data: pd.DataFrame,
    training_start: int,
    training_end: int,
    test_end: int,
    production_bundle: dict,
    production_receipt: dict,
) -> dict:
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
        experiment_id=EXPERIMENT_ID,
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
        raise RuntimeError("Refusing comparison because production and Experiment 3 were not built from the same prepared dataset fingerprint.")
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
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "model_role": "experiment",
        "run_id": manifest["run_id"],
        "production_run_id": production_receipt.get("run_id"),
        "status": "COMPLETED",
        "decision": decision,
        "promotion_allowed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_period": [training_start, training_end],
        "testing_period": [training_end + 1, test_end],
        "experiment_scope": EXPERIMENT_SCOPE,
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
    }

    destination = experiment_run_directory(EXPERIMENT_ID, context.window, manifest["run_id"])
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump(residual_model, destination / "cost_model.pkl")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (destination / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    rows.to_csv(destination / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
    record_experiment({
        "experiment_id": EXPERIMENT_ID,
        "name": "remaining_overrun_target",
        "model_role": "experiment",
        "run_id": manifest["run_id"],
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

    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": manifest["run_id"],
            "scope": EXPERIMENT_SCOPE,
            "selected_algorithm": algorithm,
            "decision": decision,
            "promotion_allowed": False,
        },
        "overall_comparison": {
            "production_cost_mae": production_metrics.get("MAE"),
            "experiment_cost_mae": experiment_metrics.get("MAE"),
            "absolute_mae_improvement_pp": report.get("absolute_mae_improvement_pp"),
            "improvement_percentage": improvement,
            "candidate_better": improvement is not None and improvement > 0,
            "success_threshold_passed": report["success_threshold"]["passed"],
            "paired_project_comparison": paired,
            "production_stage_metrics": report["production_lifecycle_stage_metrics"],
            "experiment_stage_metrics": report["experiment_lifecycle_stage_metrics"],
            "comparison_test_projects": report["comparison_control"]["test_projects"],
            "comparison_test_snapshots": report["comparison_control"]["test_rows"],
        },
        "runtime_state": {"model": residual_model, "features": features},
    }


def filter_comparable_rows(frame: pd.DataFrame, runtime_state: dict) -> pd.DataFrame:
    if CURRENT_OVERRUN not in frame:
        return frame.iloc[0:0].copy()
    return frame[pd.to_numeric(frame[CURRENT_OVERRUN], errors="coerce").notna()].copy()


def predict_project(row: pd.Series, runtime_state: dict) -> dict:
    features = list(runtime_state["features"])
    model = runtime_state["model"]
    current = float(row[CURRENT_OVERRUN])
    remaining = float(model.predict(row.to_frame().T[features])[0])
    return {
        "current_observed_cost_escalation": round(current, 4),
        "predicted_remaining_cost_overrun": round(remaining, 4),
        "predicted_cost_overrun": round(current + remaining, 4),
    }
