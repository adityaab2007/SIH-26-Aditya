"""Experiment 1: change only the cost target to a signed log1p scale.

The production model and registry are read-only inputs. Experiment artifacts
are written under ``models/experiments/log_cost_target``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.app.ml.real_time_windows import (
    CAT_FEATURE_INDICES,
    FEATURES,
    MODELS,
    REPORTS,
    _algorithm_regressor,
    _residual_corrections,
    _safe_mape,
    _sample_weights,
    add_leave_one_out_training_priors,
    apply_historical_priors,
    historical_prior_maps,
    labelled,
    outcome_data,
    sector_bucket,
    window_for,
)

BASELINE_KEY = "2001_2017"
EXPERIMENT_NAME = "log_cost_target"
EXPERIMENT_DIR = MODELS / "experiments" / EXPERIMENT_NAME
EXPERIMENT_REPORTS = REPORTS / "experiments"


def log_cost_transform(values) -> np.ndarray:
    """Apply log1p to overruns with a signed extension for underruns."""
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.log1p(np.abs(array))


def inverse_log_cost_transform(values) -> np.ndarray:
    """Return signed-log predictions to cost-overrun percentage points."""
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.expm1(np.abs(array))


class LogCostTargetModel:
    """CatBoost with the baseline hyperparameters and only a target transform."""

    def __init__(self, seed: int = 26103):
        self.seed = seed
        self.model = _algorithm_regressor("catboost_mae_d4", seed)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogCostTargetModel":
        transformed = log_cost_transform(y)
        self.model.fit(
            X,
            transformed,
            cat_features=CAT_FEATURE_INDICES,
            sample_weight=_sample_weights(pd.Series(transformed)),
        )
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return inverse_log_cost_transform(self.predict_log(X))


def evaluate_on_original_scale(actual, log_predictions) -> dict:
    """Evaluate only after converting model output back to percentage points."""
    actual_values = np.asarray(actual, dtype=float)
    predicted = inverse_log_cost_transform(log_predictions)
    return {
        "mae": round(float(mean_absolute_error(actual_values, predicted)), 3),
        "rmse": round(float(mean_squared_error(actual_values, predicted) ** 0.5), 3),
        "mape": round(_safe_mape(actual_values, predicted), 3),
        "predictions": predicted,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _target_analysis(train_data: pd.DataFrame) -> dict:
    values = train_data.actual_cost_overrun_percentage.astype(float)
    percentiles = {
        f"p{int(percentile * 100)}": round(float(values.quantile(percentile)), 4)
        for percentile in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    }
    return {
        "target": "cost_overrun_percentage",
        "training_window": f"{int(train_data.completion_year.min())}-{int(train_data.completion_year.max())}",
        "projects": int(len(values)),
        "minimum": round(float(values.min()), 4),
        "maximum": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
        "standard_deviation": round(float(values.std()), 4),
        "percentile_distribution": percentiles,
        "plain_log1p_invalid_projects": int(values.le(-1).sum()),
        "negative_overrun_projects": int(values.lt(0).sum()),
        "transform_policy": "signed_log1p preserves every baseline project; plain log1p is used unchanged for non-negative overruns",
    }


def _fit_log_model(frame: pd.DataFrame, seed: int) -> LogCostTargetModel:
    return LogCostTargetModel(seed).fit(frame[FEATURES], frame.actual_cost_overrun_percentage)


def _sector_correction(train_data: pd.DataFrame) -> dict:
    """Mirror the production nested temporal correction using the log model."""
    years = sorted(int(year) for year in train_data.completion_year.unique())
    if len(years) < 3:
        return {"enabled": False, "reason": "insufficient temporal years", "corrections": {}}
    correction_year, validation_year = years[-2], years[-1]
    base = train_data[train_data.completion_year < correction_year].copy()
    correction_rows = train_data[train_data.completion_year == correction_year].copy()
    validation = train_data[train_data.completion_year == validation_year].copy()
    if len(base) < 12 or correction_rows.empty or validation.empty:
        return {"enabled": False, "reason": "insufficient rows for nested temporal correction test", "corrections": {}}

    correction_rows = apply_historical_priors(correction_rows, historical_prior_maps(base))
    base_model = _fit_log_model(add_leave_one_out_training_priors(base), 27101)
    residuals = correction_rows.actual_cost_overrun_percentage.to_numpy() - base_model.predict(correction_rows[FEATURES])
    candidate = _residual_corrections(correction_rows, residuals)

    pre_validation = train_data[train_data.completion_year < validation_year].copy()
    validation = apply_historical_priors(validation, historical_prior_maps(pre_validation))
    validation_model = _fit_log_model(add_leave_one_out_training_priors(pre_validation), 27102)
    baseline = validation_model.predict(validation[FEATURES])
    corrected = baseline + sector_bucket(validation.sector).map(candidate).fillna(0).to_numpy(dtype=float)
    baseline_mae = float(mean_absolute_error(validation.actual_cost_overrun_percentage, baseline))
    corrected_mae = float(mean_absolute_error(validation.actual_cost_overrun_percentage, corrected))
    enabled = corrected_mae < baseline_mae

    final_base = train_data[train_data.completion_year < validation_year].copy()
    latest = train_data[train_data.completion_year == validation_year].copy()
    latest = apply_historical_priors(latest, historical_prior_maps(final_base))
    final_model = _fit_log_model(add_leave_one_out_training_priors(final_base), 27103)
    final_residuals = latest.actual_cost_overrun_percentage.to_numpy() - final_model.predict(latest[FEATURES])
    return {
        "enabled": enabled,
        "validation_year": validation_year,
        "correction_year": correction_year,
        "baseline_mae": round(baseline_mae, 4),
        "corrected_mae": round(corrected_mae, 4),
        "corrections": _residual_corrections(latest, final_residuals) if enabled else {},
        "policy": "Same nested temporal correction methodology as the baseline; only the cost target transform differs.",
    }


def run_experiment() -> dict:
    baseline_dir = MODELS / BASELINE_KEY
    baseline_metadata = json.loads((baseline_dir / "metadata.json").read_text())
    baseline_metrics = json.loads((baseline_dir / "evaluation_results.json").read_text())
    baseline_model_path = baseline_dir / "cost_model.pkl"
    baseline_hash_before = _sha256(baseline_model_path)
    window = window_for(BASELINE_KEY, baseline_metadata)

    all_data = labelled(outcome_data())
    train_data = all_data[all_data.completion_year.between(window.training_start, window.training_end)].copy()
    test_end = min(window.test_end, int(all_data.completion_year.max()))
    test_data = all_data[all_data.completion_year.between(window.test_start, test_end)].copy()
    priors = historical_prior_maps(train_data)
    model_train_data = add_leave_one_out_training_priors(train_data)
    test_data = apply_historical_priors(test_data, priors)

    if baseline_metadata["features_used"] != FEATURES:
        raise ValueError("Experiment feature contract differs from the baseline.")
    model = _fit_log_model(model_train_data, 26103)
    correction = _sector_correction(train_data)
    log_predictions = model.predict_log(test_data[FEATURES])
    cost_metrics = evaluate_on_original_scale(test_data.actual_cost_overrun_percentage, log_predictions)
    predictions = cost_metrics.pop("predictions")
    if correction["enabled"]:
        predictions = predictions + sector_bucket(test_data.sector).map(correction["corrections"]).fillna(0).to_numpy(dtype=float)
        cost_metrics = {
            "mae": round(float(mean_absolute_error(test_data.actual_cost_overrun_percentage, predictions)), 3),
            "rmse": round(float(mean_squared_error(test_data.actual_cost_overrun_percentage, predictions) ** 0.5), 3),
            "mape": round(_safe_mape(test_data.actual_cost_overrun_percentage.to_numpy(), predictions), 3),
        }

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, EXPERIMENT_DIR / "cost_model.pkl")
    prediction_rows = pd.DataFrame({
        "project_id": test_data.project_id.astype(str),
        "actual_cost_overrun": test_data.actual_cost_overrun_percentage,
        "predicted_cost_overrun": predictions,
    })
    prediction_rows["cost_error"] = prediction_rows.predicted_cost_overrun - prediction_rows.actual_cost_overrun
    prediction_rows.to_csv(EXPERIMENT_DIR / "prediction_validation.csv", index=False)

    baseline_cost_mae = float(baseline_metrics["cost_model"]["MAE"])
    baseline_delay_mae = float(baseline_metrics["delay_model"]["MAE_days"])
    baseline_risk = baseline_metrics["risk_model"]
    absolute_improvement = baseline_cost_mae - cost_metrics["mae"]
    percentage_improvement = absolute_improvement / baseline_cost_mae * 100
    baseline_report = {
        "experiment": "baseline",
        "target": "cost_overrun_percentage",
        "model_version": BASELINE_KEY,
        "cost_mae": baseline_cost_mae,
        "delay_mae": baseline_delay_mae,
        "feature_count": len(FEATURES),
    }
    v2_report = {
        "experiment": EXPERIMENT_NAME,
        "cost_model": cost_metrics,
        "delay_model": {"mae": baseline_delay_mae, "status": "unchanged; copied from baseline evaluation"},
        "risk_model": {**baseline_risk, "status": "unchanged; copied from baseline evaluation"},
        "feature_count": len(FEATURES),
        "training_samples": int(len(train_data)),
        "testing_samples": int(len(test_data)),
        "evaluation_scale": "original cost-overrun percentage points after inverse transform",
    }
    evolution = [
        {"version": "v1", "experiment": "baseline", "cost_mae": baseline_cost_mae},
        {
            "version": "v2", "experiment": EXPERIMENT_NAME, "cost_mae": cost_metrics["mae"],
            "absolute_improvement": round(absolute_improvement, 3),
            "percentage_improvement": round(percentage_improvement, 2),
        },
    ]
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "target": "log_cost_overrun",
        "target_transform": "log1p",
        "negative_value_policy": "signed extension: sign(y) * log1p(abs(y)); inverse sign(z) * expm1(abs(z))",
        "features_used": FEATURES,
        "feature_count": len(FEATURES),
        "training_window": f"{window.training_start}-{window.training_end}",
        "testing_window": f"{window.test_start}-{test_end}",
        "created_from": "v1 baseline",
        "baseline_model_version": BASELINE_KEY,
        "algorithm": baseline_metadata["model_type"]["cost"],
        "temporal_split": baseline_metadata["validation_method"],
        "training_samples": int(len(train_data)),
        "testing_samples": int(len(test_data)),
        "sector_correction": correction,
        "evaluation_scale": "original cost-overrun percentage points",
        "baseline_cost_model_sha256": baseline_hash_before,
        "baseline_unchanged": baseline_hash_before == _sha256(baseline_model_path),
        "delay_model_changed": False,
        "risk_model_changed": False,
        "feature_pipeline_changed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(metadata, EXPERIMENT_DIR / "metadata.json")
    _write_json(baseline_report, EXPERIMENT_REPORTS / "v1_baseline.json")
    _write_json(_target_analysis(train_data), EXPERIMENT_REPORTS / "cost_target_analysis_before.json")
    _write_json(v2_report, EXPERIMENT_REPORTS / "v2_log_cost_target.json")
    _write_json(evolution, EXPERIMENT_REPORTS / "model_evolution.json")
    return {"metadata": metadata, "baseline": baseline_report, "experiment": v2_report, "evolution": evolution}


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
