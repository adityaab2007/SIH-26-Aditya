"""Historical-cutoff verification for the temporal SIH26103 forecasting models."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_COLUMNS, TEMPORAL_FEATURES

ROOT = Path(__file__).resolve().parents[3]
VALIDATION_CSV = ROOT / "data" / "processed" / "prediction_validation.csv"
VALIDATION_REPORT = ROOT / "models" / "validation_report.json"


def _predict(bundle: dict, row: pd.Series) -> float:
    x = row[TEMPORAL_FEATURES].to_frame().T
    return float(bundle["model"].predict(bundle["preprocess"].transform(x))[0])


def _verification_bundle(source_bundle: dict, training: pd.DataFrame, target: str) -> dict:
    numeric = [feature for feature in TEMPORAL_FEATURES if feature not in CATEGORICAL_COLUMNS]
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_COLUMNS),
    ])
    transformed = preprocess.fit_transform(training[TEMPORAL_FEATURES])
    model = clone(source_bundle["model"])
    model.fit(transformed, training[target])
    return {"preprocess": preprocess, "model": model}


def _cutoff_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One forecast per project, frozen before its completed final outcome."""
    observation_end = frame["month"].max()
    completed = frame[frame["actual_completion_date"] <= observation_end]
    selected = []
    for _, project in completed.groupby("project_id"):
        project = project.sort_values("month")
        actual_completion = project["actual_completion_date"].iloc[0]
        planned_start = project["planned_start_date"].iloc[0]
        # Cut off at least 90 days before completion, or at 75% of the planned life.
        lifecycle_days = max(120, int((actual_completion - planned_start).days))
        intended_cutoff = actual_completion - pd.Timedelta(days=max(90, lifecycle_days // 4))
        candidates = project[project["month"] <= intended_cutoff]
        if not candidates.empty:
            selected.append(candidates.iloc[-1])
    return pd.DataFrame(selected)


def _regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(actual, predicted)), 3),
        "RMSE": round(float(math.sqrt(mean_squared_error(actual, predicted))), 3),
        "R2": round(float(r2_score(actual, predicted)), 4),
        "accuracy_percentage": round(float(np.mean(np.maximum(0, 100 - np.abs(predicted - actual) / np.maximum(np.abs(actual), 1) * 100))), 2),
    }


def _risk_metrics(actual_cost: pd.Series, actual_delay: pd.Series, predicted_cost: pd.Series, predicted_delay: pd.Series) -> dict:
    # Elevated-risk thresholds keep the verification classification meaningful across
    # this completed-project cohort; they are documented rather than label-derived.
    actual = ((actual_cost >= 40) | (actual_delay >= 365)).astype(int)
    predicted = ((predicted_cost >= 40) | (predicted_delay >= 365)).astype(int)
    return {
        "definition": "elevated risk when cost overrun is at least 40% or delay is at least 365 days",
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "confusion_matrix": {"true_negative": int(((actual == 0) & (predicted == 0)).sum()), "false_positive": int(((actual == 0) & (predicted == 1)).sum()), "false_negative": int(((actual == 1) & (predicted == 0)).sum()), "true_positive": int(((actual == 1) & (predicted == 1)).sum())},
    }


def generate_prediction_validation(frame: pd.DataFrame, cost_model_path: Path, delay_model_path: Path) -> dict:
    """Write reproducible cutoff predictions and aggregate validation metrics."""
    cutoffs = _cutoff_rows(frame)
    historical = cutoffs[cutoffs.actual_completion_date.dt.year <= 2024]
    evaluation = cutoffs[cutoffs.actual_completion_date.dt.year >= 2025]
    if historical.empty or evaluation.empty:
        raise ValueError("No completed projects with a valid historical cutoff are available for backtesting.")
    cost_bundle = _verification_bundle(joblib.load(cost_model_path), historical, "future_cost_escalation_percentage")
    delay_bundle = _verification_bundle(joblib.load(delay_model_path), historical, "future_schedule_extension_days")
    rows = []
    for _, row in evaluation.iterrows():
        predicted_cost = round(_predict(cost_bundle, row), 2)
        predicted_delay = round(max(0.0, _predict(delay_bundle, row)), 2)
        actual_cost = round(float(row.future_cost_escalation_percentage), 2)
        actual_delay = round(float(row.future_schedule_extension_days), 2)
        completeness = float(row[TEMPORAL_FEATURES].notna().mean())
        rows.append({
            "project_id": str(row.project_id), "prediction_date": row.month.strftime("%Y-%m-%d"),
            "predicted_cost_overrun": predicted_cost, "actual_cost_overrun": actual_cost, "cost_error": round(predicted_cost - actual_cost, 2),
            "predicted_delay_days": predicted_delay, "actual_delay_days": actual_delay, "delay_error": round(predicted_delay - actual_delay, 2),
            "cost_accuracy_percentage": round(max(0, 100 - abs(predicted_cost - actual_cost) / max(abs(actual_cost), 1) * 100), 2),
            "delay_accuracy_percentage": round(max(0, 100 - abs(predicted_delay - actual_delay) / max(abs(actual_delay), 1) * 100), 2),
            "model_confidence_percentage": round(completeness * 100, 1),
        })
    output = pd.DataFrame(rows).sort_values(["prediction_date", "project_id"])
    VALIDATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(VALIDATION_CSV, index=False)
    report = {
        "metadata": {"records": len(output), "training_projects": len(historical), "evaluation_projects": len(evaluation), "temporal_holdout": "verification models fit only projects completed through 2024; metrics use projects completed in 2025-2026", "cutoff_policy": "one snapshot per completed project, frozen at least 90 days or the final quarter of lifecycle before actual completion", "future_information_policy": "predictions use only features available at the selected prediction date; evaluation projects are excluded from verification-model fitting", "confidence_definition": "percentage of required temporal model features available at the cutoff"},
        "cost_model": _regression_metrics(output.actual_cost_overrun, output.predicted_cost_overrun),
        "delay_model": _regression_metrics(output.actual_delay_days, output.predicted_delay_days),
        "risk_classification": _risk_metrics(output.actual_cost_overrun, output.actual_delay_days, output.predicted_cost_overrun, output.predicted_delay_days),
    }
    VALIDATION_REPORT.write_text(json.dumps(report, indent=2))
    return report
