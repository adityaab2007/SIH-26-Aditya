"""Real PAIMANA completed-project time-window training and evaluation.

This module deliberately never reads ``data/project_history.csv``.  That file is
the older deterministic demonstration trajectory; the simulations here use only
records extracted from the official PAIMANA completed-project archive reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
OUTCOMES = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"
MODELS = ROOT / "models"
FEATURES = ["approved_cost_cr", "sector", "implementing_agency", "planned_commissioning_year"]
NUMERIC = ["approved_cost_cr", "planned_commissioning_year"]
CATEGORICAL = ["sector", "implementing_agency"]


@dataclass(frozen=True)
class Window:
    version: str
    training_start: int
    training_end: int
    test_start: int
    test_end: int


WINDOWS = {
    "2001_2015": Window("v1", 2001, 2015, 2016, 2021),
    "2015_2021": Window("v2", 2015, 2021, 2022, 2028),
}


def outcome_data() -> pd.DataFrame:
    if not OUTCOMES.exists():
        raise FileNotFoundError(f"{OUTCOMES} is missing. Run scripts/ingest_paimana_completed_reports.py first.")
    frame = pd.read_csv(OUTCOMES, dtype={"project_id": str})
    for column in ["completion_date", "planned_commissioning_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["approved_cost_cr"] = pd.to_numeric(result["approved_cost_cr"], errors="coerce")
    result["planned_commissioning_year"] = result["planned_commissioning_date"].dt.year
    result["completion_year"] = result["completion_date"].dt.year
    for column in CATEGORICAL:
        result[column] = result[column].fillna("Not reported").astype(str)
    return result


def labelled(frame: pd.DataFrame) -> pd.DataFrame:
    result = features(frame)
    result["actual_cost_overrun_percentage"] = np.where(
        result["approved_cost_cr"] > 0,
        (pd.to_numeric(result["reported_completion_expenditure_cr"], errors="coerce") - result["approved_cost_cr"]) / result["approved_cost_cr"] * 100,
        np.nan,
    )
    result["actual_delay_days"] = (result["completion_date"] - result["planned_commissioning_date"]).dt.days
    result["actual_delay_days"] = result["actual_delay_days"].clip(lower=0)
    result["actual_risk"] = ((result["actual_cost_overrun_percentage"] >= 5) | (result["actual_delay_days"] >= 90)).astype(int)
    return result.dropna(subset=["approved_cost_cr", "planned_commissioning_year", "actual_cost_overrun_percentage", "actual_delay_days"])


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def _regressor(seed: int = 26103) -> Pipeline:
    return Pipeline([("preprocess", preprocessor()), ("model", RandomForestRegressor(n_estimators=220, min_samples_leaf=2, random_state=seed, n_jobs=2))])


def _classifier(y: pd.Series, seed: int = 26103) -> Pipeline:
    model = RandomForestClassifier(n_estimators=220, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=2) if y.nunique() > 1 else DummyClassifier(strategy="most_frequent")
    return Pipeline([("preprocess", preprocessor()), ("model", model)])


def model_dir(key: str) -> Path:
    if key not in WINDOWS:
        raise KeyError(key)
    return MODELS / key


def train(key: str) -> dict:
    window = WINDOWS[key]
    all_data = labelled(outcome_data())
    train_data = all_data[all_data.completion_year.between(window.training_start, window.training_end)].copy()
    if len(train_data) < 12:
        raise ValueError(f"{key} has only {len(train_data)} real completed-project records; at least 12 are required.")
    X = train_data[FEATURES]
    cost = _regressor(); delay = _regressor(seed=26104); risk = _classifier(train_data.actual_risk)
    cost.fit(X, train_data.actual_cost_overrun_percentage)
    delay.fit(X, train_data.actual_delay_days)
    risk.fit(X, train_data.actual_risk)
    target = model_dir(key); target.mkdir(parents=True, exist_ok=True)
    joblib.dump(cost, target / "cost_model.pkl")
    joblib.dump(delay, target / "delay_model.pkl")
    joblib.dump(risk, target / "risk_model.pkl")
    # The preprocessing scaler is also persisted explicitly for registry compatibility.
    joblib.dump(cost.named_steps["preprocess"], target / "scaler.pkl")
    metadata = {
        "model_version": window.version,
        "training_start": window.training_start,
        "training_end": window.training_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "available_actual_end": int(all_data.completion_year.max()),
        "features_used": FEATURES,
        "training_samples": int(len(train_data)),
        "data_source": "Official PAIMANA completed-project archive reports",
        "outcome_definition": "Reported completion expenditure versus approved cost; reported completion month versus original commissioning month.",
        "leakage_policy": "Completion expenditure, completion date, and derived outcomes are targets only and never input features.",
        "status": "trained",
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-9
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0


def evaluate(key: str, save: bool = True) -> dict:
    target = model_dir(key)
    metadata = json.loads((target / "metadata.json").read_text())
    window = WINDOWS[key]
    all_data = labelled(outcome_data())
    actual_end = min(window.test_end, int(all_data.completion_year.max()))
    test_data = all_data[all_data.completion_year.between(window.test_start, actual_end)].copy()
    if test_data.empty:
        raise ValueError(f"No official completed-project outcomes are available for {window.test_start}-{actual_end}.")
    X = test_data[FEATURES]
    cost = joblib.load(target / "cost_model.pkl"); delay = joblib.load(target / "delay_model.pkl"); risk = joblib.load(target / "risk_model.pkl")
    test_data["predicted_cost_overrun"] = cost.predict(X)
    test_data["predicted_delay_days"] = np.maximum(0, delay.predict(X))
    test_data["predicted_risk"] = risk.predict(X).astype(int)
    test_data["cost_error"] = test_data.predicted_cost_overrun - test_data.actual_cost_overrun_percentage
    test_data["delay_error"] = test_data.predicted_delay_days - test_data.actual_delay_days
    cost_y = test_data.actual_cost_overrun_percentage.to_numpy(); cost_p = test_data.predicted_cost_overrun.to_numpy()
    delay_y = test_data.actual_delay_days.to_numpy(); delay_p = test_data.predicted_delay_days.to_numpy()
    metrics = {
        "cost_model": {"MAE": round(float(mean_absolute_error(cost_y, cost_p)), 3), "RMSE": round(float(mean_squared_error(cost_y, cost_p) ** .5), 3), "MAPE": round(_safe_mape(cost_y, cost_p), 3)},
        "delay_model": {"MAE_days": round(float(mean_absolute_error(delay_y, delay_p)), 3), "RMSE_days": round(float(mean_squared_error(delay_y, delay_p) ** .5), 3)},
        "risk_model": {"accuracy": round(float(accuracy_score(test_data.actual_risk, test_data.predicted_risk)), 4), "precision": round(float(precision_score(test_data.actual_risk, test_data.predicted_risk, zero_division=0)), 4), "recall": round(float(recall_score(test_data.actual_risk, test_data.predicted_risk, zero_division=0)), 4), "f1": round(float(f1_score(test_data.actual_risk, test_data.predicted_risk, zero_division=0)), 4)},
        "metadata": {**metadata, "evaluated_test_start": window.test_start, "evaluated_test_end": actual_end, "pending_actual_outcomes": list(range(actual_end + 1, window.test_end + 1)), "testing_samples": int(len(test_data)), "actual_outcome_policy": "Only official completed-project records are evaluated. Future years with no official completion record are forecast-only and excluded from metrics."},
    }
    columns = ["project_id", "project_name", "sector", "implementing_agency", "planned_commissioning_year", "completion_date", "approved_cost_cr", "reported_completion_expenditure_cr", "predicted_cost_overrun", "actual_cost_overrun_percentage", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error", "predicted_risk", "actual_risk"]
    rows = test_data[columns].sort_values(["completion_date", "project_name"])
    if save:
        rows.to_csv(target / "evaluation_results.csv", index=False)
        (target / "evaluation_results.json").write_text(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "rows": rows}


def versions() -> list[dict]:
    result = []
    for key, window in WINDOWS.items():
        metadata_path = model_dir(key) / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"status": "not_trained"}
        result.append({"key": key, **metadata, "training_label": f"{window.training_start}-{window.training_end}", "testing_label": f"{window.test_start}-{window.test_end}"})
    return result
