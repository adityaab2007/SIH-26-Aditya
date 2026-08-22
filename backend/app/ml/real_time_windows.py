"""Real PAIMANA completed-project time-window training and evaluation.

This module deliberately never reads ``data/project_history.csv``.  That file is
the older deterministic demonstration trajectory; the simulations here use only
records extracted from the official PAIMANA completed-project archive reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
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
TIME_WINDOW_REGISTRY = MODELS / "time_window_registry.json"
FEATURES = ["approved_cost_cr", "sector", "implementing_agency", "planned_commissioning_year"]
NUMERIC = ["approved_cost_cr", "planned_commissioning_year"]
CATEGORICAL = ["sector", "implementing_agency"]
CAT_FEATURE_INDICES = [FEATURES.index(column) for column in CATEGORICAL]


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
KEY_PATTERN = re.compile(r"^\d{4}_\d{4}$")


def version_key(start_year: int, end_year: int) -> str:
    return f"{int(start_year)}_{int(end_year)}"


def window_for(key: str, metadata: dict | None = None) -> Window:
    """Resolve registered and newly generated time windows without hardcoding them."""
    if metadata:
        return Window(key, int(metadata["training_start"]), int(metadata["training_end"]), int(metadata["test_start"]), int(metadata["test_end"]))
    if key in WINDOWS:
        return WINDOWS[key]
    if not KEY_PATTERN.fullmatch(key):
        raise KeyError(key)
    start, end = (int(value) for value in key.split("_"))
    if start > end:
        raise KeyError(key)
    actual_end = int(labelled(outcome_data()).completion_year.max())
    return Window(key, start, end, end + 1, actual_end)


def active_version() -> str | None:
    if not TIME_WINDOW_REGISTRY.exists():
        # The supplied registered baseline is a generated time-window artifact,
        # not the legacy global validation report.
        return "2001_2015" if (MODELS / "2001_2015" / "evaluation_results.json").exists() else None
    return json.loads(TIME_WINDOW_REGISTRY.read_text()).get("active_model_version")


def _set_active_version(key: str) -> None:
    TIME_WINDOW_REGISTRY.write_text(json.dumps({"active_model_version": key, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2))


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


def _regressor(seed: int = 26103, parameters: dict | None = None) -> CatBoostRegressor:
    """Robust categorical regressor; MAE loss limits influence of extreme projects."""
    options = {"iterations": 450, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 6.0}
    options.update(parameters or {})
    return CatBoostRegressor(loss_function="MAE", eval_metric="MAE", random_seed=seed, verbose=False, allow_writing_files=False, **options)


def _classifier(y: pd.Series, seed: int = 26103):
    if y.nunique() <= 1:
        return DummyClassifier(strategy="most_frequent")
    return CatBoostClassifier(
        loss_function="Logloss", eval_metric="F1", iterations=400, depth=6,
        learning_rate=0.05, l2_leaf_reg=6.0, random_seed=seed,
        verbose=False, allow_writing_files=False, auto_class_weights="Balanced",
    )


def _fit_regressor(model, X: pd.DataFrame, y: pd.Series, *, delay_target: bool = False) -> None:
    target = np.log1p(np.maximum(0, y)) if delay_target else y
    model.fit(X, target, cat_features=CAT_FEATURE_INDICES)


def _predict_regressor(model, X: pd.DataFrame, *, delay_target: bool = False) -> np.ndarray:
    values = np.asarray(model.predict(X), dtype=float)
    # Existing registry artifacts before this upgrade predict delay directly;
    # only CatBoost delay models are fitted on log1p(delay).
    if delay_target and model.__class__.__module__.startswith("catboost"):
        return np.maximum(0, np.expm1(np.clip(values, 0, 20)))
    return np.maximum(0, values) if delay_target else values


def _fit_classifier(model, X: pd.DataFrame, y: pd.Series) -> None:
    if isinstance(model, DummyClassifier):
        model.fit(X, y)
    else:
        model.fit(X, y, cat_features=CAT_FEATURE_INDICES)


TUNING_GRID = [
    {"iterations": 300, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
    {"iterations": 450, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 6.0},
    {"iterations": 550, "depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
]


def _tune_regressor(train_data: pd.DataFrame, target_column: str, *, delay_target: bool) -> tuple[dict, list[dict]]:
    """Small temporal grid search; the latest training year is never fitted."""
    validation_year = int(train_data.completion_year.max())
    fitting = train_data[train_data.completion_year < validation_year]
    validation = train_data[train_data.completion_year == validation_year]
    if len(fitting) < 12 or validation.empty:
        return TUNING_GRID[1], [{"parameters": TUNING_GRID[1], "MAE": None, "note": "insufficient internal temporal validation records"}]
    results = []
    for options in TUNING_GRID:
        candidate = _regressor(parameters=options)
        _fit_regressor(candidate, fitting[FEATURES], fitting[target_column], delay_target=delay_target)
        predicted = _predict_regressor(candidate, validation[FEATURES], delay_target=delay_target)
        results.append({"parameters": options, "MAE": round(float(mean_absolute_error(validation[target_column], predicted)), 4)})
    winner = min(results, key=lambda item: item["MAE"])
    return winner["parameters"], results


def model_dir(key: str) -> Path:
    window_for(key)
    return MODELS / key


def train(key: str) -> dict:
    window = window_for(key)
    all_data = labelled(outcome_data())
    train_data = all_data[all_data.completion_year.between(window.training_start, window.training_end)].copy()
    if len(train_data) < 12:
        raise ValueError(f"{key} has only {len(train_data)} real completed-project records; at least 12 are required.")
    cost_parameters, cost_comparison = _tune_regressor(train_data, "actual_cost_overrun_percentage", delay_target=False)
    delay_parameters, delay_comparison = _tune_regressor(train_data, "actual_delay_days", delay_target=True)
    X = train_data[FEATURES]
    cost = _regressor(parameters=cost_parameters); delay = _regressor(seed=26104, parameters=delay_parameters); risk = _classifier(train_data.actual_risk)
    _fit_regressor(cost, X, train_data.actual_cost_overrun_percentage)
    _fit_regressor(delay, X, train_data.actual_delay_days, delay_target=True)
    _fit_classifier(risk, X, train_data.actual_risk)
    target = model_dir(key); target.mkdir(parents=True, exist_ok=True)
    joblib.dump(cost, target / "cost_model.pkl")
    joblib.dump(delay, target / "delay_model.pkl")
    joblib.dump(risk, target / "risk_model.pkl")
    # CatBoost natively processes categorical columns; retain a documented
    # preprocessing artifact for the model-registry contract.
    joblib.dump({"numeric_features": NUMERIC, "categorical_features": CATEGORICAL, "native_categorical_encoding": True}, target / "scaler.pkl")
    metadata = {
        "model_version": key,
        "training_start": window.training_start,
        "training_end": window.training_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "available_actual_end": int(all_data.completion_year.max()),
        "features_used": FEATURES,
        "feature_availability": {
            "available": FEATURES,
            "not_used_without_precompletion_snapshots": ["revised_cost_cr", "expenditure_cr", "physical_progress_pct", "progress_deviation", "duration_ratio", "milestone_delay_count"],
        },
        "algorithms": {"cost": "CatBoostRegressor(MAE)", "delay": "CatBoostRegressor(log1p delay, MAE)", "risk": "CatBoostClassifier"},
        "best_parameters": {"cost": cost_parameters, "delay": delay_parameters},
        "outlier_policy": "Cost uses CatBoost MAE loss. Delay is trained on log1p(delay days) and converted back to days for evaluation; no official record is deleted.",
        "training_samples": int(len(train_data)),
        "data_source": "Official PAIMANA completed-project archive reports",
        "outcome_definition": "Reported completion expenditure versus approved cost; reported completion month versus original commissioning month.",
        "leakage_policy": "Completion expenditure, completion date, and derived outcomes are targets only and never input features.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained",
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (target / "model_comparison.json").write_text(json.dumps({
        "selection_method": "latest-year temporal validation within the selected training range",
        "cost_candidates": cost_comparison,
        "delay_candidates": delay_comparison,
        "selected": {"cost": cost_parameters, "delay": delay_parameters},
    }, indent=2))
    return metadata


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-9
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0


def evaluate(key: str, save: bool = True) -> dict:
    target = model_dir(key)
    metadata = json.loads((target / "metadata.json").read_text())
    window = window_for(key, metadata)
    all_data = labelled(outcome_data())
    actual_end = min(window.test_end, int(all_data.completion_year.max()))
    test_data = all_data[all_data.completion_year.between(window.test_start, actual_end)].copy()
    if test_data.empty:
        raise ValueError(f"No official completed-project outcomes are available for {window.test_start}-{actual_end}.")
    X = test_data[FEATURES]
    cost = joblib.load(target / "cost_model.pkl"); delay = joblib.load(target / "delay_model.pkl"); risk = joblib.load(target / "risk_model.pkl")
    test_data["predicted_cost_overrun"] = _predict_regressor(cost, X)
    test_data["predicted_delay_days"] = _predict_regressor(delay, X, delay_target=True)
    test_data["predicted_risk"] = np.asarray(risk.predict(X), dtype=int).reshape(-1)
    test_data["cost_error"] = test_data.predicted_cost_overrun - test_data.actual_cost_overrun_percentage
    test_data["delay_error"] = test_data.predicted_delay_days - test_data.actual_delay_days
    cost_y = test_data.actual_cost_overrun_percentage.to_numpy(); cost_p = test_data.predicted_cost_overrun.to_numpy()
    delay_y = test_data.actual_delay_days.to_numpy(); delay_p = test_data.predicted_delay_days.to_numpy()
    cost_mae = float(mean_absolute_error(cost_y, cost_p))
    delay_mae = float(mean_absolute_error(delay_y, delay_p))
    cost_scale = max(float(np.mean(np.abs(cost_y))), 1e-9)
    delay_scale = max(float(np.mean(np.abs(delay_y))), 1e-9)
    metrics = {
        "model_version": key,
        "cost_model": {"MAE": round(cost_mae, 3), "RMSE": round(float(mean_squared_error(cost_y, cost_p) ** .5), 3), "MAPE": round(_safe_mape(cost_y, cost_p), 3), "accuracy_percentage": round(max(0.0, 100 * (1 - cost_mae / cost_scale)), 2)},
        "delay_model": {"MAE": round(delay_mae, 3), "MAE_days": round(delay_mae, 3), "RMSE": round(float(mean_squared_error(delay_y, delay_p) ** .5), 3), "RMSE_days": round(float(mean_squared_error(delay_y, delay_p) ** .5), 3), "log_target_RMSE": round(float(mean_squared_error(np.log1p(delay_y), np.log1p(delay_p)) ** .5), 4), "accuracy_percentage": round(max(0.0, 100 * (1 - delay_mae / delay_scale)), 2)},
        "risk_model": {"accuracy": round(float(accuracy_score(test_data.actual_risk, test_data.predicted_risk)), 4), "precision": round(float(precision_score(test_data.actual_risk, test_data.predicted_risk, zero_division=0)), 4), "recall": round(float(recall_score(test_data.actual_risk, test_data.predicted_risk, zero_division=0)), 4), "f1": round(float(f1_score(test_data.actual_risk, test_data.predicted_risk, zero_division=0)), 4)},
        "metadata": {**metadata, "evaluated_test_start": window.test_start, "evaluated_test_end": actual_end, "pending_actual_outcomes": list(range(actual_end + 1, window.test_end + 1)), "testing_samples": int(len(test_data)), "actual_outcome_policy": "Only official completed-project records are evaluated. Future years with no official completion record are forecast-only and excluded from metrics."},
    }
    columns = ["project_id", "project_name", "sector", "implementing_agency", "planned_commissioning_year", "completion_date", "approved_cost_cr", "reported_completion_expenditure_cr", "predicted_cost_overrun", "actual_cost_overrun_percentage", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error", "predicted_risk", "actual_risk"]
    rows = test_data[columns].sort_values(["completion_date", "project_name"]).rename(columns={"actual_cost_overrun_percentage": "actual_cost_overrun"})
    if save:
        rows.to_csv(target / "evaluation_results.csv", index=False)
        rows.to_csv(target / "prediction_validation.csv", index=False)
        (target / "evaluation_results.json").write_text(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "rows": rows}


def rolling_validation(key: str) -> dict:
    """Expanding-window validation using only earlier official completion years."""
    metadata = json.loads((model_dir(key) / "metadata.json").read_text())
    window = window_for(key, metadata)
    all_data = labelled(outcome_data())
    folds = []
    for test_year in range(window.training_start + 1, min(window.test_end, int(all_data.completion_year.max())) + 1):
        fitting = all_data[all_data.completion_year.between(window.training_start, test_year - 1)]
        testing = all_data[all_data.completion_year == test_year]
        if len(fitting) < 12 or testing.empty:
            continue
        # Use the chosen settings but lower iterations for the diagnostic folds.
        params = {**metadata.get("best_parameters", {}).get("cost", {}), "iterations": 180}
        cost = _regressor(seed=26103 + test_year, parameters=params)
        delay = _regressor(seed=26104 + test_year, parameters={**metadata.get("best_parameters", {}).get("delay", {}), "iterations": 180})
        _fit_regressor(cost, fitting[FEATURES], fitting.actual_cost_overrun_percentage)
        _fit_regressor(delay, fitting[FEATURES], fitting.actual_delay_days, delay_target=True)
        cost_pred = _predict_regressor(cost, testing[FEATURES])
        delay_pred = _predict_regressor(delay, testing[FEATURES], delay_target=True)
        folds.append({
            "train_start": int(window.training_start), "train_end": int(test_year - 1), "test_year": int(test_year), "training_samples": int(len(fitting)), "testing_samples": int(len(testing)),
            "cost_MAE": round(float(mean_absolute_error(testing.actual_cost_overrun_percentage, cost_pred)), 3),
            "cost_RMSE": round(float(mean_squared_error(testing.actual_cost_overrun_percentage, cost_pred) ** .5), 3),
            "cost_MAPE": round(_safe_mape(testing.actual_cost_overrun_percentage.to_numpy(), cost_pred), 3),
            "delay_MAE_days": round(float(mean_absolute_error(testing.actual_delay_days, delay_pred)), 3),
            "delay_RMSE_days": round(float(mean_squared_error(testing.actual_delay_days, delay_pred) ** .5), 3),
        })
    report = {
        "model_version": key, "folds": folds, "fold_count": len(folds),
        "average_cost_mae": round(float(np.mean([fold["cost_MAE"] for fold in folds])), 3) if folds else None,
        "average_delay_mae_days": round(float(np.mean([fold["delay_MAE_days"] for fold in folds])), 3) if folds else None,
        "policy": "Each fold trains only on completion years before its test year. No future outcomes are used in fitting.",
    }
    (model_dir(key) / "rolling_validation_results.json").write_text(json.dumps(report, indent=2))
    return report


def retrain(start_year: int, end_year: int) -> dict:
    """Persist a newly selected historical window and its future-only evaluation."""
    key = version_key(start_year, end_year)
    available_end = int(labelled(outcome_data()).completion_year.max())
    if end_year >= available_end:
        raise ValueError(f"Training must end before {available_end} so an unseen future period remains.")
    metadata = train(key)
    result = evaluate(key, save=True)
    rolling = rolling_validation(key)
    _set_active_version(key)
    return {
        "status": "success", "model_version": key,
        "training_years": f"{metadata['training_start']}-{metadata['training_end']}",
        "testing_years": f"{metadata['test_start']}-{metadata['test_end']}",
        "metrics": result["metrics"], "training_samples": metadata["training_samples"],
        "testing_samples": result["metrics"]["metadata"]["testing_samples"], "rolling_validation": rolling,
    }


def versions() -> list[dict]:
    result = []
    for target in sorted(MODELS.iterdir() if MODELS.exists() else [], key=lambda path: path.name):
        if not target.is_dir() or not KEY_PATTERN.fullmatch(target.name) or not (target / "metadata.json").exists():
            continue
        metadata = json.loads((target / "metadata.json").read_text())
        result.append({"key": target.name, **metadata, "training_label": f"{metadata['training_start']}-{metadata['training_end']}", "testing_label": f"{metadata['test_start']}-{metadata['test_end']}", "active": target.name == active_version()})
    return result
