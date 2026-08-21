"""SIH26103 model validation utilities.

Uses time-aware validation to avoid future data leakage in project forecasting.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
import numpy as np


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def backtest_predictions(project_ids, actual, predicted):
    rows = []
    for pid, a, p in zip(project_ids, actual, predicted):
        rows.append({
            "project_id": pid,
            "actual": float(a),
            "predicted": float(p),
            "error": float(abs(a-p))
        })
    return rows
