"""SIH26103 model validation utilities.

Provides time-aware backtesting helpers so predictions are evaluated only on
information that would have been available at prediction time.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score


def temporal_split(df: pd.DataFrame, date_column: str, train_ratio: float = 0.8):
    """Split project snapshots chronologically instead of randomly."""
    data = df.sort_values(date_column).reset_index(drop=True)
    cut = int(len(data) * train_ratio)
    return data.iloc[:cut], data.iloc[cut:]


def evaluate_regression(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_classification(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def build_prediction_proof(project_id, predicted_cost, actual_cost, predicted_delay, actual_delay):
    """Creates judge-friendly AI prediction vs actual comparison."""
    return {
        "project_id": project_id,
        "cost": {
            "predicted_overrun_percent": predicted_cost,
            "actual_overrun_percent": actual_cost,
            "absolute_error": abs(predicted_cost - actual_cost),
        },
        "schedule": {
            "predicted_delay_days": predicted_delay,
            "actual_delay_days": actual_delay,
            "absolute_error": abs(predicted_delay - actual_delay),
        },
    }
