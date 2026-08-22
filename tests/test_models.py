from pathlib import Path
import json
import math
import pandas as pd

from backend.app.ml.features import TEMPORAL_FEATURES, engineer_temporal_features, load_project_history
from backend.app.ml.forward_labels import build_forward_labels

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_artifacts_and_metrics_exist():
    assert (ROOT / "models" / "cost_model.pkl").exists()
    assert (ROOT / "models" / "delay_model.pkl").exists()
    metrics = json.loads((ROOT / "models" / "model_metrics.json").read_text())
    assert "time" in metrics["metadata"]["split_strategy"]
    assert metrics["metadata"]["validation_timestamp_utc"]
    for task in ["cost_model", "delay_model"]:
        assert metrics[task]["best_model"] in {"xgboost", "random_forest", "catboost"}
        assert math.isfinite(metrics[task]["MAE"])
        assert {"MAE", "RMSE", "R2"}.issubset(metrics[task])


def test_forward_labels_are_future_outcomes_only():
    labelled = build_forward_labels(engineer_temporal_features(load_project_history()))
    assert not labelled.empty
    assert (labelled.actual_completion_date > labelled.month).all()
    expected = (labelled.actual_cost - labelled.original_cost) / labelled.original_cost * 100
    assert (labelled.future_cost_escalation_percentage.round(8) == expected.round(8)).all()
    forbidden = {"actual_cost", "actual_completion_date", "future_cost_overrun_percentage", "future_delay_days", "future_cost_escalation_percentage", "future_schedule_extension_days"}
    assert forbidden.isdisjoint(TEMPORAL_FEATURES)


def test_backtest_rows_are_cutoff_predictions():
    validation = pd.read_csv(ROOT / "data" / "processed" / "prediction_validation.csv", dtype={"project_id": str})
    assert not validation.empty
    assert {"project_id", "prediction_date", "predicted_cost_overrun", "actual_cost_overrun", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error"}.issubset(validation)
    assert (validation.cost_error.round(8) == (validation.predicted_cost_overrun - validation.actual_cost_overrun).round(8)).all()
