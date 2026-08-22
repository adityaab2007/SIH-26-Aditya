from pathlib import Path
import json
import math
import pandas as pd

from backend.app.ml.features import TEMPORAL_FEATURES, engineer_temporal_features, load_project_history
from backend.app.ml.forward_labels import build_forward_labels
from backend.app.ml.real_time_windows import labelled

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


def test_official_completed_archive_includes_older_available_years():
    outcomes = pd.read_csv(ROOT / "data" / "processed" / "paimana_completed_outcomes.csv")
    years = pd.to_datetime(outcomes["completion_date"], errors="coerce").dt.year.dropna().astype(int)
    assert years.min() == 2001
    assert set(range(2001, 2009)).issubset(set(years))


def test_completed_outcome_labels_require_real_final_values_and_dates():
    frame = pd.DataFrame({
        "project_id": ["valid", "no-final-cost", "no-date"],
        "project_name": ["Valid", "No final", "No date"],
        "sector": ["Road", "Road", "Road"],
        "implementing_agency": ["Agency", "Agency", "Agency"],
        "approved_cost_cr": [100.0, 100.0, 100.0],
        "reported_completion_expenditure_cr": [125.0, None, 125.0],
        "planned_commissioning_date": ["2020-01-01", "2020-01-01", None],
        "completion_date": ["2020-02-15", "2020-02-15", "2020-02-15"],
    })
    output = labelled(frame)
    assert output["project_id"].tolist() == ["valid"]
    assert output.iloc[0]["actual_cost_overrun_percentage"] == 25.0
    assert output.iloc[0]["actual_delay_days"] == 45
