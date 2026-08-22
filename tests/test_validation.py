import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.ml.temporal_validation import assert_no_target_leakage, chronological_holdout

client = TestClient(app)


def test_leakage_guard_rejects_target_and_future_columns():
    with pytest.raises(ValueError):
        assert_no_target_leakage(["original_cost_cr", "future_cost_escalation_pct"], "cost_escalation_pct")
    with pytest.raises(ValueError):
        assert_no_target_leakage(["original_cost_cr", "cost_escalation_pct"], "cost_escalation_pct")
    assert_no_target_leakage(["original_cost_cr", "physical_progress_pct"], "cost_escalation_pct")


def test_chronological_holdout_never_trains_on_future_rows():
    frame = pd.DataFrame({
        "snapshot_date": ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31"],
        "project_code": ["A", "B", "C", "D", "E"],
    })
    split = chronological_holdout(frame, test_fraction=0.4)
    assert split is not None
    train_dates = pd.to_datetime(frame.loc[split.train_index, "snapshot_date"])
    test_dates = pd.to_datetime(frame.loc[split.test_index, "snapshot_date"])
    assert train_dates.max() < test_dates.min()


def test_single_snapshot_has_no_fake_temporal_split():
    frame = pd.DataFrame({"snapshot_date": ["2026-05-31"] * 4})
    assert chronological_holdout(frame) is None


def test_validation_dashboard_apis_are_live():
    summary = client.get("/api/models/validation")
    assert summary.status_code == 200
    body = summary.json()
    assert "future outcomes" in body["methodology"]["forecasting_rule"].lower()
    assert body["best_models"]["cost_regressor"]["model"]

    comparison = client.get("/api/models/comparison")
    assert comparison.status_code == 200
    names = {r["model"] for r in comparison.json()["tasks"]["cost_regressor"]}
    assert {"random_forest", "xgboost", "catboost"}.issubset(names)

    backtest = client.get("/api/models/backtest")
    assert backtest.status_code == 200
    assert backtest.json()["available"] is True
    assert len(backtest.json()["tasks"]["cost_regressor"]["rows"]) > 0


def test_shap_explanation_endpoint_uses_trained_artifact():
    response = client.get("/api/models/explain/701263")
    assert response.status_code == 200
    body = response.json()
    assert body["project_code"] == "701263"
    assert len(body["schedule_drivers"]) > 0
    assert len(body["cost_drivers"]) > 0
