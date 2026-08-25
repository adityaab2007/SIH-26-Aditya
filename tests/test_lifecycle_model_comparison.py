from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from backend.app.services import lifecycle_model_comparison_service as comparison


class _AdapterModule:
    @staticmethod
    def fit_against_production(**kwargs):
        return {
            "experiment": {
                "experiment_id": "exp_99",
                "experiment_name": "Test challenger",
                "run_id": "exp-run",
                "scope": "cost_only",
                "decision": "PENDING",
                "promotion_allowed": False,
            },
            "overall_comparison": {
                "production_cost_mae": 40.0,
                "experiment_cost_mae": 35.0,
                "absolute_mae_improvement_pp": 5.0,
                "improvement_percentage": 12.5,
            },
            "runtime_state": {"offset": 2.0},
        }

    @staticmethod
    def filter_comparable_rows(frame, state):
        return frame[frame["eligible"].eq(True)].copy()

    @staticmethod
    def predict_project(row, state):
        return {"predicted_cost_overrun": float(row["anchor"]) + float(state["offset"])}


ADAPTER = SimpleNamespace(
    experiment_id="exp_99",
    sequence=99,
    name="Test challenger",
    scope="cost_only",
    module=_AdapterModule,
)


def test_retrain_and_compare_uses_registered_adapter_and_same_production_run(monkeypatch):
    data = pd.DataFrame([
        {"canonical_project_id": "T", "completion_year": 2015, "snapshot_date": "2015-06-30"},
        {"canonical_project_id": "H", "completion_year": 2019, "snapshot_date": "2019-06-30"},
    ])
    monkeypatch.setattr(comparison, "get_experiment_adapter", lambda experiment_id=None: ADAPTER)
    monkeypatch.setattr(comparison.retraining, "_training_data", lambda: (data, pd.DataFrame(), 2001, 2025))
    production = {
        "run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "model_version": "monthly-2001-2015",
        "window": "2001_2015",
    }
    monkeypatch.setattr(comparison.retraining, "retrain_lifecycle", lambda start, end: production)
    monkeypatch.setattr(comparison.simulation, "_artifact_bundle", lambda start, end, run_id=None: {"metadata": {}})
    monkeypatch.setattr(comparison, "_open_production_session_from_frozen_data", lambda **kwargs: {
        "session_id": "prod-session",
        "run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "leakage_guard": "guard",
    })
    held = pd.DataFrame([
        {"record_index": 4, "completion_year": 2019, "eligible": True},
        {"record_index": 5, "completion_year": 2020, "eligible": False},
    ])
    monkeypatch.setattr(comparison.simulation, "_session", lambda session_id: {"held_out": held})
    comparison._COMPARISON_SESSIONS.clear()

    result = comparison.retrain_and_compare(2001, 2015, "exp_99")
    assert result["production"]["run_id"] == "prod-run"
    assert result["experiment"]["experiment_id"] == "exp_99"
    assert result["experiment"]["run_id"] == "exp-run"
    assert result["overall_comparison"]["experiment_cost_mae"] == 35.0
    assert result["session"]["production_run_id"] == "prod-run"
    assert result["session"]["experiment_run_id"] == "exp-run"
    assert result["session"]["eligible_test_years"] == [{"year": 2019, "projects": 1}]


def test_project_compare_generates_both_predictions_before_reveal(monkeypatch):
    comparison._COMPARISON_SESSIONS.clear()
    comparison._COMPARISON_SESSIONS["cmp"] = {
        "production_session_id": "prod-session",
        "production_run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "adapter_id": "exp_99",
        "experiment_run_id": "exp-run",
        "runtime_state": {"offset": 2.0},
        "comparable_indices": {7},
        "overall": {},
        "candidate_predictions": {},
    }
    monkeypatch.setattr(comparison, "get_experiment_adapter", lambda experiment_id=None: ADAPTER)
    row = pd.Series({"record_index": 7, "anchor": 18.0})
    prod_session = {"predictions": {}}
    monkeypatch.setattr(comparison.simulation, "_session", lambda session_id: prod_session)
    monkeypatch.setattr(comparison.simulation, "_session_row", lambda session, record_index: row)

    def fake_predict(session_id, record_index):
        prod_session["predictions"][7] = {"predicted_cost_overrun": 25.0}
        return {
            "run_id": "prod-run",
            "dataset_fingerprint": "dataset-sha",
            "record_index": 7,
            "project": {"project_name": "Held-out"},
            "predicted_cost_overrun": 25.0,
            "predicted_delay_days": 100.0,
            "predicted_risk": "MEDIUM",
            "audit": {"actual_outcomes_sent_to_browser": False},
        }

    monkeypatch.setattr(comparison.simulation, "predict_custom", fake_predict)
    monkeypatch.setattr(comparison.simulation, "reveal_custom", lambda session_id, record_index: {
        "run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "record_index": 7,
        "actual_cost_overrun": 18.0,
        "cost_error_absolute_pp": 7.0,
    })

    prediction = comparison.predict_comparison("cmp", 7)
    assert prediction["comparison"]["experiment"]["predicted_cost_overrun"] == 20.0
    assert prediction["comparison"]["actual_outcome_sent_to_browser"] is False

    actual = comparison.reveal_comparison("cmp", 7)
    assert actual["actual_cost_overrun"] == 18.0
    assert actual["comparison"]["production_cost_error_absolute_pp"] == 7.0
    assert actual["comparison"]["experiment_cost_error_absolute_pp"] == 2.0
    assert actual["comparison"]["experiment_better_for_project"] is True
    assert actual["comparison"]["individual_error_improvement_percentage"] == 71.429
