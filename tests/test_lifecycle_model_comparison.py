from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.services import lifecycle_model_comparison_service as comparison


class _Candidate:
    def predict(self, frame):
        return np.array([8.0] * len(frame), dtype=float)


def test_retrain_and_compare_binds_fresh_production_and_experiment(monkeypatch):
    data = pd.DataFrame([
        {"canonical_project_id": "T", "completion_year": 2015, "snapshot_date": "2015-06-30", "record_index": 0, "cost_escalation_percentage": 5.0},
        {"canonical_project_id": "H", "completion_year": 2019, "snapshot_date": "2019-06-30", "record_index": 1, "cost_escalation_percentage": 10.0},
    ])
    monkeypatch.setattr(comparison.retraining, "_training_data", lambda: (data, pd.DataFrame(), 2001, 2025))
    production = {
        "run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "model_version": "monthly-2001-2015",
        "window": "2001_2015",
        "features_used": ["cost_escalation_percentage"],
        "selected_algorithms": {"cost": "xgboost"},
    }
    monkeypatch.setattr(comparison.retraining, "retrain_lifecycle", lambda start, end: production)
    monkeypatch.setattr(comparison.simulation, "_artifact_bundle", lambda start, end, run_id=None: {"metadata": {"features_used": ["cost_escalation_percentage"], "selected_algorithms": {"cost": "xgboost"}}, "cost": object()})
    held = pd.DataFrame([
        {"record_index": 4, "completion_year": 2019, "cost_escalation_percentage": 12.0},
        {"record_index": 5, "completion_year": 2020, "cost_escalation_percentage": np.nan},
    ])
    monkeypatch.setattr(comparison.simulation, "_session", lambda session_id: {"held_out": held})
    report = {
        "experiment_id": "exp_03",
        "experiment_name": "Remaining-overrun forecasting",
        "run_id": "exp-run",
        "experiment_scope": "cost_only",
        "selected_algorithm": "xgboost",
        "decision": "REJECTED",
        "production_final_overrun_metrics": {"MAE": 40.0},
        "experiment_reconstructed_final_overrun_metrics": {"MAE": 50.0},
        "absolute_mae_improvement_pp": -10.0,
        "final_mae_improvement_percentage": -25.0,
        "success_threshold": {"passed": False},
        "paired_project_comparison": {"probability_candidate_better": 0.1},
        "production_lifecycle_stage_metrics": {},
        "experiment_lifecycle_stage_metrics": {},
        "comparison_control": {"test_projects": 1, "test_rows": 1},
        "features_used": ["cost_escalation_percentage"],
    }
    monkeypatch.setattr(comparison, "_fit_exp03_against_production", lambda **kwargs: (report, _Candidate()))
    comparison._COMPARISON_SESSIONS.clear()
    comparison.simulation._CUSTOM_SESSIONS.clear()

    result = comparison.retrain_and_compare(2001, 2015)
    assert result["production"]["run_id"] == "prod-run"
    assert result["experiment"]["run_id"] == "exp-run"
    assert result["overall_comparison"]["production_cost_mae"] == 40.0
    assert result["overall_comparison"]["experiment_cost_mae"] == 50.0
    assert result["overall_comparison"]["improvement_percentage"] == -25.0
    assert result["session"]["production_run_id"] == "prod-run"
    assert result["session"]["experiment_run_id"] == "exp-run"
    assert result["session"]["eligible_test_years"] == [{"year": 2019, "projects": 1}]


def test_project_compare_generates_both_predictions_before_one_reveal(monkeypatch):
    comparison._COMPARISON_SESSIONS.clear()
    comparison._COMPARISON_SESSIONS["cmp"] = {
        "production_session_id": "prod-session",
        "production_run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "experiment_id": "exp_03",
        "experiment_run_id": "exp-run",
        "experiment_model": _Candidate(),
        "features": ["cost_escalation_percentage"],
        "comparable_indices": {7},
        "overall": {},
        "candidate_predictions": {},
    }
    row = pd.Series({
        "record_index": 7,
        "cost_escalation_percentage": 12.0,
        "actual_cost_overrun_percentage": 18.0,
    })
    prod_session = {"predictions": {}}
    monkeypatch.setattr(comparison.simulation, "_session", lambda session_id: prod_session)
    monkeypatch.setattr(comparison.simulation, "_session_row", lambda session, record_index: row)

    def fake_predict(session_id, record_index):
        prod_session["predictions"][7] = {"predicted_cost_overrun": 25.0}
        return {
            "session_id": "prod-session",
            "run_id": "prod-run",
            "dataset_fingerprint": "dataset-sha",
            "record_index": 7,
            "project": {"project_name": "Held-out"},
            "predicted_cost_overrun": 25.0,
            "predicted_delay_days": 100.0,
            "predicted_risk": "MEDIUM",
            "risk_probability_percentage": 70.0,
            "model_inputs": {"cost_escalation_percentage": 12.0},
            "shap_explanation": [],
            "audit": {"actual_outcomes_sent_to_browser": False, "project_excluded_from_training": True},
        }

    monkeypatch.setattr(comparison.simulation, "predict_custom", fake_predict)
    monkeypatch.setattr(comparison.simulation, "reveal_custom", lambda session_id, record_index: {
        "run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "record_index": 7,
        "actual_cost_overrun": 18.0,
        "actual_delay_days": 110.0,
        "actual_risk": "MEDIUM",
        "cost_error_absolute_pp": 7.0,
        "delay_error_absolute_days": 10.0,
        "reveal_policy": "hidden until reveal",
        "actual_outcomes_sent_to_browser": True,
    })

    prediction = comparison.predict_comparison("cmp", 7)
    assert prediction["predicted_cost_overrun"] == 25.0
    assert prediction["comparison"]["experiment"]["predicted_remaining_cost_overrun"] == 8.0
    assert prediction["comparison"]["experiment"]["predicted_cost_overrun"] == 20.0
    assert prediction["comparison"]["actual_outcome_sent_to_browser"] is False

    actual = comparison.reveal_comparison("cmp", 7)
    assert actual["actual_cost_overrun"] == 18.0
    assert actual["comparison"]["production_cost_error_absolute_pp"] == 7.0
    assert actual["comparison"]["experiment_cost_error_absolute_pp"] == 2.0
    assert actual["comparison"]["experiment_better_for_project"] is True
    assert actual["comparison"]["individual_error_improvement_percentage"] == 71.429
