from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from backend.app.ml.experiments.adapters import get_experiment_adapter
from backend.app.ml.experiments.trajectory_exp12 import EXP12_FEATURES, engineer_history, enrich_rows
from backend.app.services import lifecycle_model_comparison_service as comparison


def _history():
    return pd.DataFrame([
        {"canonical_project_id": "P", "snapshot_date": "2021-01-31", "revised_cost_cr": 100.0, "cumulative_expenditure_cr": 20.0, "schedule_slippage_days": 0.0},
        {"canonical_project_id": "P", "snapshot_date": "2021-02-28", "revised_cost_cr": 105.0, "cumulative_expenditure_cr": 30.0, "schedule_slippage_days": 10.0},
        {"canonical_project_id": "P", "snapshot_date": "2021-03-31", "revised_cost_cr": 115.0, "cumulative_expenditure_cr": 45.0, "schedule_slippage_days": 25.0},
        {"canonical_project_id": "P", "snapshot_date": "2021-04-30", "revised_cost_cr": 130.0, "cumulative_expenditure_cr": 65.0, "schedule_slippage_days": 50.0},
    ])


def test_exp12_features_are_as_of_safe_when_future_reports_are_added():
    history = _history()
    before = engineer_history(history)
    extended = pd.concat([history, pd.DataFrame([{
        "canonical_project_id": "P",
        "snapshot_date": "2021-05-31",
        "revised_cost_cr": 9999.0,
        "cumulative_expenditure_cr": 9999.0,
        "schedule_slippage_days": 9999.0,
    }])], ignore_index=True)
    after = engineer_history(extended)
    cutoff = pd.Timestamp("2021-04-30")
    before_rows = before[pd.to_datetime(before.snapshot_date).le(cutoff)].reset_index(drop=True)
    after_rows = after[pd.to_datetime(after.snapshot_date).le(cutoff)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_rows[EXP12_FEATURES], after_rows[EXP12_FEATURES])


def test_exp12_enrichment_preserves_exact_supervised_rows():
    history = _history()
    supervised = history.iloc[[1, 3]].copy()
    supervised["completion_year"] = [2021, 2021]
    supervised["sample_weight"] = [0.5, 0.5]
    enriched = enrich_rows(supervised, history)
    assert len(enriched) == len(supervised)
    assert enriched.canonical_project_id.tolist() == supervised.canonical_project_id.tolist()
    assert enriched.exp12_history_12m.tolist() == [2.0, 4.0]
    assert enriched.exp12_expenditure_velocity_3m.notna().all()
    assert enriched.exp12_slippage_velocity_3m.notna().all()


def test_exp12_is_registered_as_sequence_12_cost_only_challenger():
    adapter = get_experiment_adapter("exp_12")
    assert adapter.sequence == 12
    assert adapter.scope == "cost"
    assert adapter.name == "Trajectory-enhanced cost forecasting"
    assert adapter.module.__name__.endswith("adapter_exp12")


class _CostDelayAdapter:
    @staticmethod
    def predict_project(row, state):
        return {"predicted_cost_overrun": 20.0, "predicted_delay_days": 80.0}


COST_DELAY_ADAPTER = SimpleNamespace(
    experiment_id="exp_delay_test",
    sequence=100,
    name="Cost + delay test",
    scope="cost_delay",
    module=_CostDelayAdapter,
)


def test_compare_pipeline_reveals_cost_and_delay_errors(monkeypatch):
    comparison._COMPARISON_SESSIONS.clear()
    comparison._COMPARISON_SESSIONS["cmp-delay"] = {
        "production_session_id": "prod-session",
        "production_run_id": "prod-run",
        "dataset_fingerprint": "dataset-sha",
        "adapter_id": "exp_delay_test",
        "experiment_run_id": "exp-run",
        "runtime_state": {},
        "comparable_indices": {7},
        "overall": {},
        "candidate_predictions": {},
    }
    monkeypatch.setattr(comparison, "get_experiment_adapter", lambda experiment_id=None: COST_DELAY_ADAPTER)
    row = pd.Series({"record_index": 7})
    prod_session = {"predictions": {}}
    monkeypatch.setattr(comparison.simulation, "_session", lambda session_id: prod_session)
    monkeypatch.setattr(comparison.simulation, "_session_row", lambda session, record_index: row)

    def fake_predict(session_id, record_index):
        prod_session["predictions"][7] = {"predicted_cost_overrun": 25.0, "predicted_delay_days": 100.0}
        return {
            "record_index": 7,
            "predicted_cost_overrun": 25.0,
            "predicted_delay_days": 100.0,
            "predicted_risk": "MEDIUM",
        }

    monkeypatch.setattr(comparison.simulation, "predict_custom", fake_predict)
    monkeypatch.setattr(comparison.simulation, "reveal_custom", lambda session_id, record_index: {
        "record_index": 7,
        "actual_cost_overrun": 18.0,
        "actual_delay_days": 70.0,
        "cost_error_absolute_pp": 7.0,
        "delay_error_absolute_days": 30.0,
    })

    prediction = comparison.predict_comparison("cmp-delay", 7)
    assert prediction["comparison"]["experiment"]["predicted_delay_days"] == 80.0
    assert prediction["comparison"]["delay_prediction_difference_days"] == -20.0

    actual = comparison.reveal_comparison("cmp-delay", 7)
    reveal = actual["comparison"]
    assert reveal["experiment_cost_error_absolute_pp"] == 2.0
    assert reveal["experiment_delay_error_absolute_days"] == 10.0
    assert reveal["experiment_better_cost_for_project"] is True
    assert reveal["experiment_better_delay_for_project"] is True
