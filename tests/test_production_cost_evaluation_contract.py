from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.production_cost_baseline import (
    PRODUCTION_COST_EVALUATION_COHORT,
    PRODUCTION_COST_MIN_HISTORY,
    _prediction_rows,
    _production_cost_evaluation_rows,
)


class _ConstantRegressor:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class _ConstantRisk:
    def predict(self, frame):
        return np.array(["MEDIUM"] * len(frame), dtype=object)


def _holdout() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "canonical_project_id": "A",
                "project_name": "Project A",
                "snapshot_date": pd.Timestamp("2022-03-31"),
                "completion_year": 2022,
                "lifecycle_stage": "early",
                "actual_cost_overrun_percentage": 1000.0,
                "actual_delay_days": 100.0,
                "actual_risk": "HIGH",
                "sample_weight": 0.5,
                "exp12_history_12m": 1,
                "feature": 1.0,
            },
            {
                "canonical_project_id": "A",
                "project_name": "Project A",
                "snapshot_date": pd.Timestamp("2022-06-30"),
                "completion_year": 2022,
                "lifecycle_stage": "mid",
                "actual_cost_overrun_percentage": 10.0,
                "actual_delay_days": 100.0,
                "actual_risk": "MEDIUM",
                "sample_weight": 0.5,
                "exp12_history_12m": 2,
                "feature": 1.0,
            },
            {
                "canonical_project_id": "B",
                "project_name": "Project B",
                "snapshot_date": pd.Timestamp("2023-03-31"),
                "completion_year": 2023,
                "lifecycle_stage": "mid",
                "actual_cost_overrun_percentage": 20.0,
                "actual_delay_days": 200.0,
                "actual_risk": "MEDIUM",
                "sample_weight": 0.5,
                "exp12_history_12m": 2,
                "feature": 2.0,
            },
            {
                "canonical_project_id": "B",
                "project_name": "Project B",
                "snapshot_date": pd.Timestamp("2023-06-30"),
                "completion_year": 2023,
                "lifecycle_stage": "late",
                "actual_cost_overrun_percentage": 20.0,
                "actual_delay_days": 200.0,
                "actual_risk": "MEDIUM",
                "sample_weight": 0.5,
                "exp12_history_12m": 3,
                "feature": 2.0,
            },
        ]
    )


def test_production_cost_cohort_matches_exp12_history_filter_and_reweights():
    comparable = _production_cost_evaluation_rows(_holdout())

    assert PRODUCTION_COST_MIN_HISTORY == 2
    assert PRODUCTION_COST_EVALUATION_COHORT == "exp12_comparable_trailing_12m_history"
    assert len(comparable) == 3
    assert comparable.canonical_project_id.nunique() == 2
    assert comparable.groupby("canonical_project_id").sample_weight.sum().to_dict() == {
        "A": 1.0,
        "B": 1.0,
    }


def test_headline_cost_mae_uses_filtered_cohort_not_full_holdout():
    cost_metrics, rows, contract = _prediction_rows(
        _holdout(),
        cost_model=_ConstantRegressor(0.0),
        cost_features=["feature"],
        delay_model=_ConstantRegressor(0.0),
        delay_features=["feature"],
        risk_model=_ConstantRisk(),
        risk_features=["feature"],
    )

    # After filtering and reweighting: project A contributes MAE 10 and project B
    # contributes MAE 20, so project-balanced headline MAE is exactly 15. The
    # excluded first A snapshot has a 1000-point error and must not affect it.
    assert cost_metrics["MAE"] == 15.0
    assert rows.cost_evaluation_eligible.tolist() == [False, True, True, True]
    assert contract["test_projects"] == 2
    assert contract["test_snapshots"] == 3
    assert contract["full_holdout_projects"] == 2
    assert contract["full_holdout_snapshots"] == 4
