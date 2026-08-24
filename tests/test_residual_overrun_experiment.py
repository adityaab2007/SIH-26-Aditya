from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.residual_overrun_experiment import (
    CURRENT_OVERRUN,
    FINAL_TARGET,
    RESIDUAL_TARGET,
    prepare_common_cost_cohort,
    reconstruct_final_overrun,
)


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append({
            "canonical_project_id": f"TRAIN-{index}",
            "snapshot_date": pd.Timestamp("2014-06-30") + pd.Timedelta(days=index),
            "completion_year": 2014,
            "sample_weight": 1.0,
            CURRENT_OVERRUN: float(index),
            FINAL_TARGET: float(index + 10),
        })
    for index in range(3):
        rows.append({
            "canonical_project_id": f"TEST-{index}",
            "snapshot_date": pd.Timestamp("2018-06-30") + pd.Timedelta(days=index),
            "completion_year": 2018,
            "sample_weight": 1.0,
            CURRENT_OVERRUN: float(20 + index),
            FINAL_TARGET: float(35 + index),
        })
    return pd.DataFrame(rows)


def test_residual_target_is_final_minus_current_on_one_common_cohort():
    train, test = prepare_common_cost_cohort(_frame(), 2001, 2015, 2020)
    assert len(train) == 12
    assert len(test) == 3
    assert train.canonical_project_id.nunique() == 12
    assert test.canonical_project_id.nunique() == 3
    assert np.allclose(train[RESIDUAL_TARGET], train[FINAL_TARGET] - train[CURRENT_OVERRUN])
    assert np.allclose(test[RESIDUAL_TARGET], test[FINAL_TARGET] - test[CURRENT_OVERRUN])


def test_common_cohort_reweights_each_project_to_total_one():
    frame = _frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate["snapshot_date"] = pd.Timestamp("2014-07-31")
    duplicate[CURRENT_OVERRUN] = 2.0
    duplicate[FINAL_TARGET] = 12.0
    duplicate["sample_weight"] = 0.001
    frame = pd.concat([frame, duplicate], ignore_index=True)

    train, _ = prepare_common_cost_cohort(frame, 2001, 2015, 2020)
    project_totals = train.groupby("canonical_project_id").sample_weight.sum()
    assert np.allclose(project_totals.to_numpy(dtype=float), np.ones(len(project_totals)))
    repeated = train[train.canonical_project_id.eq("TRAIN-0")]
    assert len(repeated) == 2
    assert np.allclose(repeated.sample_weight, [0.5, 0.5])


def test_reconstructed_output_is_still_final_cost_overrun():
    current = np.array([20.0, 5.0, 0.0])
    predicted_remaining = np.array([12.0, 7.5, -1.0])
    reconstructed = reconstruct_final_overrun(current, predicted_remaining)
    assert np.allclose(reconstructed, np.array([32.0, 12.5, -1.0]))
