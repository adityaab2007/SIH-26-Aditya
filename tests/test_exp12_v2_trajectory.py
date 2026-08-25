from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments.trajectory_exp12_v2 import engineer_history


def _history() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-31", "2020-04-30", "2020-07-31", "2020-10-31", "2021-01-31"])
    return pd.DataFrame({
        "canonical_project_id": ["P1"] * len(dates),
        "snapshot_date": dates,
        "approved_cost_cr": [1000.0] * len(dates),
        "revised_cost_cr": [1000.0, 1020.0, 1080.0, 1160.0, 1300.0],
        "cumulative_expenditure_cr": [100.0, 180.0, 310.0, 500.0, 720.0],
        "planned_duration_days": [1000.0] * len(dates),
        "schedule_slippage_days": [0.0, 0.0, 60.0, 120.0, 210.0],
        "expected_progress_percentage": [10.0, 20.0, 30.0, 40.0, 50.0],
        "planned_completion_date": pd.to_datetime(["2023-01-01"] * len(dates)),
        "revised_completion_date": pd.to_datetime([None, None, "2023-03-02", "2023-05-01", "2023-07-30"]),
    })


def test_v2_adds_scale_normalized_trajectory_features():
    result = engineer_history(_history())
    last = result.iloc[-1]
    assert np.isfinite(last["exp12_cost_growth_pct_12m"])
    assert np.isfinite(last["exp12_expenditure_ratio_velocity_12m"])
    assert np.isfinite(last["exp12_slippage_ratio_velocity_12m"])
    assert last["exp12_cost_revision_magnitude_12m_pct"] > 0
    assert last["exp12_schedule_revision_magnitude_12m_pct"] > 0
    assert last["exp12_cost_worsening_streak"] >= 1
    assert last["exp12_slippage_worsening_streak"] >= 1


def test_v2_earlier_features_do_not_change_when_future_report_is_appended():
    history = _history()
    before = engineer_history(history)
    future = history.iloc[[-1]].copy()
    future["snapshot_date"] = pd.Timestamp("2022-01-31")
    future["revised_cost_cr"] = 99999.0
    future["cumulative_expenditure_cr"] = 99999.0
    future["schedule_slippage_days"] = 9999.0
    future["revised_completion_date"] = pd.Timestamp("2035-01-01")
    after = engineer_history(pd.concat([history, future], ignore_index=True)).iloc[: len(history)]

    feature_columns = [column for column in before.columns if column.startswith("exp12_")]
    pd.testing.assert_frame_equal(
        before[feature_columns].reset_index(drop=True),
        after[feature_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_schedule_revision_count_tracks_effective_completion_date_changes():
    result = engineer_history(_history())
    assert result.iloc[-1]["exp12_schedule_revisions_12m"] >= 2
