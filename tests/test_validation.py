import pandas as pd
import pytest
from backend.app.ml.temporal_validation import assert_no_target_leakage, chronological_holdout

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
