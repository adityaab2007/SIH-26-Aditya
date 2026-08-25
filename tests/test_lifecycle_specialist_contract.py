import numpy as np
import pandas as pd

from backend.app.ml.experiments.lifecycle_specialists import (
    improvement_percent, predict_with_specialist, renormalize_stage_weights, select_lifecycle_stage,
)
from backend.app.ml.monthly_lifecycle import IMPROVED_EARLY_FEATURES, as_of_feature_evidence, engineer_as_of_features


def test_lifecycle_boundaries_and_overrun_stage():
    assert [select_lifecycle_stage(value) for value in [0, .2499, .25, .4999, .5, .7499, .75, 1, 1.5]] == [
        "early", "early", "early_mid", "early_mid", "late_mid", "late_mid", "late", "late", "late"
    ]
    assert select_lifecycle_stage(np.nan) is None


class _Model:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return np.array([self.value])


def test_routing_calls_exactly_one_specialist_without_averaging():
    models = {stage: {"cost": _Model(index), "delay": _Model(index + 10), "selected_algorithms": {"cost": "test", "delay": "test"}} for index, stage in enumerate(("early", "early_mid", "late_mid", "late"), 1)}
    result = predict_with_specialist(pd.Series({"duration_ratio": .62, "approved_cost_cr": 1}), {"report": {"features": ["approved_cost_cr"]}, "bundles": models})
    assert result["lifecycle_stage"] == "late_mid"
    assert result["cost"]["predicted_final_overrun_percentage"] == 3
    assert models["late_mid"]["cost"].calls == 1
    assert sum(item["cost"].calls for stage, item in models.items() if stage != "late_mid") == 0


def test_missing_stage_uses_explicit_global_fallback():
    result = predict_with_specialist({"duration_ratio": None}, {"report": {"features": []}, "bundles": {}}, {"cost": {"predicted": 1}, "delay": {"predicted": 2}})
    assert result["specialist_used"] is False
    assert result["fallback_to_global"] is True
    assert result["fallback_reason"] == "missing or invalid lifecycle ratio"


def test_stage_weights_are_renormalized_per_project():
    frame = pd.DataFrame({"canonical_project_id": ["a", "a", "b"], "sample_weight": [.5, .5, 1.0]})
    result = renormalize_stage_weights(frame)
    assert result.groupby("canonical_project_id").sample_weight.sum().to_dict() == {"a": 1.0, "b": 1.0}


def test_improved_feature_lineage_is_explicit_and_as_of_safe():
    evidence = as_of_feature_evidence(IMPROVED_EARLY_FEATURES)
    assert set(IMPROVED_EARLY_FEATURES) <= set(evidence)
    assert all(item["proven"] and "missing_data_behavior" in item for item in evidence.values())


def test_previous_snapshot_features_do_not_look_forward():
    frame = pd.DataFrame({
        "canonical_project_id": ["a", "a"], "snapshot_date": pd.to_datetime(["2020-01-01", "2020-04-01"]),
        "approved_cost_cr": [100, 100], "revised_cost_cr": [100, 120],
        "cumulative_expenditure_cr": [10, 30], "physical_progress": [np.nan, np.nan],
        "planned_start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "approval_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "planned_completion_date": pd.to_datetime(["2021-01-01", "2021-01-01"]),
        "revised_completion_date": pd.to_datetime(["2021-01-01", "2021-02-01"]),
        "project_name": ["A", "A"], "sector": ["Road", "Road"], "project_size_category": ["small", "small"],
        "implementing_agency": ["Agency", "Agency"], "ministry": ["M", "M"],
        "current_schedule_status": ["on", "late"], "identity_verified": [True, True],
        "completion_date": pd.NaT, "actual_delay_days": np.nan, "actual_cost_overrun_percentage": np.nan,
    })
    engineered = engineer_as_of_features(frame, pd.DataFrame())
    assert pd.isna(engineered.iloc[0].cost_change_from_previous_snapshot)
    assert engineered.iloc[1].cost_change_from_previous_snapshot == 20
    assert pd.isna(engineered.iloc[0].expenditure_change_from_previous_snapshot)


def test_improvement_preserves_regressions():
    assert improvement_percent(10, 7.5) == 25
    assert improvement_percent(10, 12.5) == -25
