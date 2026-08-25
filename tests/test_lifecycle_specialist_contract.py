import numpy as np
import pandas as pd

from backend.app.ml.experiments.lifecycle_specialists import (
    improvement_percent, predict_with_specialist, select_lifecycle_stage,
)


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


def test_improvement_preserves_regressions():
    assert improvement_percent(10, 7.5) == 25
    assert improvement_percent(10, 12.5) == -25
