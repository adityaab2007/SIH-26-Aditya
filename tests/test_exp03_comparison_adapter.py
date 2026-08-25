from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.experiments import adapter_exp03
from backend.app.ml.experiments.adapters import default_experiment_adapter, get_experiment_adapter


class _Model:
    def predict(self, frame):
        return np.array([4.5] * len(frame), dtype=float)


def test_exp03_is_discovered_as_registered_challenger():
    adapter = get_experiment_adapter("exp_03")
    assert adapter.experiment_id == "exp_03"
    assert adapter.sequence == 3
    assert adapter.name == "Remaining-overrun forecasting"
    assert default_experiment_adapter().experiment_id == "exp_03"


def test_exp03_adapter_filters_anchor_and_reconstructs_final_prediction():
    frame = pd.DataFrame([
        {"record_index": 1, "cost_escalation_percentage": 10.0},
        {"record_index": 2, "cost_escalation_percentage": np.nan},
    ])
    comparable = adapter_exp03.filter_comparable_rows(frame, {})
    assert comparable.record_index.tolist() == [1]

    row = pd.Series({"cost_escalation_percentage": 10.0, "feature": 2.0})
    result = adapter_exp03.predict_project(row, {"model": _Model(), "features": ["feature"]})
    assert result["current_observed_cost_escalation"] == 10.0
    assert result["predicted_remaining_cost_overrun"] == 4.5
    assert result["predicted_cost_overrun"] == 14.5
