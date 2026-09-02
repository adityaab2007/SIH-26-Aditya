import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp124_continuous_delay_calibration import BASE_INPUTS, FEATURES, SCALES, _design, _model
from backend.app.ml.experiments.scientific_challenger_utils import BASE_25_FEATURES


def test_exp124_freezes_exact_25_feature_base():
    assert len(BASE_25_FEATURES) == 25
    assert len(set(BASE_25_FEATURES)) == 25


def test_exp124_is_low_capacity_spline_median_calibrator_with_zero_scale():
    model = _model()
    assert SCALES[0] == 0.0
    assert model.named_steps["spline"].n_knots == 4
    assert model.named_steps["spline"].degree == 2
    assert model.named_steps["median"].quantile == 0.5
    assert len(FEATURES) <= 10


def test_exp124_design_is_causal_and_future_append_does_not_change_past_rows():
    base = pd.DataFrame({
        "predicted_delay_days": [100.0, 120.0],
        "duration_ratio": [0.4, 0.5],
        "elapsed_duration_days": [40.0, 50.0],
        "planned_duration_days": [100.0, 100.0],
        "physical_progress": [30.0, 40.0],
        "progress_deviation": [-10.0, -10.0],
        "schedule_slippage_days": [5.0, 8.0],
    })
    before = _design(base)[FEATURES].copy()
    future = pd.concat([base, pd.DataFrame({
        "predicted_delay_days": [999.0], "duration_ratio": [2.0],
        "elapsed_duration_days": [200.0], "planned_duration_days": [100.0],
        "physical_progress": [100.0], "progress_deviation": [0.0],
        "schedule_slippage_days": [500.0],
    })], ignore_index=True)
    after = _design(future).iloc[:2][FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)
    assert "actual_delay_days" not in FEATURES
