import numpy as np
import pandas as pd

from backend.app.ml.production_u1_delay_baseline import (
    U1DelayResidualProductionModel,
    _design,
)


class _BaseDelay:
    features = ["schedule_slippage_days", "sector", "implementing_agency"]
    model_features = ["schedule_slippage_days", "exp58_delay_hier_prior", "exp58_group_support"]
    weights = {"lightgbm": 1.0}
    calibration = {"dummy": True}
    fallback_model = None

    def _enrich(self, frame):
        out = frame.copy()
        out["exp58_delay_hier_prior"] = 100.0
        out["exp58_group_support"] = 5.0
        return out

    def predict(self, frame):
        return np.full(len(frame), 400.0)


class _Booster:
    def predict(self, frame):
        # Deliberately exceed the cap so the production wrapper must bound it.
        return np.full(len(frame), -80.0)


def test_u1_delay_wrapper_keeps_base_and_applies_bounded_correction():
    model = U1DelayResidualProductionModel(
        base_model=_BaseDelay(),
        booster=_Booster(),
        booster_features=["production_prediction", "schedule_slippage_days", "exp58_delay_hier_prior"],
        medians={"production_prediction": 400.0, "schedule_slippage_days": 20.0, "exp58_delay_hier_prior": 100.0},
        correction_cap=30.0,
        input_features=["schedule_slippage_days", "sector", "implementing_agency"],
    )
    frame = pd.DataFrame({"schedule_slippage_days": [10.0, 25.0], "sector": ["road", "rail"], "implementing_agency": ["a", "b"]})
    pred = model.predict(frame)
    assert np.allclose(pred, [370.0, 370.0])
    assert model.features == ["schedule_slippage_days", "sector", "implementing_agency"]


def test_u1_design_uses_training_only_medians_and_finite_fallback():
    train = pd.DataFrame(
        {
            "production_prediction": [100.0, 200.0, 300.0],
            "schedule_slippage_days": [10.0, np.nan, 30.0],
            "exp58_group_support": [np.nan, np.nan, np.nan],
        }
    )
    score = pd.DataFrame(
        {
            "production_prediction": [150.0],
            "schedule_slippage_days": [np.nan],
            "exp58_group_support": [np.inf],
        }
    )
    cols, medians, x_train, x_score = _design(train, score)
    assert cols == ["production_prediction", "schedule_slippage_days", "exp58_group_support"]
    assert medians["schedule_slippage_days"] == 20.0
    assert medians["exp58_group_support"] == 0.0
    assert np.isfinite(x_train.to_numpy()).all()
    assert np.isfinite(x_score.to_numpy()).all()
    assert x_score.loc[0, "schedule_slippage_days"] == 20.0
    assert x_score.loc[0, "exp58_group_support"] == 0.0
