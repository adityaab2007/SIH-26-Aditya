from backend.app.ml.experiments.adapter_exp78 import (
    CONFIG,
    EXPERIMENT_ID,
    _production_cost_family,
)


def test_exp78_identity_and_strategy():
    assert EXPERIMENT_ID == "exp_78"
    assert CONFIG.strategy == "cross_window_consensus"


def test_exp78_unwraps_residual_calibrated_production_cost_model():
    class LGBMRegressor:
        pass

    class Pipeline:
        def __init__(self):
            self.named_steps = {"model": LGBMRegressor()}

    class ResidualCalibratedCostModel:
        def __init__(self, model):
            self.model = model

    bundle = {"cost": ResidualCalibratedCostModel(Pipeline())}
    assert _production_cost_family(bundle) == "lightgbm"
