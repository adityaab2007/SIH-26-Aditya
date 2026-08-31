from backend.app.ml.experiments.adapter_exp79 import CONFIG, EXPERIMENT_ID

def test_exp79_identity_and_strategy():
    assert EXPERIMENT_ID == "exp_79"
    assert CONFIG.strategy == "uncertainty_shrinkage"
