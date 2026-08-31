from backend.app.ml.experiments.adapter_exp78 import CONFIG, EXPERIMENT_ID

def test_exp78_identity_and_strategy():
    assert EXPERIMENT_ID == "exp_78"
    assert CONFIG.strategy == "cross_window_consensus"
