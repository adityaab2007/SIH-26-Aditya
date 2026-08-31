from backend.app.ml.experiments.adapter_exp77 import CONFIG, EXPERIMENT_ID

def test_exp77_identity_and_strategy():
    assert EXPERIMENT_ID == "exp_77"
    assert CONFIG.strategy == "early_financial_surrogate"
