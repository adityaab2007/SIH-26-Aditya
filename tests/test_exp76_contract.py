from backend.app.ml.experiments.adapter_exp76 import CONFIG, EXPERIMENT_ID

def test_exp76_identity_and_strategy():
    assert EXPERIMENT_ID == "exp_76"
    assert CONFIG.strategy == "medium_project_specialist"
