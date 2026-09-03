from backend.app.ml.experiments.exp142_dart_dropout import dart_estimator,is_dart_estimator

def test_lightgbm_dart_is_actually_enabled_and_seeded():
    m=dart_estimator("lightgbm",14203,.05,.2);p=m.get_params();assert is_dart_estimator(m);assert p["boosting_type"]=="dart" and p["random_state"]==14203

def test_xgboost_dart_is_actually_enabled_and_seeded():
    m=dart_estimator("xgboost",14203,.1,.5);p=m.get_params();assert is_dart_estimator(m);assert p["booster"]=="dart" and p["rate_drop"]==.1 and p["skip_drop"]==.5 and p["random_state"]==14203

def test_dart_does_not_change_base_feature_contract():
    m=dart_estimator("lightgbm",1);assert "n_estimators" in m.get_params() and m.get_params()["n_estimators"]==240
