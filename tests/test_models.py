from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parents[1]

def test_all_model_families_trained():
    files=list((ROOT/'models').glob('*.joblib'))
    assert len(files)==16, 'training pipeline should reproduce all 16 artifacts locally'
    metrics=json.loads((ROOT/'models/metrics.json').read_text())
    assert metrics['schedule_classifier']['best_model'] in {'logistic_regression','random_forest','xgboost','catboost'}
    assert metrics['cost_classifier']['best_model'] in {'logistic_regression','random_forest','xgboost','catboost'}
    for task in ['schedule_classifier','cost_classifier']:
        for name,m in metrics[task].items():
            if name=='best_model': continue
            assert 0 <= m['roc_auc'] <= 1
    for task in ['schedule_regressor','cost_regressor']:
        for name,m in metrics[task].items():
            if name=='best_model': continue
            assert math.isfinite(m['mae']) and m['mae'] >= 0
