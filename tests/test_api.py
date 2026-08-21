from fastapi.testclient import TestClient
from backend.app.main import app

client=TestClient(app)

def test_health():
    r=client.get('/api/health'); assert r.status_code==200; assert r.json()['status']=='ok'

def test_real_rajasthan_refinery_prediction():
    p=client.get('/api/projects/701263').json()
    assert p['project_name']=='Rajasthan Refinery Project'
    assert round(p['cost_escalation_pct'],1)==84.2
    r=client.get('/api/projects/701263/prediction')
    assert r.status_code==200
    data=r.json()
    assert data['priority_level'] in {'critical','high','medium','low'}
    assert 0 <= data['schedule_risk_probability'] <= 1
    assert data['best_models']['schedule_classifier'] == 'xgboost'
    assert len(data['schedule_drivers']) > 0

def test_portfolio_and_model_metrics():
    s=client.get('/api/portfolio/summary').json()
    assert s['projects']==96
    m=client.get('/api/models/metrics').json()
    assert m['metadata']['dataset_rows']==96
    assert 'catboost' in m['schedule_classifier']

def test_scenario_disclaimer():
    r=client.post('/api/scenario',json={'project_code':'602099','expenditure_cr':236.369})
    assert r.status_code==200
    assert 'not causal' in r.json()['note'].lower()
