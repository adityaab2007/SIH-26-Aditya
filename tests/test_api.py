from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_temporal_forecast_contract_and_actual_factors():
    response = client.get("/api/projects/701263/forecast")
    assert response.status_code == 200
    data = response.json()
    assert {"project_name", "current_status", "predicted_cost_overrun_percentage", "predicted_delay_days", "risk_score", "risk_level", "explanation"}.issubset(data)
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert data["predicted_delay_days"] >= 0
    assert data["cost_factors"] or data["delay_factors"]
    assert all("feature" in factor and "impact" in factor for factor in data["explanation"])


def test_model_performance_and_history_remain_available():
    metrics = client.get("/api/models/metrics")
    assert metrics.status_code == 200
    assert "cost_model" in metrics.json()
    history = client.get("/api/history/705728")
    assert history.status_code == 200
    assert history.json()["snapshots"]
