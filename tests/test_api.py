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
    assert data["current_progress"] == data["current_status"]["physical_progress_percentage"]
    assert data["predicted_delay_months"] >= 0
    assert data["shap_explanation"] == data["explanation"]


def test_model_performance_and_history_remain_available():
    metrics = client.get("/api/models/metrics")
    assert metrics.status_code == 200
    assert "cost_model" in metrics.json()
    history = client.get("/api/history/705728")
    assert history.status_code == 200
    assert history.json()["snapshots"]


def test_portfolio_uses_temporal_models():
    response = client.get("/api/portfolio/risk?limit=3")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 3
    assert all(item["model_scope"] == "temporal cost and delay forecasting" for item in items)


def test_prediction_validation_artifacts_are_served():
    report = client.get("/api/models/validation")
    assert report.status_code == 200
    payload = report.json()
    assert payload["cost_model"]["MAE"] >= 0
    assert {"accuracy", "precision", "recall", "f1"}.issubset(payload["risk_classification"])
    rows = client.get("/api/models/prediction-validation?limit=5")
    assert rows.status_code == 200
    first = rows.json()["items"][0]
    assert {"project_id", "prediction_date", "predicted_cost_overrun", "actual_cost_overrun", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error"}.issubset(first)


def test_real_paimana_model_simulation_contract():
    versions = client.get("/api/model-simulations")
    assert versions.status_code == 200
    assert {item["key"] for item in versions.json()["items"]} == {"2001_2015", "2015_2021"}
    projects = client.get("/api/model-simulations/2001_2015/projects")
    assert projects.status_code == 200
    selected = projects.json()["items"][0]
    simulation = client.post("/api/model-simulations/2001_2015/run", json={"record_index": selected["record_index"]})
    assert simulation.status_code == 200
    payload = simulation.json()
    assert payload["metrics"]["metadata"]["data_source"].startswith("Official PAIMANA")
    assert payload["generated_at"]
    assert payload["item"]["project_name"] == selected["project_name"]
    assert {"predicted_cost_overrun", "actual_cost_overrun", "predicted_delay_days", "actual_delay_days", "shap_explanation"}.issubset(payload["item"])
