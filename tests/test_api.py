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
    risk = payload.get("risk_model", payload.get("risk_classification"))
    assert {"accuracy", "precision", "recall", "f1"}.issubset(risk)
    rows = client.get("/api/models/prediction-validation?limit=5")
    assert rows.status_code == 200
    first = rows.json()["items"][0]
    assert {"project_id", "project_name", "predicted_cost_overrun", "actual_cost_overrun", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error"}.issubset(first)


def test_real_paimana_model_simulation_contract():
    versions = client.get("/api/model-simulations")
    assert versions.status_code == 200
    payload = versions.json()
    assert {"2001_2015", "2015_2021"}.issubset({item["key"] for item in payload["items"]})
    assert payload["data_years"]
    simulation = client.post("/api/model-simulations/2001_2015/run")
    assert simulation.status_code == 200
    payload = simulation.json()
    assert payload["metrics"]["metadata"]["data_source"].startswith("Official PAIMANA")
    first = payload["items"][0]
    assert {"predicted_cost_overrun", "actual_cost_overrun", "predicted_delay_days", "actual_delay_days", "shap_explanation"}.issubset(first)


def test_judge_controlled_backtest_hides_actual_until_reveal():
    trained = client.post("/api/model-simulations/custom/train", json={"start_year": 2001, "end_year": 2015})
    assert trained.status_code == 200
    training = trained.json()
    assert training["training_samples"] >= 12
    assert training["actual_outcomes_sent_to_browser"] is False
    assert training["eligible_test_years"]
    assert len(training["training_fingerprint_sha256"]) == 64

    test_year = training["eligible_test_years"][0]["year"]
    assert test_year > training["training_end"]
    projects = client.get(f"/api/model-simulations/custom/{training['session_id']}/projects?year={test_year}")
    assert projects.status_code == 200
    project_payload = projects.json()
    assert project_payload["actual_outcomes_sent_to_browser"] is False
    selected = project_payload["items"][0]
    assert "actual_cost_overrun" not in selected
    assert "actual_delay_days" not in selected
    assert "completion_date" not in selected

    selection = {"record_index": selected["record_index"]}
    prediction = client.post(f"/api/model-simulations/custom/{training['session_id']}/predict", json=selection)
    assert prediction.status_code == 200
    predicted = prediction.json()
    assert predicted["audit"]["project_excluded_from_training"] is True
    assert predicted["audit"]["actual_outcomes_sent_to_browser"] is False
    assert "actual_cost_overrun" not in predicted
    assert "actual_delay_days" not in predicted

    reveal = client.post(f"/api/model-simulations/custom/{training['session_id']}/reveal", json=selection)
    assert reveal.status_code == 200
    actual = reveal.json()
    assert {"actual_cost_overrun", "actual_delay_days", "cost_error_absolute_pp", "delay_error_absolute_days", "source_url"}.issubset(actual)
    assert actual["cost_error_absolute_pp"] >= 0
    assert actual["delay_error_absolute_days"] >= 0
