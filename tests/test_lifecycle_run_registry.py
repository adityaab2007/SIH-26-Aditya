import gzip
import json

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import lifecycle_run_service, validation_service

client = TestClient(app)


def _evaluation(start=2004, end=2020):
    return {
        "window": f"{start}_{end}",
        "metadata": {
            "model_version": f"monthly-{start}-{end}",
            "training_period": [start, end],
            "testing_period": [end + 1, 2025],
            "features_used": ["approved_cost_cr", "schedule_slippage_days"],
            "feature_availability": {"data_quality_score": 97.2, "removed_invalid_feature_count": 2},
        },
        "lifecycle": {
            "metrics": {
                "cost": {"MAE": 29.5},
                "delay": {"MAE": 520.0},
                "risk": {"macro_f1": 0.44},
            }
        },
    }


def test_lifecycle_run_registry_discovers_runtime_windows(tmp_path, monkeypatch):
    root = tmp_path / "monthly_lifecycle"

    summary = root / "2001_2015"
    summary.mkdir(parents=True)
    (summary / "evaluation_results.json").write_text(json.dumps(_evaluation(2001, 2015)))

    complete = root / "2004_2020"
    complete.mkdir(parents=True)
    (complete / "evaluation_results.json").write_text(json.dumps(_evaluation()))
    (complete / "metadata.json").write_text(json.dumps(_evaluation()["metadata"]))
    (complete / "prediction_validation.csv").write_text(
        "canonical_project_id,actual_cost_overrun_percentage,actual_delay_days,predicted_cost_overrun,predicted_delay_days,cost_error,delay_error\n"
        "P-1,20,100,18,90,-2,-10\n"
    )
    for name in ("cost", "delay", "risk"):
        (complete / f"{name}_model.pkl").touch()
    (complete / "run_manifest.json").write_text(json.dumps({"status": "complete", "created_at": "2026-08-24T00:00:00Z"}))

    monkeypatch.setattr(lifecycle_run_service, "MODELS_DIR", tmp_path)
    response = client.get("/api/models/lifecycle-runs")
    assert response.status_code == 200
    payload = response.json()
    by_window = {item["window"]: item for item in payload["items"]}

    assert by_window["2001_2015"]["summary_available"] is True
    assert by_window["2001_2015"]["complete"] is False
    assert by_window["2004_2020"]["complete"] is True
    assert by_window["2004_2020"]["has_validation_rows"] is True
    assert by_window["2004_2020"]["cost_mae"] == 29.5
    assert by_window["2004_2020"]["testing_start"] == 2021


def test_training_marker_prevents_half_written_run_from_looking_complete(tmp_path, monkeypatch):
    root = tmp_path / "monthly_lifecycle" / "2002_2012"
    root.mkdir(parents=True)
    (root / "evaluation_results.json").write_text(json.dumps(_evaluation(2002, 2012)))
    (root / "metadata.json").write_text(json.dumps(_evaluation(2002, 2012)["metadata"]))
    (root / "prediction_validation.csv").write_text("canonical_project_id\nP-1\n")
    for name in ("cost", "delay", "risk"):
        (root / f"{name}_model.pkl").touch()
    (root / ".training").write_text("in progress")

    monkeypatch.setattr(lifecycle_run_service, "MODELS_DIR", tmp_path)
    item = client.get("/api/models/lifecycle-runs").json()["items"][0]
    assert item["in_progress"] is True
    assert item["complete"] is False
    assert item["status"] == "training"


def test_prediction_validation_supports_gzip_rows(tmp_path, monkeypatch):
    root = tmp_path / "monthly_lifecycle" / "2004_2020"
    root.mkdir(parents=True)
    (root / "evaluation_results.json").write_text(json.dumps(_evaluation()))
    csv = (
        "canonical_project_id,project_name,completion_year,actual_cost_overrun_percentage,actual_delay_days,predicted_cost_overrun,predicted_delay_days,cost_error,delay_error\n"
        "P-1,Gzip holdout,2022,20.0,100.0,18.0,90.0,-2.0,-10.0\n"
    )
    with gzip.open(root / "prediction_validation.csv.gz", "wt") as handle:
        handle.write(csv)

    monkeypatch.setattr(validation_service, "MODELS_DIR", tmp_path)
    response = client.get("/api/models/prediction-validation?model=2004_2020")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["project_id"] == "P-1"
    assert item["actual_cost_overrun"] == 20.0
    assert item["completion_year"] == 2022
