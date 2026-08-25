import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.services.model_service import model_table, global_importances
from backend.app.services.validation_service import rolling_validation_report, validation_payload, validation_report
from backend.app.services.lifecycle_retraining_service import retrain_lifecycle
from backend.app.services.lifecycle_run_service import lifecycle_runs
from backend.app.services.monthly_prediction_service import lifecycle_comparison, forecast_evolution, lifecycle_specialist_comparison, lifecycle_specialist_convergence
from backend.app.ml.experiments.lifecycle_specialists import EXPERIMENT_ROOT, STAGES, train_lifecycle_specialists
from backend.app.ml.residual_overrun_experiment import run_residual_overrun_experiment

router = APIRouter(prefix="/api/models", tags=["models"])


class TrainingRange(BaseModel):
    start_year: int
    end_year: int


@router.get("/metrics")
def metrics():
    return model_table()


@router.get("/importance")
def importance():
    return global_importances()


@router.get("/lifecycle-runs")
def lifecycle_run_registry():
    """Return lifecycle model runs that really exist in this checkout/runtime."""
    return lifecycle_runs()


@router.post("/retrain")
def retrain_model(payload: TrainingRange):
    """Retrain the production monthly-lifecycle stack for the selected years."""
    try:
        return retrain_lifecycle(payload.start_year, payload.end_year)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@router.post("/experiments/residual-overrun")
def residual_overrun_experiment(payload: TrainingRange):
    """Run Experiment 3 without replacing the production lifecycle model."""
    try:
        return run_residual_overrun_experiment(payload.start_year, payload.end_year)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@router.post("/experiments/lifecycle-specialists/retrain")
def lifecycle_specialists_retrain(payload: TrainingRange):
    """Train Experiment 4 without replacing the production global model."""
    try:
        from backend.app.services.lifecycle_retraining_service import _training_data
        data, identity, _, max_year = _training_data()
        return train_lifecycle_specialists(payload.start_year, payload.end_year, max_year, data=data, identity=identity)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@router.get("/lifecycle-specialists/{window}/comparison")
def lifecycle_specialists_comparison(window: str):
    try:
        return lifecycle_specialist_comparison(window)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/lifecycle-specialists/{window}/stages")
def lifecycle_specialists_stages(window: str):
    try:
        report = lifecycle_specialist_comparison(window)
        stages = report.get("specialists", {})
        for stage in STAGES:
            path = EXPERIMENT_ROOT / window / stage / "shap_importance.json"
            if path.exists():
                stages.setdefault(stage, {})["feature_importance"] = json.loads(path.read_text())
        return {"window": window, "boundaries": report.get("lifecycle_boundaries"), "stages": stages}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/lifecycle-specialists/{window}/convergence/{project_id}")
def lifecycle_specialists_convergence(window: str, project_id: str, reveal: bool = False):
    try:
        return lifecycle_specialist_convergence(project_id, window, include_actual=reveal)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(404, str(exc))


@router.get("/validation")
def validation(model_version: str | None = None, model: str | None = None):
    selected = model_version or model
    try:
        return validation_report(selected)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/prediction-validation")
def prediction_validation(limit: int = 100, model_version: str | None = None, model: str | None = None):
    selected = model_version or model
    try:
        return validation_payload(limit, selected)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/rolling-validation")
def rolling_validation(model_version: str | None = None, model: str | None = None):
    selected = model_version or model
    try:
        return rolling_validation_report(selected)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/monthly-lifecycle-comparison")
def monthly_lifecycle_comparison():
    return lifecycle_comparison()


@router.get("/monthly-lifecycle-evolution/{project_id}")
def monthly_lifecycle_evolution(project_id: str, window: str = "2015_2021"):
    try:
        return forecast_evolution(project_id, window)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
