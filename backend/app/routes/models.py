from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.services.model_service import model_table, global_importances
from backend.app.services.validation_service import rolling_validation_report, validation_payload, validation_report
from backend.app.ml.real_time_windows import retrain
from backend.app.services.monthly_prediction_service import lifecycle_comparison, forecast_evolution

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

@router.post("/retrain")
def retrain_model(payload: TrainingRange):
    try:
        return retrain(payload.start_year, payload.end_year)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@router.get("/validation")
def validation(model_version: str | None = None, model: str | None = None):
    return validation_report(model_version or model)

@router.get("/prediction-validation")
def prediction_validation(limit: int = 100, model_version: str | None = None, model: str | None = None):
    return validation_payload(limit, model_version or model)


@router.get("/rolling-validation")
def rolling_validation(model_version: str | None = None, model: str | None = None):
    return rolling_validation_report(model_version or model)


@router.get("/monthly-lifecycle-comparison")
def monthly_lifecycle_comparison():
    return lifecycle_comparison()


@router.get("/monthly-lifecycle-evolution/{project_id}")
def monthly_lifecycle_evolution(project_id: str, window: str = "2015_2021"):
    try:
        return forecast_evolution(project_id, window)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
