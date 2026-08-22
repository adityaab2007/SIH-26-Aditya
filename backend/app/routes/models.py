from fastapi import APIRouter
from backend.app.services.model_service import model_table, global_importances
from backend.app.services.validation_service import validation_payload, validation_report

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/metrics")
def metrics():
    return model_table()

@router.get("/importance")
def importance():
    return global_importances()

@router.get("/validation")
def validation(model: str | None = None):
    return validation_report(model)

@router.get("/prediction-validation")
def prediction_validation(model: str | None = None, limit: int = 100):
    return validation_payload(model, limit)
