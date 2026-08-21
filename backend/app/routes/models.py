from fastapi import APIRouter
from backend.app.services.model_service import model_table, global_importances

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/metrics")
def metrics():
    return model_table()

@router.get("/importance")
def importance():
    return global_importances()
