from fastapi import APIRouter

from app.ml.validation import regression_metrics

router = APIRouter(prefix="/models", tags=["Model Validation"])


@router.get("/validation")
def validation_summary():
    """Returns model validation summary for SIH26103 dashboard."""
    return {
        "cost_model": {
            "metric": "R2",
            "value": None,
            "status": "pending training evaluation"
        },
        "delay_model": {
            "metric": "F1",
            "value": None,
            "status": "pending training evaluation"
        },
        "method": "temporal backtesting"
    }
