from fastapi import APIRouter

from backend.app.services.validation_service import (
    backtest_payload,
    explain_project,
    model_comparison,
    validation_summary,
)

router = APIRouter(prefix="/api/models", tags=["model-validation"])


@router.get("/validation")
def validation():
    return validation_summary()


@router.get("/backtest")
def backtest():
    return backtest_payload()


@router.get("/comparison")
def comparison():
    return model_comparison()


@router.get("/explain/{project_code}")
def explain(project_code: str):
    return explain_project(project_code)
