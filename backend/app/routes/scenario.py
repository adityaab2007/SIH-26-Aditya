from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from backend.app.services.prediction_service import project_prediction

router = APIRouter(prefix="/api/scenario", tags=["scenario"])

class ScenarioInput(BaseModel):
    project_code: str
    physical_progress_pct: float | None = Field(default=None, ge=0, le=100)
    expenditure_cr: float | None = Field(default=None, ge=0)

@router.post("")
def scenario(body: ScenarioInput):
    try:
        baseline = project_prediction(body.project_code, include_explanations=False)
        override = {k: v for k, v in body.model_dump(exclude={"project_code"}).items() if v is not None}
        changed = project_prediction(body.project_code, override=override, include_explanations=False)
        return {
            "note": "Scenario estimates are model sensitivity outputs, not causal guarantees.",
            "baseline": baseline,
            "scenario": changed,
            "inputs": override,
        }
    except KeyError:
        raise HTTPException(404, "Project not found")
