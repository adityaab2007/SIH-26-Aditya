from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.app.services.simulation_service import available_versions, run
from backend.app.services.lifecycle_simulation_service import (
    available_data_years,
    custom_projects,
    predict_custom,
    reveal_custom,
    train_custom,
)
from backend.app.services.lifecycle_model_comparison_service import (
    LATEST_EXPERIMENT_ID,
    comparison_projects,
    predict_comparison,
    retrain_and_compare,
    reveal_comparison,
)

router = APIRouter(prefix="/api/model-simulations", tags=["model-simulations"])

# PR #26 deliberately keeps Experiment 3 as the pinned manual challenger.
# The comparison service already retrains the current monthly-lifecycle production
# stack first and compares Exp3 against that exact fresh production run.  When a
# future experiment replaces Exp3, this pin can be changed in the experiment PR
# without altering the production model itself.
EXPERIMENT_3_ID = LATEST_EXPERIMENT_ID
EXPERIMENT_3_NAME = "Remaining-overrun forecasting (Experiment 3)"


class TrainingRange(BaseModel):
    start_year: int
    end_year: int
    run_id: str | None = None


class CompareTrainingRange(BaseModel):
    start_year: int
    end_year: int
    experiment_id: str = EXPERIMENT_3_ID


class ProjectSelection(BaseModel):
    record_index: int


@router.get("")
def list_versions():
    comparison_contract = {
        "comparison_experiment_id": EXPERIMENT_3_ID,
        "comparison_experiment_name": EXPERIMENT_3_NAME,
        # Kept for frontend compatibility while PR #26 is still open.
        "latest_experiment_id": EXPERIMENT_3_ID,
    }
    try:
        data_years = available_data_years()
        lifecycle_data_available = True
    except FileNotFoundError as exc:
        data_years = []
        lifecycle_data_available = False
        return {
            "items": available_versions(),
            "data_years": data_years,
            "lifecycle_data_available": lifecycle_data_available,
            "lifecycle_data_unavailable_reason": str(exc),
            **comparison_contract,
        }
    return {
        "items": available_versions(),
        "data_years": data_years,
        "lifecycle_data_available": lifecycle_data_available,
        **comparison_contract,
    }


@router.post("/{version}/run")
def run_simulation(version: str):
    try:
        return run(version)
    except KeyError:
        raise HTTPException(404, "Unknown model version")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


@router.post("/custom/train")
def train_custom_simulation(payload: TrainingRange):
    try:
        return train_custom(payload.start_year, payload.end_year, payload.run_id)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@router.post("/custom/retrain-compare")
def retrain_compare_simulation(payload: CompareTrainingRange):
    """Retrain current lifecycle production and compare it with pinned Experiment 3."""
    if payload.experiment_id != EXPERIMENT_3_ID:
        raise HTTPException(
            409,
            f"PR #26 is pinned to {EXPERIMENT_3_ID} for manual lifecycle-vs-Experiment-3 verification.",
        )
    try:
        return retrain_and_compare(payload.start_year, payload.end_year, EXPERIMENT_3_ID)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(409, str(exc))


@router.get("/custom/{session_id}/projects")
def list_custom_projects(session_id: str, year: int = Query(...)):
    try:
        return custom_projects(session_id, year)
    except KeyError:
        raise HTTPException(404, "Training session not found or expired. Retrain the model and try again.")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/compare/{session_id}/projects")
def list_comparison_projects(session_id: str, year: int = Query(...)):
    try:
        return comparison_projects(session_id, year)
    except KeyError:
        raise HTTPException(404, "Comparison session not found or expired. Retrain & Compare again.")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/custom/{session_id}/predict")
def predict_custom_project(session_id: str, payload: ProjectSelection):
    try:
        return predict_custom(session_id, payload.record_index)
    except KeyError:
        raise HTTPException(404, "Training session not found or expired. Retrain the model and try again.")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/compare/{session_id}/predict")
def predict_comparison_project(session_id: str, payload: ProjectSelection):
    try:
        return predict_comparison(session_id, payload.record_index)
    except KeyError:
        raise HTTPException(404, "Comparison session not found or expired. Retrain & Compare again.")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/custom/{session_id}/reveal")
def reveal_custom_project(session_id: str, payload: ProjectSelection):
    try:
        return reveal_custom(session_id, payload.record_index)
    except KeyError:
        raise HTTPException(404, "Training session not found or expired. Retrain the model and try again.")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/compare/{session_id}/reveal")
def reveal_comparison_project(session_id: str, payload: ProjectSelection):
    try:
        return reveal_comparison(session_id, payload.record_index)
    except KeyError:
        raise HTTPException(404, "Comparison session not found or expired. Retrain & Compare again.")
    except ValueError as exc:
        raise HTTPException(409, str(exc))