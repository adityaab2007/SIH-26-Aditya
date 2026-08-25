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
    comparison_projects,
    experiment_catalog,
    predict_comparison,
    retrain_and_compare,
    reveal_comparison,
)

router = APIRouter(prefix="/api/model-simulations", tags=["model-simulations"])


class TrainingRange(BaseModel):
    start_year: int
    end_year: int
    run_id: str | None = None


class CompareTrainingRange(BaseModel):
    start_year: int
    end_year: int
    experiment_id: str | None = None


class ProjectSelection(BaseModel):
    record_index: int


@router.get("")
def list_versions():
    experiments = experiment_catalog()
    try:
        data_years = available_data_years()
        lifecycle_data_available = True
    except FileNotFoundError as exc:
        return {
            "items": available_versions(),
            "data_years": [],
            "lifecycle_data_available": False,
            "lifecycle_data_unavailable_reason": str(exc),
            "comparison_experiments": experiments["items"],
            "active_experiment_id": experiments["active_experiment_id"],
            "active_experiment_name": experiments["active_experiment_name"],
        }
    return {
        "items": available_versions(),
        "data_years": data_years,
        "lifecycle_data_available": lifecycle_data_available,
        "comparison_experiments": experiments["items"],
        "active_experiment_id": experiments["active_experiment_id"],
        "active_experiment_name": experiments["active_experiment_name"],
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
    """Retrain production plus one registered experiment and bind one judge session."""
    try:
        return retrain_and_compare(payload.start_year, payload.end_year, payload.experiment_id)
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
