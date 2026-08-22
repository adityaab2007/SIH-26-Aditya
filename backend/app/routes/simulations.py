from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.simulation_service import available_versions, candidate_projects, run, run_project

router = APIRouter(prefix="/api/model-simulations", tags=["model-simulations"])


@router.get("")
def list_versions():
    return {"items": available_versions()}


@router.get("/{version}/projects")
def list_projects(version: str):
    try:
        return {"items": candidate_projects(version)}
    except KeyError:
        raise HTTPException(404, "Unknown model version")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))


class SimulationRequest(BaseModel):
    record_index: int | None = None


@router.post("/{version}/run")
def run_simulation(version: str, request: SimulationRequest | None = None):
    try:
        return run_project(version, request.record_index) if request and request.record_index is not None else run(version)
    except KeyError:
        raise HTTPException(404, "Unknown model version")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))
