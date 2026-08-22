from fastapi import APIRouter, HTTPException

from backend.app.services.simulation_service import available_versions, run

router = APIRouter(prefix="/api/model-simulations", tags=["model-simulations"])


@router.get("")
def list_versions():
    return {"items": available_versions()}


@router.post("/{version}/run")
def run_simulation(version: str):
    try:
        return run(version)
    except KeyError:
        raise HTTPException(404, "Unknown model version")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, str(exc))
