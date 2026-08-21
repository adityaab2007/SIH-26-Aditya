from fastapi import APIRouter, HTTPException
from backend.app.services.history_service import available_projects, replay

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("")
def list_history():
    return {"items": available_projects()}

@router.get("/{code}")
def history(code: str):
    try:
        return replay(code)
    except KeyError:
        raise HTTPException(404, "Historical project not found")
