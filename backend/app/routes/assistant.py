from pydantic import BaseModel
from fastapi import APIRouter
from backend.app.services.assistant_service import answer

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

class Query(BaseModel):
    query: str

@router.post("/query")
def query(body: Query):
    return answer(body.query)
