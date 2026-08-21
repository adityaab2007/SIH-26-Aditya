from fastapi import APIRouter
from backend.app.services.portfolio_service import summary, portfolio_rows

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

@router.get("/summary")
def portfolio_summary():
    return summary()

@router.get("/risk")
def portfolio_risk(limit: int = 20):
    rows = sorted(portfolio_rows(), key=lambda x: x["priority_score"], reverse=True)
    return {"items": rows[: max(1, min(limit, 100))]}
