from __future__ import annotations

from backend.app.services.data_service import projects_df
from backend.app.services.portfolio_service import portfolio_rows


def answer(query: str) -> dict:
    q = (query or "").lower().strip()
    preds = portfolio_rows()
    pred_map = {p["project_code"]: p for p in preds}
    df = projects_df().copy()

    if any(w in q for w in ["riskiest", "highest risk", "critical", "priority"]):
        top = sorted(preds, key=lambda x: x["priority_score"], reverse=True)[:5]
        return {
            "answer": "The highest baseline review-priority projects in the current real-data subset are listed below.",
            "items": [{"project": p["project_name"], "code": p["project_code"], "score": p["priority_score"]} for p in top],
        }
    if "cost overrun" in q or "cost escalation" in q:
        top = df[df["cost_escalation_pct"].notna()].sort_values("cost_escalation_pct", ascending=False).head(5)
        return {
            "answer": "These projects have the largest observed cost escalation versus original approved cost in the current subset.",
            "items": [{"project": r.project_name, "code": str(r.project_code), "value": round(float(r.cost_escalation_pct), 1)} for r in top.itertuples()],
        }
    if "delay" in q or "schedule" in q:
        top = df[df["schedule_extension_days"].notna()].sort_values("schedule_extension_days", ascending=False).head(5)
        return {
            "answer": "These projects have the largest observed extension between original and revised completion dates in the current subset.",
            "items": [{"project": r.project_name, "code": str(r.project_code), "value": int(r.schedule_extension_days)} for r in top.itertuples()],
        }
    if "sector" in q:
        grouped = df.groupby("sector").agg(projects=("project_code", "count"), original_cost=("original_cost_cr", "sum")).sort_values("original_cost", ascending=False).head(7)
        return {
            "answer": "Largest represented sectors by original project cost in this curated PAIMANA subset:",
            "items": [{"sector": idx, "projects": int(r.projects), "original_cost_cr": round(float(r.original_cost), 1)} for idx, r in grouped.iterrows()],
        }
    return {
        "answer": "I can answer local analytics questions such as ‘highest risk projects’, ‘largest cost overruns’, ‘largest schedule delays’, or ‘compare sectors’. This demo assistant reads the project's analytics directly; it does not invent an LLM answer.",
        "items": [],
    }
