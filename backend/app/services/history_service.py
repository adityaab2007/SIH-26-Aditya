from __future__ import annotations

import pandas as pd

from backend.app.services.data_service import get_project, history_df
from backend.app.services.prediction_service import project_prediction


def available_projects() -> list[dict]:
    df = history_df()
    return [
        {
            "project_code": str(code),
            "project_name": group.iloc[0]["project_name"],
            "snapshots": int(len(group)),
        }
        for code, group in df.groupby("project_code")
    ]


def replay(code: str) -> dict:
    hist = history_df()[history_df()["project_code"] == str(code)].sort_values("snapshot_date")
    if hist.empty:
        raise KeyError(code)
    master = get_project(code)
    snapshots = []
    for _, r in hist.iterrows():
        override = {
            "snapshot_date": r["snapshot_date"],
            "revised_cost_cr": r["revised_cost_cr"] if pd.notna(r["revised_cost_cr"]) else master["revised_cost_cr"],
            "expenditure_cr": r["expenditure_cr"] if pd.notna(r["expenditure_cr"]) else master["expenditure_cr"],
            "physical_progress_pct": r["physical_progress_pct"] if pd.notna(r["physical_progress_pct"]) else master["physical_progress_pct"],
            "revised_end_date": r["revised_completion_date"] if pd.notna(r["revised_completion_date"]) else master["revised_end_date"],
        }
        pred = project_prediction(code, override=override, include_explanations=False)
        snapshots.append({
            "snapshot_date": r["snapshot_date"].strftime("%Y-%m-%d"),
            "physical_progress_pct": None if pd.isna(r["physical_progress_pct"]) else float(r["physical_progress_pct"]),
            "expenditure_cr": None if pd.isna(r["expenditure_cr"]) else float(r["expenditure_cr"]),
            "revised_cost_cr": None if pd.isna(r["revised_cost_cr"]) else float(r["revised_cost_cr"]),
            "revised_completion_date": None if pd.isna(r["revised_completion_date"]) else r["revised_completion_date"].strftime("%Y-%m-%d"),
            "baseline_priority_score": pred["risk_score"],
            "schedule_risk_probability": pred["risk_score"] / 100,
            "cost_risk_probability": pred["risk_score"] / 100,
            "source_url": r["source_url"],
        })
    return {"project_code": str(code), "project_name": hist.iloc[0]["project_name"], "note": "Historical trajectory uses official snapshots. Baseline model scores are replayed for demonstration and are not claimed as forward-validated forecasts until the full archive is ingested.", "snapshots": snapshots}
