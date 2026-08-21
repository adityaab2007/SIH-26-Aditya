from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.services.data_service import get_project, projects_df, row_to_dict


def peer_benchmark(code: str, limit: int = 6) -> dict:
    project = get_project(code)
    df = projects_df().copy()
    peers = df[(df["sector"] == project["sector"]) & (df["project_code"] != str(code))].copy()
    if peers.empty:
        peers = df[df["project_code"] != str(code)].copy()
    base = max(float(project["original_cost_cr"]), 1.0)
    peers["cost_distance"] = abs(np.log1p(peers["original_cost_cr"].astype(float)) - np.log1p(base))
    peers = peers.sort_values("cost_distance").head(limit)

    def med(col):
        x = peers[col].dropna()
        return None if x.empty else round(float(x.median()), 2)

    return {
        "sector": project["sector"],
        "peer_count": int(len(peers)),
        "medians": {
            "original_cost_cr": med("original_cost_cr"),
            "cost_escalation_pct": med("cost_escalation_pct"),
            "schedule_extension_days": med("schedule_extension_days"),
            "financial_progress_pct": med("financial_progress_pct"),
            "physical_progress_pct": med("physical_progress_pct"),
        },
        "peers": [
            {
                "project_code": str(r["project_code"]),
                "project_name": r["project_name"],
                "original_cost_cr": None if pd.isna(r["original_cost_cr"]) else round(float(r["original_cost_cr"]), 2),
                "cost_escalation_pct": None if pd.isna(r["cost_escalation_pct"]) else round(float(r["cost_escalation_pct"]), 2),
                "schedule_extension_days": None if pd.isna(r["schedule_extension_days"]) else round(float(r["schedule_extension_days"]), 1),
            }
            for _, r in peers.iterrows()
        ],
    }
