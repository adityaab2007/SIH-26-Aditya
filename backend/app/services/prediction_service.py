from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.ml.features import TEMPORAL_FEATURES
from backend.app.services.data_service import latest_temporal_snapshot
from backend.app.services.explanation_service import local_shap
from backend.app.services.model_service import best_model_info, predict


def project_forecast(code: str) -> dict:
    row = latest_temporal_snapshot(code).copy()
    frame = pd.DataFrame([row])
    cost = float(predict("cost_model", frame)[0])
    delay = max(0.0, float(predict("delay_model", frame)[0]))
    risk_score = max(0, min(100, 0.55 * min(100, max(0, cost * 2.2)) + 0.25 * min(100, delay / 7) + 0.20 * min(100, row.progress_delay * 2)))
    risk_level = "HIGH" if risk_score >= 65 else "MEDIUM" if risk_score >= 35 else "LOW"
    cost_factors = local_shap("cost_model", frame)
    delay_factors = local_shap("delay_model", frame)
    factors = sorted(cost_factors + delay_factors, key=lambda item: abs(item["impact"]), reverse=True)[:5]
    current_status = {"snapshot_month": row.month.strftime("%Y-%m-%d"), "physical_progress_percentage": round(float(row.physical_progress_percentage), 1), "current_estimated_cost": round(float(row.current_estimated_cost), 2), "planned_completion_date": row.planned_completion_date.strftime("%Y-%m-%d"), "progress_delay_percentage_points": round(float(row.progress_delay), 1)}
    return {
        "project_id": str(row.project_id), "project_name": row.project_name, "current_status": current_status,
        "predicted_cost_overrun_percentage": round(cost, 2), "predicted_delay_days": round(delay, 1),
        "current_progress": current_status["physical_progress_percentage"], "predicted_cost_overrun": round(cost, 2),
        "predicted_delay_months": round(delay / 30.4375, 1),
        "risk_score": round(risk_score, 1), "risk_level": risk_level,
        "explanation": factors, "shap_explanation": factors, "cost_factors": cost_factors, "delay_factors": delay_factors,
        "best_models": {"cost": best_model_info("cost_model")["model"], "delay": best_model_info("delay_model")["model"]},
        "model_scope": "Temporal demonstration forecast trained on documented synthetic longitudinal data; replace with governed PAIMANA/OCMS monthly exports for operational use.",
    }


def project_prediction(code: str, override: dict | None = None, include_explanations: bool = True) -> dict:
    """Compatibility alias for existing callers; new clients use /forecast."""
    return project_forecast(code)
