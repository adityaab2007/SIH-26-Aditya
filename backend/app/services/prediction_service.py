"""Current-project forecasts using the active real PAIMANA time-window model."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.real_time_windows import FEATURES, _predict_regressor, active_version, model_dir
from backend.app.services.data_service import get_project
from backend.app.services.simulation_service import _shap_factors_for_model


def _model_inputs(project: pd.Series) -> pd.DataFrame:
    approved_cost = pd.to_numeric(project.get("original_cost_cr"), errors="coerce")
    planned_date = pd.to_datetime(project.get("original_end_date"), errors="coerce")
    if pd.isna(approved_cost) or approved_cost <= 0:
        raise ValueError("Forecast unavailable: approved project cost is missing or not positive.")
    if pd.isna(planned_date):
        raise ValueError("Forecast unavailable: planned completion date is missing.")
    # The current PAIMANA monitoring extract does not publish implementing
    # agency for every row. Use its explicit categorical missing value, not a
    # fabricated numerical fallback.
    return pd.DataFrame([{
        "approved_cost_cr": float(approved_cost),
        "sector": str(project.get("sector") or "Not reported"),
        "implementing_agency": str(project.get("implementing_agency") or "Not reported"),
        "planned_commissioning_year": int(planned_date.year),
    }])


def project_forecast(code: str) -> dict:
    project = get_project(code)
    version = active_version()
    if not version:
        raise ValueError("No real PAIMANA time-window model is available. Retrain a model first.")
    target = model_dir(version)
    metadata = json.loads((target / "metadata.json").read_text())
    X = _model_inputs(project)
    cost_model = joblib.load(target / "cost_model.pkl")
    delay_model = joblib.load(target / "delay_model.pkl")
    risk_model = joblib.load(target / "risk_model.pkl")
    cost = float(_predict_regressor(cost_model, X)[0])
    # A live forecast reports additional delay days.  The backtest retains
    # signed early/late completion outcomes, while a negative live estimate is
    # represented as no predicted delay rather than a negative delay duration.
    delay = max(0.0, float(_predict_regressor(delay_model, X, delay_target=True)[0]))
    risk_prediction = int(np.asarray(risk_model.predict(X), dtype=int).reshape(-1)[0])
    probability = float(np.asarray(risk_model.predict_proba(X))[0][1]) if hasattr(risk_model, "predict_proba") and len(np.asarray(risk_model.predict_proba(X))[0]) > 1 else float(risk_prediction)
    planned = pd.to_datetime(project.original_end_date, errors="coerce")
    progress = pd.to_numeric(project.get("physical_progress_pct"), errors="coerce")
    revised = pd.to_numeric(project.get("revised_cost_cr"), errors="coerce")
    expenditure = pd.to_numeric(project.get("expenditure_cr"), errors="coerce")
    factors = _shap_factors_for_model(cost_model, X.iloc[0])
    current_status = {
        "snapshot_month": pd.to_datetime(project.snapshot_date).strftime("%Y-%m-%d"),
        "physical_progress_percentage": None if pd.isna(progress) else round(float(progress), 1),
        "current_estimated_cost": None if pd.isna(revised) else round(float(revised), 2),
        "expenditure_cr": None if pd.isna(expenditure) else round(float(expenditure), 2),
        "planned_completion_date": planned.strftime("%Y-%m-%d"),
        "progress_delay_percentage_points": None,
    }
    return {
        "project_id": str(project.project_code), "project_name": project.project_name,
        "current_status": current_status, "predicted_cost_overrun_percentage": round(cost, 2),
        "predicted_delay_days": round(delay, 1), "predicted_cost_overrun": round(cost, 2),
        "current_progress": current_status["physical_progress_percentage"],
        "predicted_delay_months": round(delay / 30.4375, 1),
        "risk_score": round(probability * 100, 1), "risk_level": "HIGH" if risk_prediction else "LOW",
        "explanation": factors, "shap_explanation": factors, "cost_factors": factors, "delay_factors": _shap_factors_for_model(delay_model, X.iloc[0]),
        "best_models": {"cost": metadata.get("algorithms", {}).get("cost", "registered model"), "delay": metadata.get("algorithms", {}).get("delay", "registered model")},
        "model_scope": f"Real PAIMANA time-window model {version}; final expenditure and completion date are not inference inputs.",
    }


def project_prediction(code: str, override: dict | None = None, include_explanations: bool = True) -> dict:
    return project_forecast(code)
