from __future__ import annotations
import pandas as pd
from backend.app.services.data_service import get_project, projects_df
from backend.app.services.explanation_service import local_shap
from backend.app.services.model_service import best_model_info, predict, predict_proba

def _exposure_percentile(cost):
    if cost is None or pd.isna(cost): return .4
    vals=projects_df()["original_cost_cr"].dropna().astype(float); return float((vals<=float(cost)).mean())

def project_prediction(code: str, override: dict | None=None, include_explanations: bool=True) -> dict:
    row=get_project(code).copy()
    if override:
        for key,value in override.items():
            if key in row.index and value is not None: row[key]=value
    original=float(row["original_cost_cr"]) if pd.notna(row.get("original_cost_cr")) else None; revised=float(row["revised_cost_cr"]) if pd.notna(row.get("revised_cost_cr")) else None; exp=float(row["expenditure_cr"]) if pd.notna(row.get("expenditure_cr")) else None
    row["cost_escalation_pct"]=((revised-original)/original*100) if revised is not None and original else None; row["expenditure_to_original_pct"]=(exp/original*100) if exp is not None and original else None; basis=revised or original; row["financial_progress_pct"]=(exp/basis*100) if exp is not None and basis else None
    snapshot=pd.to_datetime(row.get("snapshot_date"),errors="coerce"); original_end=pd.to_datetime(row.get("original_end_date"),errors="coerce"); revised_end=pd.to_datetime(row.get("revised_end_date"),errors="coerce"); row["days_to_original_deadline"]=(original_end-snapshot).days if pd.notna(original_end) and pd.notna(snapshot) else None; row["schedule_extension_days"]=(revised_end-original_end).days if pd.notna(revised_end) and pd.notna(original_end) else None
    frame=pd.DataFrame([row]); schedule_prob=float(predict_proba("schedule_classifier",frame)[0]); cost_prob=float(predict_proba("cost_classifier",frame)[0]); schedule_days=float(predict("schedule_regressor",frame)[0]); cost_pct=float(predict("cost_regressor",frame)[0]); exposure=_exposure_percentile(row.get("original_cost_cr")); priority=max(0,min(100,100*(.45*schedule_prob+.35*cost_prob+.20*exposure))); level="critical" if priority>=80 else "high" if priority>=65 else "medium" if priority>=45 else "low"; complete=sum(pd.notna(row.get(c)) for c in ["revised_cost_cr","expenditure_cr","physical_progress_pct","revised_end_date"])/4; confidence="high" if complete>=.75 else "medium" if complete>=.5 else "low"
    return {"project_code":str(row["project_code"]),"project_name":row["project_name"],"model_scope":"real-data baseline overrun intelligence","schedule_risk_probability":round(schedule_prob,4),"cost_risk_probability":round(cost_prob,4),"estimated_schedule_extension_days":round(schedule_days,1),"estimated_cost_escalation_pct":round(cost_pct,2),"priority_score":round(priority,1),"priority_level":level,"confidence":confidence,"exposure_percentile":round(exposure,4),"best_models":{"schedule_classifier":best_model_info("schedule_classifier")["model"],"cost_classifier":best_model_info("cost_classifier")["model"],"schedule_regressor":best_model_info("schedule_regressor")["model"],"cost_regressor":best_model_info("cost_regressor")["model"]},"schedule_drivers":local_shap("schedule_classifier",frame) if include_explanations else [],"cost_drivers":local_shap("cost_classifier",frame) if include_explanations else [],"observed":{"schedule_extension_days":None if pd.isna(row.get("schedule_extension_days")) else round(float(row["schedule_extension_days"]),1),"cost_escalation_pct":None if pd.isna(row.get("cost_escalation_pct")) else round(float(row["cost_escalation_pct"]),2),"financial_progress_pct":None if pd.isna(row.get("financial_progress_pct")) else round(float(row["financial_progress_pct"]),1),"physical_progress_pct":None if pd.isna(row.get("physical_progress_pct")) else round(float(row["physical_progress_pct"]),1)}}
