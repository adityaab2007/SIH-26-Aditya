from __future__ import annotations
from functools import lru_cache
import pandas as pd
from backend.app.services.data_service import projects_df
from backend.app.services.model_service import best_model_info, predict, predict_proba

@lru_cache(maxsize=1)
def portfolio_rows() -> list[dict]:
    df=projects_df().copy(); schedule_prob=predict_proba("schedule_classifier",df); cost_prob=predict_proba("cost_classifier",df); schedule_days=predict("schedule_regressor",df); cost_pct=predict("cost_regressor",df); exposure=df["original_cost_cr"].rank(method="average",pct=True).fillna(.4).to_numpy(); rows=[]
    for i,(_,r) in enumerate(df.iterrows()):
        priority=max(0,min(100,100*(.45*float(schedule_prob[i])+.35*float(cost_prob[i])+.20*float(exposure[i])))); level="critical" if priority>=80 else "high" if priority>=65 else "medium" if priority>=45 else "low"; complete=sum(pd.notna(r.get(c)) for c in ["revised_cost_cr","expenditure_cr","physical_progress_pct","revised_end_date"])/4; confidence="high" if complete>=.75 else "medium" if complete>=.5 else "low"
        rows.append({"project_code":str(r["project_code"]),"project_name":r["project_name"],"model_scope":"real-data baseline overrun intelligence","schedule_risk_probability":round(float(schedule_prob[i]),4),"cost_risk_probability":round(float(cost_prob[i]),4),"estimated_schedule_extension_days":round(float(schedule_days[i]),1),"estimated_cost_escalation_pct":round(float(cost_pct[i]),2),"priority_score":round(priority,1),"priority_level":level,"confidence":confidence,"exposure_percentile":round(float(exposure[i]),4),"best_models":{"schedule_classifier":best_model_info("schedule_classifier")["model"],"cost_classifier":best_model_info("cost_classifier")["model"],"schedule_regressor":best_model_info("schedule_regressor")["model"],"cost_regressor":best_model_info("cost_regressor")["model"]},"schedule_drivers":[],"cost_drivers":[],"observed":{"schedule_extension_days":None if pd.isna(r.get("schedule_extension_days")) else round(float(r["schedule_extension_days"]),1),"cost_escalation_pct":None if pd.isna(r.get("cost_escalation_pct")) else round(float(r["cost_escalation_pct"]),2),"financial_progress_pct":None if pd.isna(r.get("financial_progress_pct")) else round(float(r["financial_progress_pct"]),1),"physical_progress_pct":None if pd.isna(r.get("physical_progress_pct")) else round(float(r["physical_progress_pct"]),1)}})
    return rows

def summary() -> dict:
    df=projects_df(); preds=portfolio_rows(); levels={k:0 for k in ["critical","high","medium","low"]}
    for p in preds: levels[p["priority_level"]]+=1
    return {"projects":int(len(df)),"original_cost_cr":round(float(df["original_cost_cr"].fillna(0).sum()),2),"current_cost_basis_cr":round(float(df["revised_cost_cr"].fillna(df["original_cost_cr"]).sum()),2),"expenditure_cr":round(float(df["expenditure_cr"].fillna(0).sum()),2),"risk_distribution":levels,"sectors":int(df["sector"].nunique()),"dataset_snapshot":"2026-05-31","dataset_scope":"curated official PAIMANA public-project subset"}
