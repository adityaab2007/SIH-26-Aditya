from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd
from backend.app.services.data_service import projects_df
from backend.app.services.prediction_service import project_forecast

@lru_cache(maxsize=1)
def portfolio_rows() -> list[dict]:
    df=projects_df().copy()
    forecasts=[project_forecast(str(code), include_explanations=False) for code in df["project_code"]]
    schedule_days=np.asarray([max(0.0, float(item["predicted_delay_days"])) for item in forecasts])
    cost_pct=np.asarray([max(0.0, float(item["predicted_cost_overrun_percentage"])) for item in forecasts])
    schedule_prob=np.clip(schedule_days/730,0,1); cost_prob=np.clip(cost_pct/40,0,1)
    exposure=df["original_cost_cr"].rank(method="average",pct=True).fillna(.4).to_numpy(); rows=[]
    for i,(_,r) in enumerate(df.iterrows()):
        priority=max(0,min(100,100*(.45*float(schedule_prob[i])+.35*float(cost_prob[i])+.20*float(exposure[i])))); level="critical" if priority>=80 else "high" if priority>=65 else "medium" if priority>=45 else "low"; complete=sum(pd.notna(r.get(c)) for c in ["revised_cost_cr","expenditure_cr","physical_progress_pct","revised_end_date"])/4; confidence="high" if complete>=.75 else "medium" if complete>=.5 else "low"
        forecast=forecasts[i]
        rows.append({"project_code":str(r["project_code"]),"project_name":r["project_name"],"model_scope":"temporal cost and delay forecasting","schedule_risk_probability":round(float(schedule_prob[i]),4),"cost_risk_probability":round(float(cost_prob[i]),4),"estimated_schedule_extension_days":round(float(schedule_days[i]),1),"estimated_cost_escalation_pct":round(float(cost_pct[i]),2),"priority_score":round(priority,1),"priority_level":level,"confidence":confidence,"exposure_percentile":round(float(exposure[i]),4),"best_models":{"schedule_classifier":forecast["best_models"]["delay"],"cost_classifier":forecast["best_models"]["cost"],"schedule_regressor":forecast["best_models"]["delay"],"cost_regressor":forecast["best_models"]["cost"]},"schedule_drivers":[],"cost_drivers":[],"observed":{"schedule_extension_days":None if pd.isna(r.get("schedule_extension_days")) else round(float(r["schedule_extension_days"]),1),"cost_escalation_pct":None if pd.isna(r.get("cost_escalation_pct")) else round(float(r["cost_escalation_pct"]),2),"financial_progress_pct":None if pd.isna(r.get("financial_progress_pct")) else round(float(r["financial_progress_pct"]),1),"physical_progress_pct":None if pd.isna(r.get("physical_progress_pct")) else round(float(r["physical_progress_pct"]),1)}})
    return rows

def summary() -> dict:
    df=projects_df(); preds=portfolio_rows(); levels={k:0 for k in ["critical","high","medium","low"]}
    for p in preds: levels[p["priority_level"]]+=1
    return {"projects":int(len(df)),"original_cost_cr":round(float(df["original_cost_cr"].fillna(0).sum()),2),"current_cost_basis_cr":round(float(df["revised_cost_cr"].fillna(df["original_cost_cr"]).sum()),2),"expenditure_cr":round(float(df["expenditure_cr"].fillna(0).sum()),2),"risk_distribution":levels,"sectors":int(df["sector"].nunique()),"dataset_snapshot":"2026-05-31","dataset_scope":"curated official PAIMANA public-project subset"}
