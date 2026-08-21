from fastapi import APIRouter
from backend.app.services.data_service import projects_df

router = APIRouter(prefix="/api/data-quality", tags=["data-quality"])

@router.get("")
def data_quality():
    df = projects_df()
    flags = {
        "missing_revised_cost": int(df["dq_missing_revised_cost"].sum()),
        "missing_revised_date": int(df["dq_missing_revised_date"].sum()),
        "missing_physical_progress": int(df["dq_missing_progress"].sum()),
        "expenditure_above_revised_cost": int(df["dq_expenditure_gt_revised"].sum()),
        "revised_date_before_original": int(df["dq_revised_date_before_original"].sum()),
    }
    issues = []
    flag_cols = [c for c in df.columns if c.startswith("dq_")]
    for _, r in df[df[flag_cols].sum(axis=1) > 0].head(50).iterrows():
        active = [c.replace("dq_", "") for c in flag_cols if r[c] == 1]
        issues.append({"project_code": str(r["project_code"]), "project_name": r["project_name"], "flags": active})
    return {"rows": int(len(df)), "flags": flags, "issues": issues}
