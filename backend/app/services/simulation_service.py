"""API-facing helpers for the real PAIMANA historical model simulations."""
from __future__ import annotations

import math
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.real_time_windows import FEATURES, evaluate, model_dir, versions


def _value(value: Any):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 4)
    return value


def available_versions() -> list[dict]:
    return versions()


def _shap_factors(key: str, row: pd.Series) -> list[dict]:
    """Return local SHAP contributions from the selected real-data cost model."""
    model = joblib.load(model_dir(key) / "cost_model.pkl")
    try:
        import shap
        transformed = model.named_steps["preprocess"].transform(pd.DataFrame([row[FEATURES]]))
        values = shap.TreeExplainer(model.named_steps["model"]).shap_values(transformed.toarray() if hasattr(transformed, "toarray") else transformed)
        names = model.named_steps["preprocess"].get_feature_names_out()
        values = np.asarray(values)[0]
        result = []
        for name, impact in sorted(zip(names, values), key=lambda item: abs(item[1]), reverse=True)[:5]:
            result.append({"feature": str(name).replace("numeric__", "").replace("category__", ""), "impact": round(float(impact), 4), "direction": "increases" if impact >= 0 else "reduces"})
        return result
    except Exception:
        # A truthful fallback for platforms where the optional SHAP backend cannot load.
        return [{"feature": "approved_cost_cr", "impact": 0.0, "direction": "not available"}]


def run(key: str) -> dict:
    result = evaluate(key, save=True)
    frame = result["rows"].reset_index(drop=True)
    items = []
    for index, row in frame.iterrows():
        items.append({
            "record_index": int(index), "project_id": _value(row.project_id) or "Not published", "project_name": _value(row.project_name), "sector": _value(row.sector),
            "completion_date": _value(row.completion_date), "approved_cost_cr": _value(row.approved_cost_cr),
            "predicted_cost_overrun": _value(row.predicted_cost_overrun), "actual_cost_overrun": _value(row.actual_cost_overrun_percentage), "cost_error": _value(row.cost_error),
            "predicted_delay_days": _value(row.predicted_delay_days), "actual_delay_days": _value(row.actual_delay_days), "delay_error": _value(row.delay_error),
            "predicted_risk": "HIGH" if int(row.predicted_risk) else "LOW", "actual_risk": "HIGH" if int(row.actual_risk) else "LOW",
            "snapshot": {"approved_cost_cr": _value(row.approved_cost_cr), "current_cost_cr": None, "physical_progress_percentage": None, "expenditure_cr": None, "sector": _value(row.sector), "note": "The completed-project archive reports approved cost and completion expenditure. Pre-completion current cost, progress, and expenditure were not reported in this source row."},
            "shap_explanation": _shap_factors(key, row),
        })
    return {"version": key, "metrics": result["metrics"], "items": items, "reveal_policy": "Actual outcome values are present for the historical evaluation response but should remain hidden until the user selects Reveal Actual Outcome."}
