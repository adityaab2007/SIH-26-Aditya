"""API-facing helpers for the real PAIMANA historical model simulations."""
from __future__ import annotations

import math
import json
from typing import Any
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.real_time_windows import FEATURES, WINDOWS, evaluate, labelled, model_dir, outcome_data, versions


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


def _test_data(key: str) -> pd.DataFrame:
    if key not in WINDOWS:
        raise KeyError(key)
    window = WINDOWS[key]
    data = labelled(outcome_data())
    actual_end = min(window.test_end, int(data.completion_year.max()))
    result = data[data.completion_year.between(window.test_start, actual_end)].sort_values(["completion_date", "project_name"]).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"No official completed-project outcomes are available for {window.test_start}-{actual_end}.")
    return result


def candidate_projects(key: str) -> list[dict]:
    rows = _test_data(key)
    return [{
        "record_index": int(index), "project_id": _value(row.project_id) or "Not published", "project_name": _value(row.project_name),
        "sector": _value(row.sector), "approved_cost_cr": _value(row.approved_cost_cr), "planned_commissioning_date": _value(row.planned_commissioning_date),
    } for index, row in rows.iterrows()]


def _metrics(key: str) -> dict:
    path = model_dir(key) / "evaluation_results.json"
    if path.exists():
        return json.loads(path.read_text())
    return evaluate(key, save=True)["metrics"]


def run_project(key: str, record_index: int) -> dict:
    rows = _test_data(key)
    if record_index < 0 or record_index >= len(rows):
        raise ValueError("Selected project is not in this model's official evaluation cohort.")
    row = rows.iloc[record_index]
    X = pd.DataFrame([row[FEATURES]])
    cost_model = joblib.load(model_dir(key) / "cost_model.pkl")
    delay_model = joblib.load(model_dir(key) / "delay_model.pkl")
    risk_model = joblib.load(model_dir(key) / "risk_model.pkl")
    predicted_cost = float(cost_model.predict(X)[0])
    predicted_delay = max(0.0, float(delay_model.predict(X)[0]))
    predicted_risk = int(risk_model.predict(X)[0])
    actual_cost = float(row.actual_cost_overrun_percentage)
    actual_delay = float(row.actual_delay_days)
    item = {
        "record_index": int(record_index), "project_id": _value(row.project_id) or "Not published", "project_name": _value(row.project_name), "sector": _value(row.sector),
        "predicted_cost_overrun": round(predicted_cost, 4), "actual_cost_overrun": round(actual_cost, 4), "cost_error": round(predicted_cost - actual_cost, 4),
        "predicted_delay_days": round(predicted_delay, 4), "actual_delay_days": round(actual_delay, 4), "delay_error": round(predicted_delay - actual_delay, 4),
        "predicted_risk": "HIGH" if predicted_risk else "LOW", "actual_risk": "HIGH" if int(row.actual_risk) else "LOW",
        "snapshot": {"approved_cost_cr": _value(row.approved_cost_cr), "current_cost_cr": None, "physical_progress_percentage": None, "expenditure_cr": None, "sector": _value(row.sector), "planned_commissioning_date": _value(row.planned_commissioning_date), "implementing_agency": _value(row.implementing_agency), "note": "This prediction is computed now from the selected model and the official fields available before the reported completion outcome. Current cost, progress, and expenditure were not reported in this completed-project source row."},
        "shap_explanation": _shap_factors(key, row),
    }
    return {"version": key, "generated_at": datetime.now(timezone.utc).isoformat(), "metrics": _metrics(key), "item": item, "reveal_policy": "The API returns the official outcome only for this historical completed project. The interface keeps it hidden until Reveal Actual Outcome is selected."}


def run(key: str) -> dict:
    result = evaluate(key, save=False)
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
