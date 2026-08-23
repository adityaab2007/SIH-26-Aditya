"""Inference and audit APIs for monthly lifecycle models."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import OUTCOMES, SNAPSHOTS, TRAJECTORIES, engineer_as_of_features, resolve_identities

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "monthly_lifecycle"
COMPARISON = ROOT / "reports" / "monthly_lifecycle_model_comparison.json"


def lifecycle_comparison() -> dict:
    if not COMPARISON.exists():
        return {"available": False, "reason": "Monthly lifecycle training report has not been generated yet.", "windows": []}
    return {"available": True, **json.loads(COMPARISON.read_text())}


@lru_cache(maxsize=2)
def _bundle(window: str) -> dict:
    target = MODEL_ROOT / window
    if not target.exists():
        raise FileNotFoundError(f"Monthly lifecycle model {window} is not available")
    return {"metadata": json.loads((target / "metadata.json").read_text()),
            "importance": json.loads((target / "shap_importance.json").read_text()),
            "cost": joblib.load(target / "cost_model.pkl"), "delay": joblib.load(target / "delay_model.pkl"), "risk": joblib.load(target / "risk_model.pkl")}


@lru_cache(maxsize=1)
def _inference_frame() -> pd.DataFrame:
    if TRAJECTORIES.exists():
        frame = pd.read_csv(TRAJECTORIES, dtype={"project_id": "string"}, low_memory=False)
        frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date, errors="coerce")
        return frame
    snapshots = pd.read_csv(SNAPSHOTS, dtype={"project_id": "string"}, low_memory=False)
    outcomes = pd.read_csv(OUTCOMES, dtype={"project_id": "string"}, low_memory=False)
    resolved, _ = resolve_identities(snapshots, outcomes)
    return engineer_as_of_features(resolved, outcomes)


def lifecycle_project_forecast(code: str, window: str = "2015_2021") -> dict:
    code = str(code).strip().upper(); frame = _inference_frame(); rows = frame[frame.project_id.astype("string").str.upper().eq(code)].sort_values("snapshot_date")
    if rows.empty:
        raise KeyError(code)
    latest = rows.iloc[-1]; bundle = _bundle(window); features = bundle["metadata"]["features_used"]
    X = latest.to_frame().T[features]; cost = float(bundle["cost"].predict(X)[0]); delay = max(0.0, float(bundle["delay"].predict(X)[0])); risk = str(bundle["risk"].predict(X)[0])
    importance = bundle["importance"]
    factors = [{"feature": item["feature"], "impact": item["importance"], "direction": "global importance"} for item in importance["cost"]["features"][:8]]
    inputs = {name: (None if pd.isna(latest.get(name)) else latest.get(name)) for name in features}
    for key, value in list(inputs.items()):
        if isinstance(value, (np.integer, np.floating)):
            inputs[key] = value.item()
        elif isinstance(value, pd.Timestamp):
            inputs[key] = value.strftime("%Y-%m-%d")
    return {"project_id": code, "project_name": latest.project_name, "model_version": bundle["metadata"]["model_version"],
            "snapshot_date": pd.Timestamp(latest.snapshot_date).strftime("%Y-%m-%d"), "history_snapshots": int(len(rows)),
            "predicted_cost_overrun_percentage": round(cost, 2), "predicted_delay_days": round(delay, 1), "risk_level": risk,
            "model_inputs": inputs, "shap_explanation": factors,
            "model_scope": "Official PAIMANA monthly lifecycle model; trajectory features use this project only through the displayed snapshot date."}


def forecast_evolution(project_id: str, window: str = "2015_2021") -> dict:
    path = MODEL_ROOT / window / "prediction_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Validation rows for {window} are unavailable")
    frame = pd.read_csv(path, dtype={"canonical_project_id": "string"})
    rows = frame[frame.canonical_project_id.eq(str(project_id))].sort_values("snapshot_date")
    safe = rows.astype(object).where(pd.notna(rows), None)
    return {"model_version": window, "project_id": project_id, "items": safe.to_dict("records"), "count": int(len(rows)),
            "source_policy": "Every point is an official historical snapshot; no synthetic interpolation is used."}
