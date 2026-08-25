"""Inference and audit APIs for monthly lifecycle models."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import OUTCOMES, SNAPSHOTS, SNAPSHOTS_GZ, TRAJECTORIES, engineer_as_of_features, load_monthly_snapshots, resolve_identities
from backend.app.ml.experiments.lifecycle_specialists import load_specialist_bundle, predict_with_specialist
from backend.app.ml.provenance import file_sha256
from backend.app.services.simulation_service import _shap_factors_for_model

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "monthly_lifecycle"
COMPARISON = ROOT / "reports" / "monthly_lifecycle_model_comparison.json"


def lifecycle_comparison() -> dict:
    if not COMPARISON.exists():
        return {"available": False, "reason": "Monthly lifecycle training report has not been generated yet.", "windows": []}
    return {"available": True, **json.loads(COMPARISON.read_text())}


def _current_source_hashes() -> dict[str, str | None]:
    snapshot_path = SNAPSHOTS if SNAPSHOTS.exists() else SNAPSHOTS_GZ
    return {
        "monthly_snapshots": file_sha256(snapshot_path) if snapshot_path.exists() else None,
        "completed_outcomes": file_sha256(OUTCOMES) if OUTCOMES.exists() else None,
    }


def _validate_bundle_provenance(window: str, metadata: dict, manifest: dict) -> None:
    if not manifest:
        # Older committed runs remain readable for backwards compatibility, but
        # callers can see provenance.verified=false in the response.
        return
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Lifecycle model {window} is not provenance-valid ({manifest.get('status') or 'invalid manifest'}). Retrain this window before inference."
        )
    manifest_run = manifest.get("run_id")
    metadata_run = metadata.get("run_id") or (metadata.get("provenance") or {}).get("run_id")
    manifest_dataset = manifest.get("dataset_fingerprint")
    metadata_dataset = metadata.get("dataset_fingerprint") or (metadata.get("provenance") or {}).get("dataset_fingerprint")
    if not manifest_run or not metadata_run or not manifest_dataset or not metadata_dataset:
        raise RuntimeError(f"Lifecycle model {window} has an incomplete provenance manifest. Retrain this window before inference.")
    if manifest_run != metadata_run:
        raise RuntimeError(f"Lifecycle model {window} failed provenance validation: manifest/metadata run IDs differ.")
    if manifest_dataset != metadata_dataset:
        raise RuntimeError(f"Lifecycle model {window} failed provenance validation: manifest/metadata dataset fingerprints differ.")

    expected_sources = manifest.get("source_dataset_files") or {}
    current_sources = _current_source_hashes()
    for name, expected in expected_sources.items():
        current = current_sources.get(name)
        if expected and current and expected != current:
            raise RuntimeError(
                f"Lifecycle model {window} was trained against a different {name} dataset. Retrain before inference."
            )


@lru_cache(maxsize=2)
def _bundle(window: str) -> dict:
    target = MODEL_ROOT / window
    if not target.exists():
        raise FileNotFoundError(f"Monthly lifecycle model {window} is not available")
    manifest_path = target / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    metadata = json.loads((target / "metadata.json").read_text())
    _validate_bundle_provenance(window, metadata, manifest)
    return {
        "metadata": metadata,
        "manifest": manifest,
        "importance": json.loads((target / "shap_importance.json").read_text()),
        "cost": joblib.load(target / "cost_model.pkl"),
        "delay": joblib.load(target / "delay_model.pkl"),
        "risk": joblib.load(target / "risk_model.pkl"),
    }


@lru_cache(maxsize=1)
def _inference_frame() -> pd.DataFrame:
    if TRAJECTORIES.exists():
        frame = pd.read_csv(TRAJECTORIES, dtype={"project_id": "string"}, low_memory=False)
        frame["snapshot_date"] = pd.to_datetime(frame.snapshot_date, errors="coerce")
        return frame
    snapshots = load_monthly_snapshots()
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
    global_factors = [{"feature": item["feature"], "importance": item["importance"]} for item in importance.get("cost", {}).get("features", [])[:8]]
    local_factors = _shap_factors_for_model(bundle["cost"], latest, features)
    inputs = {name: (None if pd.isna(latest.get(name)) else latest.get(name)) for name in features}
    for key, value in list(inputs.items()):
        if isinstance(value, (np.integer, np.floating)):
            inputs[key] = value.item()
        elif isinstance(value, pd.Timestamp):
            inputs[key] = value.strftime("%Y-%m-%d")
    provenance = bundle["metadata"].get("provenance") or {}
    return {
        "project_id": code,
        "project_name": latest.project_name,
        "model_version": bundle["metadata"]["model_version"],
        "snapshot_date": pd.Timestamp(latest.snapshot_date).strftime("%Y-%m-%d"),
        "history_snapshots": int(len(rows)),
        "predicted_cost_overrun_percentage": round(cost, 2),
        "predicted_delay_days": round(delay, 1),
        "risk_level": risk,
        "model_inputs": inputs,
        "shap_explanation": local_factors,
        "global_feature_importance": global_factors,
        "explanation_scope": "shap_explanation is project-specific for the displayed snapshot; global_feature_importance is aggregate training-sample importance.",
        "provenance": {
            "run_id": bundle["metadata"].get("run_id") or provenance.get("run_id"),
            "dataset_fingerprint": bundle["metadata"].get("dataset_fingerprint") or provenance.get("dataset_fingerprint"),
            "verified": bool(bundle.get("manifest") and bundle["manifest"].get("status") == "complete"),
        },
        "model_scope": "Official PAIMANA monthly lifecycle model; trajectory features use this project only through the displayed snapshot date.",
    }


def forecast_evolution(project_id: str, window: str = "2015_2021") -> dict:
    path = MODEL_ROOT / window / "prediction_validation.csv"
    if not path.exists():
        raise FileNotFoundError(f"Validation rows for {window} are unavailable")
    frame = pd.read_csv(path, dtype={"canonical_project_id": "string"})
    rows = frame[frame.canonical_project_id.eq(str(project_id))].sort_values("snapshot_date")
    safe = rows.astype(object).where(pd.notna(rows), None)
    return {"model_version": window, "project_id": project_id, "items": safe.to_dict("records"), "count": int(len(rows)),
            "source_policy": "Every point is an official historical snapshot; no synthetic interpolation is used."}


def lifecycle_specialist_comparison(window: str) -> dict:
    """Return the persisted Experiment 4 comparison without changing production inference."""
    return load_specialist_bundle(window)["report"]


def lifecycle_specialist_forecast(code: str, window: str = "2015_2021") -> dict:
    """Predict one latest historical snapshot using exactly one specialist."""
    code = str(code).strip().upper()
    frame = _inference_frame()
    rows = frame[frame.project_id.astype("string").str.upper().eq(code)].sort_values("snapshot_date")
    if rows.empty:
        raise KeyError(code)
    latest = rows.iloc[-1]
    global_bundle = _bundle(window)
    features = global_bundle["metadata"]["features_used"]
    X = latest.to_frame().T[features]
    global_prediction = {
        "cost": {"predicted_final_overrun_percentage": float(global_bundle["cost"].predict(X)[0]), "algorithm": global_bundle["metadata"].get("selected_algorithms", {}).get("cost")},
        "delay": {"predicted_final_delay_days": float(max(0, global_bundle["delay"].predict(X)[0])), "algorithm": global_bundle["metadata"].get("selected_algorithms", {}).get("delay")},
    }
    result = predict_with_specialist(latest, load_specialist_bundle(window), global_prediction)
    result.update({"project_id": code, "project_name": latest.project_name, "snapshot_date": pd.Timestamp(latest.snapshot_date).strftime("%Y-%m-%d")})
    return result


def lifecycle_specialist_convergence(code: str, window: str = "2015_2021", include_actual: bool = False) -> dict:
    """Return nearest real historical milestones for one project; never interpolate rows."""
    code = str(code).strip().upper()
    frame = _inference_frame()
    rows = frame[frame.project_id.astype("string").str.upper().eq(code)].sort_values("snapshot_date").copy()
    if rows.empty:
        raise KeyError(code)
    bundle = load_specialist_bundle(window)
    global_bundle = _bundle(window)
    features = global_bundle["metadata"]["features_used"]
    chosen = []
    for milestone in (.25, .50, .75, float(rows.duration_ratio.max())):
        if milestone is None or not np.isfinite(milestone):
            continue
        row = rows.iloc[(rows.duration_ratio - milestone).abs().argsort()[:1].to_numpy()[0]]
        if int(row.name) not in [int(item.name) for item in chosen]:
            chosen.append(row)
    items = []
    for row in chosen:
        X = row.to_frame().T[features]
        global_prediction = {"cost": {"predicted_final_overrun_percentage": float(global_bundle["cost"].predict(X)[0])}, "delay": {"predicted_final_delay_days": float(max(0, global_bundle["delay"].predict(X)[0]))}}
        routed = predict_with_specialist(row, bundle, global_prediction)
        item = {"snapshot_date": pd.Timestamp(row.snapshot_date).strftime("%Y-%m-%d"), "lifecycle_percentage": None if pd.isna(row.duration_ratio) else float(row.duration_ratio) * 100, "lifecycle_stage": routed.get("lifecycle_stage"), "predicted_final_cost_overrun": routed["cost"]["predicted_final_overrun_percentage"], "predicted_final_delay": routed["delay"]["predicted_final_delay_days"], "specialist_model": routed.get("lifecycle_stage") if routed.get("specialist_used") else "global_fallback", "algorithm": {"cost": routed["cost"].get("algorithm"), "delay": routed["delay"].get("algorithm")}}
        if include_actual:
            item.update({"actual_final_cost_overrun": None if pd.isna(row.get("actual_cost_overrun_percentage")) else float(row.actual_cost_overrun_percentage), "absolute_cost_error": None if pd.isna(row.get("actual_cost_overrun_percentage")) else abs(routed["cost"]["predicted_final_overrun_percentage"] - float(row.actual_cost_overrun_percentage)), "actual_final_delay": None if pd.isna(row.get("actual_delay_days")) else float(row.actual_delay_days), "absolute_delay_error": None if pd.isna(row.get("actual_delay_days")) else abs(routed["delay"]["predicted_final_delay_days"] - float(row.actual_delay_days))})
        items.append(item)
    return {"project_id": code, "items": items, "count": len(items), "actuals_included": bool(include_actual), "source_policy": "Nearest available official historical snapshots only; no synthetic interpolation."}
