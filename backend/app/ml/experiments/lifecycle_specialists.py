"""Experiment 4: lifecycle-specific final cost and delay specialists.

The production monthly lifecycle model remains the baseline. This module trains
one cost regressor and one delay regressor for each lifecycle stage and routes
each snapshot to exactly one available specialist; predictions are never
averaged across stages.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES, CANDIDATE_FEATURES, TARGETS, as_of_feature_evidence,
    build_training_dataset,
)
from backend.app.ml.monthly_training import (
    MODEL_ROOT, _fit_pipeline, _regression_metrics, _regressors, _train_variant,
    temporal_project_split,
)
from backend.app.ml.provenance import feature_schema_fingerprint, frame_fingerprint, git_commit_sha, new_run_id

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = MODEL_ROOT.parent / "lifecycle_specialists"
STAGES = ("early", "early_mid", "late_mid", "late")
STAGE_RANGES = {
    "early": (0.0, 0.25), "early_mid": (0.25, 0.50),
    "late_mid": (0.50, 0.75), "late": (0.75, float("inf")),
}
MIN_TRAIN_PROJECTS = 10
MIN_VALIDATION_PROJECTS = 2
MIN_TEST_PROJECTS = 2
LEAKY_COLUMNS = set(TARGETS) | {"completion_date", "reported_completion_expenditure_cr"}


def _serial_range(stage: str) -> list[float | None]:
    lower, upper = STAGE_RANGES[stage]
    return [lower, None if not np.isfinite(upper) else upper]


def select_lifecycle_stage(duration_ratio: object) -> str | None:
    """Return the one stage for a snapshot, preserving missing/invalid ratios."""
    try:
        ratio = float(duration_ratio)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ratio) or ratio < 0:
        return None
    for stage in STAGES:
        lower, upper = STAGE_RANGES[stage]
        if lower <= ratio < upper:
            return stage
    return None


def add_lifecycle_stages(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign stages from snapshot-time duration ratio using the four fixed bins."""
    result = frame.copy()
    if "duration_ratio" not in result.columns and "lifecycle_stage" in result.columns:
        # Useful for isolated callers/tests that already supplied an audited
        # stage; production datasets always derive it from duration_ratio.
        return result
    result["lifecycle_stage"] = result.get("duration_ratio", pd.Series(index=result.index, dtype=float)).map(select_lifecycle_stage).astype("string")
    return result


def _valid_features(features: list[str], frame: pd.DataFrame) -> list[str]:
    return [name for name in dict.fromkeys(features) if name in frame.columns and name not in LEAKY_COLUMNS]


def _strict_selection(train: pd.DataFrame, features: list[str], target: str, seed: int) -> tuple[str, Any, list[dict[str, Any]]] | None:
    """Select an algorithm only with a project-disjoint temporal validation cohort."""
    years = pd.to_numeric(train["completion_year"], errors="coerce").dropna()
    if years.nunique() < 2:
        return None
    validation_year = int(years.max())
    fitting = train[train.completion_year < validation_year]
    validation = train[train.completion_year == validation_year]
    if fitting.canonical_project_id.nunique() < MIN_TRAIN_PROJECTS or validation.canonical_project_id.nunique() < MIN_VALIDATION_PROJECTS:
        return None
    comparisons = []
    for name, model in _regressors(seed).items():
        fitted = _fit_pipeline(model, fitting, features, target)
        predicted = np.maximum(0, fitted.predict(validation[features])) if target == "actual_delay_days" else fitted.predict(validation[features])
        comparisons.append({"algorithm": name, **_regression_metrics(validation[target], predicted, validation.sample_weight, validation.canonical_project_id)})
    winner = min(comparisons, key=lambda item: (item["MAE"], item["RMSE"]))["algorithm"]
    return winner, _fit_pipeline(_regressors(seed)[winner], train, features, target), comparisons


def _metrics(actual: pd.Series, predicted: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    return _regression_metrics(actual, predicted, frame.sample_weight, frame.canonical_project_id)


def _comparison(global_rows: pd.DataFrame, specialist_rows: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for stage in STAGES:
        global_part = global_rows[global_rows.lifecycle_stage.eq(stage)]
        specialist_part = specialist_rows[specialist_rows.lifecycle_stage.eq(stage)]
        item: dict[str, Any] = {"lifecycle_range": _serial_range(stage), "available": not specialist_part.empty}
        if global_part.empty:
            item["global_cost"] = item["global_delay"] = None
            item["specialist_cost"] = item["specialist_delay"] = None
        else:
            item["global_cost"] = _metrics(global_part.actual_cost_overrun_percentage, global_part.predicted_cost_overrun.to_numpy(), global_part)
            item["global_delay"] = _metrics(global_part.actual_delay_days, global_part.predicted_delay_days.to_numpy(), global_part)
            item["specialist_cost"] = _metrics(specialist_part.actual_cost_overrun_percentage, specialist_part.predicted_cost_overrun.to_numpy(), specialist_part) if not specialist_part.empty else None
            item["specialist_delay"] = _metrics(specialist_part.actual_delay_days, specialist_part.predicted_delay_days.to_numpy(), specialist_part) if not specialist_part.empty else None
        for target in ("cost", "delay"):
            global_mae = (item.get(f"global_{target}") or {}).get("MAE")
            specialist_mae = (item.get(f"specialist_{target}") or {}).get("MAE")
            item[f"{target}_improvement_pct"] = round((global_mae - specialist_mae) / global_mae * 100, 3) if global_mae not in (None, 0) and specialist_mae is not None else None
        result[stage] = item
    return result


def _overall(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "cost": _metrics(rows.actual_cost_overrun_percentage, rows.predicted_cost_overrun.to_numpy(), rows),
        "delay": _metrics(rows.actual_delay_days, rows.predicted_delay_days.to_numpy(), rows),
        "rows": int(len(rows)), "unique_projects": int(rows.canonical_project_id.nunique()),
    }


def _importance(model: Any, frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    tree = model.named_steps["model"]
    values = getattr(tree, "feature_importances_", None)
    if values is None:
        return {"method": "unavailable", "features": []}
    names = model.named_steps["preprocess"].get_feature_names_out()
    aggregate: dict[str, float] = {feature: 0.0 for feature in features}
    for name, value in zip(names, values):
        clean = name.split("__", 1)[-1]
        feature = next((candidate for candidate in features if clean == candidate or clean.startswith(candidate + "_")), clean)
        aggregate[feature] = aggregate.get(feature, 0.0) + float(value)
    return {"method": "tree_feature_importance", "features": [{"feature": key, "importance": round(value, 6)} for key, value in sorted(aggregate.items(), key=lambda item: item[1], reverse=True)]}


def train_lifecycle_specialists(training_start: int, training_end: int, test_end: int, data: pd.DataFrame | None = None, identity: pd.DataFrame | None = None, artifact_root: Path | None = None) -> dict[str, Any]:
    """Train Experiment 4 for a year window and persist namespaced artifacts."""
    if data is None or identity is None:
        data, identity = build_training_dataset()
    data = add_lifecycle_stages(data.copy())
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    train, test = temporal_project_split(data, int(training_start), int(training_end), int(test_end))
    if train.canonical_project_id.nunique() < MIN_TRAIN_PROJECTS or test.canonical_project_id.nunique() < MIN_TEST_PROJECTS:
        raise ValueError("Experiment 4 requires enough project-disjoint training and future holdout projects.")
    audit = audit_features(train, CANDIDATE_FEATURES, minimum_availability=10, minimum_year_coverage=2, as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES))
    features = _valid_features(list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"])), train)
    destination = (artifact_root or EXPERIMENT_ROOT) / f"{training_start}_{training_end}"
    destination.mkdir(parents=True, exist_ok=True)
    global_bundle, global_metrics, global_rows = _train_variant(train, test, features, 26400)
    specialist_rows = []
    specialists: dict[str, Any] = {}
    for index, stage in enumerate(STAGES):
        stage_train = train[train.lifecycle_stage.eq(stage)].copy()
        stage_test = test[test.lifecycle_stage.eq(stage)].copy()
        train_projects = int(stage_train.canonical_project_id.nunique())
        valid_year = int(stage_train.completion_year.max()) if not stage_train.empty else None
        valid_projects = int(stage_train[stage_train.completion_year.eq(valid_year)].canonical_project_id.nunique()) if valid_year else 0
        test_projects = int(stage_test.canonical_project_id.nunique())
        base = {"available": False, "training_rows": int(len(stage_train)), "training_projects": train_projects, "validation_rows": int(len(stage_train[stage_train.completion_year.eq(valid_year)])) if valid_year else 0, "validation_projects": valid_projects, "test_rows": int(len(stage_test)), "test_projects": test_projects, "lifecycle_range": _serial_range(stage)}
        selected_cost = _strict_selection(stage_train, features, "actual_cost_overrun_percentage", 26500 + index * 10)
        selected_delay = _strict_selection(stage_train, features, "actual_delay_days", 26501 + index * 10)
        if train_projects < MIN_TRAIN_PROJECTS or test_projects < MIN_TEST_PROJECTS or selected_cost is None or selected_delay is None:
            base["reason"] = "insufficient specialist training data or internal temporal validation"
            specialists[stage] = base
            continue
        cost_name, cost_model, cost_comparisons = selected_cost
        delay_name, delay_model, delay_comparisons = selected_delay
        predictions = stage_test[["canonical_project_id", "project_name", "snapshot_date", "completion_year", "lifecycle_stage", "actual_cost_overrun_percentage", "actual_delay_days", "sample_weight"]].copy()
        predictions["predicted_cost_overrun"] = cost_model.predict(stage_test[features])
        predictions["predicted_delay_days"] = np.maximum(0, delay_model.predict(stage_test[features]))
        predictions["specialist_model"] = stage
        specialist_rows.append(predictions)
        stage_dir = destination / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(cost_model, stage_dir / "cost_model.pkl"); joblib.dump(delay_model, stage_dir / "delay_model.pkl")
        (stage_dir / "metadata.json").write_text(json.dumps({"stage": stage, "lifecycle_range": _serial_range(stage), "features": features, "selected_algorithms": {"cost": cost_name, "delay": delay_name}, "training_projects": train_projects, "validation_projects": valid_projects, "test_projects": test_projects}, indent=2))
        (stage_dir / "shap_importance.json").write_text(json.dumps({"cost": _importance(cost_model, stage_train, features), "delay": _importance(delay_model, stage_train, features)}, indent=2))
        specialists[stage] = {**base, "available": True, "testing_projects": test_projects, "selected_algorithms": {"cost": cost_name, "delay": delay_name}, "internal_validation": {"cost": cost_comparisons, "delay": delay_comparisons}, "metrics": {"cost": _metrics(predictions.actual_cost_overrun_percentage, predictions.predicted_cost_overrun.to_numpy(), predictions), "delay": _metrics(predictions.actual_delay_days, predictions.predicted_delay_days.to_numpy(), predictions)}}
    routed = pd.concat(specialist_rows, ignore_index=True) if specialist_rows else pd.DataFrame(columns=list(test.columns) + ["predicted_cost_overrun", "predicted_delay_days"])
    comparison = _comparison(global_rows, routed)
    result = {"experiment": "lifecycle_specialists", "experiment_name": "lifecycle_specialists", "implementation": "independent_stage_models", "run_id": new_run_id(), "model_version": f"lifecycle-specialists-{training_start}-{training_end}", "training_period": [int(training_start), int(training_end)], "holdout_period": [int(training_end) + 1, int(test_end)], "created_at": datetime.now(timezone.utc).isoformat(), "features": features, "feature_schema_fingerprint": feature_schema_fingerprint(features), "dataset_fingerprint": frame_fingerprint(data), "training_project_ids_hash": frame_fingerprint(train[["canonical_project_id"]].drop_duplicates()), "source_commit": git_commit_sha(ROOT), "lifecycle_boundaries": {stage: _serial_range(stage) for stage in STAGES}, "sample_weighting_policy": "quarterly deduplication precedes per-project 1/n retained snapshot weights", "leakage_policy": "identity-verified as-of features only; targets/final outcomes excluded; project-disjoint temporal holdout", "global_baseline": {"metrics": global_metrics, "holdout_rows": int(len(global_rows)), "holdout_projects": int(global_rows.canonical_project_id.nunique())}, "specialist_overall": _overall(routed) if not routed.empty else None, "comparison": comparison, "specialists": specialists}
    if not routed.empty:
        routed.to_csv(destination / "routed_predictions.csv", index=False)
    (destination / "metadata.json").write_text(json.dumps({key: result[key] for key in ("experiment", "run_id", "model_version", "training_period", "holdout_period", "features", "feature_schema_fingerprint", "dataset_fingerprint", "lifecycle_boundaries", "sample_weighting_policy", "leakage_policy", "source_commit")}, indent=2, allow_nan=False))
    (destination / "feature_quality_report.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    (destination / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (destination / "evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (destination / "experiment_4_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (destination / "run_manifest.json").write_text(json.dumps({"status": "complete", "experiment_name": "lifecycle_specialists", "run_id": result["run_id"], "window": f"{training_start}_{training_end}", "artifacts": {"metadata.json": True, "comparison.json": True, "evaluation_results.json": True, "feature_quality_report.json": True, "routed_predictions.csv": bool(not routed.empty), "stages": {stage: bool(item.get("available")) for stage, item in specialists.items()}}, "created_at": result["created_at"]}, indent=2))
    return result


def load_specialist_bundle(window: str, artifact_root: Path | None = None) -> dict[str, Any]:
    """Load metadata and the four stage bundles for an experiment window."""
    root = (artifact_root or EXPERIMENT_ROOT) / window
    report_path = root / "comparison.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Lifecycle specialist experiment {window} is unavailable")
    report = json.loads(report_path.read_text())
    bundles = {}
    for stage in STAGES:
        item = report["specialists"].get(stage, {})
        if item.get("available"):
            bundles[stage] = {"cost": joblib.load(root / stage / "cost_model.pkl"), "delay": joblib.load(root / stage / "delay_model.pkl"), **item}
    return {"report": report, "bundles": bundles}


def predict_with_specialist(snapshot: pd.Series | dict[str, Any], bundle: dict[str, Any], global_model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Route one snapshot to one stage specialist, or explicitly fall back globally."""
    row = snapshot if isinstance(snapshot, pd.Series) else pd.Series(snapshot)
    stage = select_lifecycle_stage(row.get("duration_ratio"))
    ratio = row.get("duration_ratio")
    if stage is None:
        reason = "missing or invalid lifecycle ratio"
    elif stage not in bundle.get("bundles", {}):
        reason = "insufficient specialist training data"
    else:
        specialist = bundle["bundles"][stage]
        features = bundle["report"]["features"]
        X = row.to_frame().T[features]
        return {"model_family": "lifecycle_specialist", "experiment": "experiment_4", "lifecycle_ratio": None if pd.isna(ratio) else float(ratio), "lifecycle_percentage": None if pd.isna(ratio) else float(ratio) * 100, "lifecycle_stage": stage, "specialist_used": True, "fallback_to_global": False, "cost": {"predicted_final_overrun_percentage": float(specialist["cost"].predict(X)[0]), "algorithm": specialist["selected_algorithms"]["cost"]}, "delay": {"predicted_final_delay_days": float(max(0, specialist["delay"].predict(X)[0])), "algorithm": specialist["selected_algorithms"]["delay"]}}
    if global_model is None:
        raise ValueError(reason)
    return {"model_family": "monthly_lifecycle_global_fallback", "experiment": "experiment_4", "lifecycle_ratio": None if pd.isna(ratio) else float(ratio), "lifecycle_percentage": None if pd.isna(ratio) else float(ratio) * 100, "lifecycle_stage": stage, "specialist_used": False, "fallback_to_global": True, "fallback_reason": reason, "cost": global_model["cost"], "delay": global_model["delay"]}


def improvement_percent(global_mae: float | None, specialist_mae: float | None) -> float | None:
    """Calculate positive-is-better error reduction without hiding regressions."""
    if global_mae in (None, 0) or specialist_mae is None:
        return None
    return (global_mae - specialist_mae) / global_mae * 100
