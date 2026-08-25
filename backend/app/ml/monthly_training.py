"""Temporal/project-grouped training for official PAIMANA monthly snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from backend.app.ml.feature_audit import audit_features, write_feature_quality_report
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES, CANDIDATE_FEATURES, TRAJECTORY_FEATURES,
    as_of_feature_evidence, build_training_dataset, training_as_of_invariants,
)
from backend.app.ml.provenance import (
    artifact_fingerprints, feature_schema_fingerprint, frame_fingerprint,
    git_commit_sha, new_run_id,
)

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "monthly_lifecycle"
REPORTS = ROOT / "reports"
COMPARISON_JSON = REPORTS / "monthly_lifecycle_model_comparison.json"
COMPARISON_MD = REPORTS / "monthly_lifecycle_model_comparison.md"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [name for name in features if name in frame and (pd.api.types.is_object_dtype(frame[name]) or pd.api.types.is_string_dtype(frame[name]))]
    numeric = [name for name in features if name not in categorical]
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def _regressors(seed: int) -> dict[str, object]:
    return {
        "lightgbm": LGBMRegressor(n_estimators=240, learning_rate=.035, max_depth=5, num_leaves=24, random_state=seed, verbosity=-1),
        "xgboost": XGBRegressor(n_estimators=240, learning_rate=.035, max_depth=4, subsample=.85, colsample_bytree=.85, objective="reg:squarederror", random_state=seed, n_jobs=2),
        "extra_trees": ExtraTreesRegressor(n_estimators=260, min_samples_leaf=3, max_features=.8, random_state=seed, n_jobs=2),
    }


def _fit_pipeline(model: object, frame: pd.DataFrame, features: list[str], target: str) -> Pipeline:
    pipe = Pipeline([("preprocess", _preprocessor(frame, features)), ("model", model)])
    pipe.fit(frame[features], frame[target], model__sample_weight=frame.sample_weight.to_numpy(dtype=float))
    return pipe


def _regression_metrics(actual: pd.Series, predicted: np.ndarray, weights: pd.Series,
                        projects: pd.Series | None = None) -> dict:
    mask = actual.notna() & np.isfinite(predicted); y = actual[mask].to_numpy(dtype=float); p = np.asarray(predicted)[mask]; w = weights[mask].to_numpy(dtype=float)
    meaningful = np.abs(y) >= 1
    return {"MAE": round(float(mean_absolute_error(y, p, sample_weight=w)), 3),
            "RMSE": round(float(math.sqrt(mean_squared_error(y, p, sample_weight=w))), 3),
            "R2": round(float(r2_score(y, p, sample_weight=w)), 4) if len(y) > 1 else None,
            "MAPE": round(float(np.average(np.abs((y[meaningful] - p[meaningful]) / y[meaningful]), weights=w[meaningful]) * 100), 3) if meaningful.any() else None,
            "rows": int(len(y)), "unique_projects": int(projects[mask].nunique()) if projects is not None else None}


def _risk_metrics(actual: pd.Series, predicted: np.ndarray, weights: pd.Series) -> dict:
    labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]; w = weights.to_numpy(dtype=float)
    return {"accuracy": round(float(accuracy_score(actual, predicted, sample_weight=w)), 4),
            "macro_precision": round(float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0, sample_weight=w)), 4),
            "macro_recall": round(float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0, sample_weight=w)), 4),
            "macro_f1": round(float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0, sample_weight=w)), 4),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels, sample_weight=w).round(4).tolist(), "labels": labels}


def _select_regressor(train: pd.DataFrame, features: list[str], target: str, seed: int) -> tuple[str, Pipeline, list[dict]]:
    validation_year = int(train.completion_year.max()); fitting = train[train.completion_year.lt(validation_year)]; validation = train[train.completion_year.eq(validation_year)]
    if fitting.canonical_project_id.nunique() < 5 or validation.canonical_project_id.nunique() < 2:
        raise ValueError(f"Internal temporal validation is unavailable for {target}: fitting_projects={fitting.canonical_project_id.nunique()}, validation_projects={validation.canonical_project_id.nunique()}")
    comparisons = []
    for name, model in _regressors(seed).items():
        fitted = _fit_pipeline(model, fitting, features, target); pred = fitted.predict(validation[features])
        metrics = _regression_metrics(validation[target], pred, validation.sample_weight, validation.canonical_project_id)
        comparisons.append({"algorithm": name, **metrics})
    winner = min(comparisons, key=lambda item: item["MAE"])["algorithm"]
    final = _fit_pipeline(_regressors(seed)[winner], train, features, target)
    return winner, final, comparisons


def _fit_risk(train: pd.DataFrame, features: list[str], seed: int) -> Pipeline:
    model = RandomForestClassifier(n_estimators=320, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=2)
    return _fit_pipeline(model, train, features, "actual_risk")


def _evaluate(models: dict, test: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    cost = models["cost"].predict(test[features]); delay = np.maximum(0, models["delay"].predict(test[features])); risk = models["risk"].predict(test[features])
    metrics = {"cost": _regression_metrics(test.actual_cost_overrun_percentage, cost, test.sample_weight, test.canonical_project_id),
               "delay": _regression_metrics(test.actual_delay_days, delay, test.sample_weight, test.canonical_project_id),
               "risk": _risk_metrics(test.actual_risk, risk, test.sample_weight)}
    rows = test[["canonical_project_id", "project_name", "snapshot_date", "completion_year", "lifecycle_stage", "actual_cost_overrun_percentage", "actual_delay_days", "actual_risk", "sample_weight"]].copy()
    rows["predicted_cost_overrun"] = cost; rows["predicted_delay_days"] = delay; rows["predicted_risk"] = risk
    rows["cost_error"] = rows.predicted_cost_overrun - rows.actual_cost_overrun_percentage
    rows["delay_error"] = rows.predicted_delay_days - rows.actual_delay_days
    return metrics, rows


def _stage_metrics(rows: pd.DataFrame) -> dict:
    result = {}
    for stage in ["early", "early_mid", "late_mid", "late"]:
        part = rows[rows.lifecycle_stage.eq(stage)]
        if part.empty:
            result[stage] = {"available": False, "reason": "no test snapshots with valid duration ratio"}; continue
        w = part.sample_weight
        result[stage] = {"available": True, "cost": _regression_metrics(part.actual_cost_overrun_percentage, part.predicted_cost_overrun.to_numpy(), w, part.canonical_project_id),
                         "delay": _regression_metrics(part.actual_delay_days, part.predicted_delay_days.to_numpy(), w, part.canonical_project_id),
                         "risk": _risk_metrics(part.actual_risk, part.predicted_risk.to_numpy(), w)}
    return result


def _stage_distribution(rows: pd.DataFrame) -> dict:
    distribution = {}
    for stage in ["early", "early_mid", "late_mid", "late"]:
        part = rows[rows.lifecycle_stage.eq(stage)]
        distribution[stage] = {"rows": int(len(part)), "unique_projects": int(part.canonical_project_id.nunique())}
    return distribution


def _balanced_stage_summary(stages: dict) -> dict:
    available = [value for value in stages.values() if value.get("available")]
    if not available:
        return {"available": False}
    def avg(path: tuple[str, str]) -> float | None:
        values = [stage.get(path[0], {}).get(path[1]) for stage in available]
        values = [float(value) for value in values if value is not None]
        return round(float(np.mean(values)), 4) if values else None
    return {
        "available": True,
        "policy": "macro-average across lifecycle stages so very-late snapshots cannot dominate the headline diagnostic",
        "cost_mae": avg(("cost", "MAE")),
        "delay_mae": avg(("delay", "MAE")),
        "risk_macro_f1": avg(("risk", "macro_f1")),
        "early_warning": {stage: stages.get(stage) for stage in ("early", "early_mid")},
    }


def _importance(pipe: Pipeline, sample: pd.DataFrame, features: list[str]) -> dict:
    preprocess = pipe.named_steps["preprocess"]; model = pipe.named_steps["model"]; transformed = preprocess.transform(sample[features])
    names = preprocess.get_feature_names_out().tolist(); method = "tree_feature_importance"
    try:
        import shap
        values = shap.TreeExplainer(model).shap_values(transformed[: min(200, transformed.shape[0])])
        array = np.asarray(values)
        feature_axes = [axis for axis, size in enumerate(array.shape) if size == len(names)]
        if not feature_axes:
            raise ValueError(f"SHAP output shape {array.shape} has no axis matching {len(names)} transformed features")
        feature_axis = feature_axes[-1]
        scores = np.abs(array).mean(axis=tuple(axis for axis in range(array.ndim) if axis != feature_axis)).reshape(-1)
        method = "mean_absolute_shap"
    except Exception:
        scores = np.asarray(getattr(model, "feature_importances_", np.zeros(len(names))), dtype=float)
    aggregate = {feature: 0.0 for feature in features}
    for name, score in zip(names, scores):
        clean = name.split("__", 1)[-1]
        feature = next((candidate for candidate in features if clean == candidate or clean.startswith(candidate + "_")), clean)
        aggregate[feature] = aggregate.get(feature, 0.0) + float(score)
    total = sum(aggregate.values()) or 1.0
    return {"method": method, "scope": "global_training_sample", "features": [{"feature": name, "importance": round(value / total, 6)} for name, value in sorted(aggregate.items(), key=lambda item: item[1], reverse=True)]}


def _train_variant(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int, selected: dict[str, str] | None = None) -> tuple[dict, dict, pd.DataFrame]:
    if selected:
        regressors = _regressors(seed)
        cost = _fit_pipeline(regressors[selected["cost"]], train, features, "actual_cost_overrun_percentage")
        delay = _fit_pipeline(regressors[selected["delay"]], train, features, "actual_delay_days")
        comparisons = {}
    else:
        cost_name, cost, cost_cmp = _select_regressor(train, features, "actual_cost_overrun_percentage", seed)
        delay_name, delay, delay_cmp = _select_regressor(train, features, "actual_delay_days", seed + 1)
        selected = {"cost": cost_name, "delay": delay_name}; comparisons = {"cost": cost_cmp, "delay": delay_cmp}
    models = {"cost": cost, "delay": delay, "risk": _fit_risk(train, features, seed + 2)}
    metrics, rows = _evaluate(models, test, features)
    return {"models": models, "selected_algorithms": selected, "internal_comparisons": comparisons}, metrics, rows


def temporal_project_split(data: pd.DataFrame, training_start: int, training_end: int, test_end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[data.completion_year.between(training_start, training_end)].copy()
    test = data[data.completion_year.between(training_end + 1, test_end)].copy()
    overlap = set(train.canonical_project_id.dropna()) & set(test.canonical_project_id.dropna())
    if overlap:
        raise ValueError(f"Project-group leakage across temporal split: {len(overlap)} project(s)")
    return train, test


def train_window(training_start: int, training_end: int, test_end: int,
                 data: pd.DataFrame | None = None, identity: pd.DataFrame | None = None,
                 artifact_root: Path | None = None) -> dict:
    if data is None or identity is None:
        data, identity = build_training_dataset()
    data = data.copy(); data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    train, test = temporal_project_split(data, training_start, training_end, test_end)
    if train.canonical_project_id.nunique() < 10 or test.canonical_project_id.nunique() < 2:
        raise ValueError(f"Insufficient identity-verified monthly trajectories: train projects={train.canonical_project_id.nunique()}, test projects={test.canonical_project_id.nunique()}")
    train_invariants = training_as_of_invariants(train)
    test_invariants = training_as_of_invariants(test)
    if not train_invariants["passed"] or not test_invariants["passed"]:
        raise ValueError(f"As-of safety invariant failure: train={train_invariants}, test={test_invariants}")

    audit = audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage signal available in the same official snapshot; evaluated by ablation",
            "cost_escalation_percentage": "late-stage signal derived from same-snapshot revised cost; evaluated by ablation",
        },
    )
    lifecycle_features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"])); baseline_features = [name for name in BASELINE_FEATURES if name in train]
    run_id = new_run_id()
    provenance = {
        "run_id": run_id,
        "dataset_fingerprint": frame_fingerprint(data),
        "training_fingerprint": frame_fingerprint(train),
        "test_fingerprint": frame_fingerprint(test),
        "feature_schema_fingerprint": feature_schema_fingerprint(lifecycle_features),
        "source_commit": git_commit_sha(ROOT),
    }

    baseline_bundle, baseline_metrics, baseline_rows = _train_variant(train, test, baseline_features, 26103)
    lifecycle_bundle, lifecycle_metrics, lifecycle_rows = _train_variant(train, test, lifecycle_features, 26203)
    variants = {
        "without_revised_cost": [f for f in lifecycle_features if f not in {"revised_cost_cr", "cost_escalation_percentage"}],
        "snapshot_only": [f for f in lifecycle_features if f not in TRAJECTORY_FEATURES],
        "without_agency_priors": [f for f in lifecycle_features if not f.startswith("agency_")],
    }
    ablations = {}
    for name, features in variants.items():
        bundle, metrics, rows = _train_variant(train, test, features, 26303)
        ablations[name] = {"features": features, "selected_algorithms": bundle["selected_algorithms"],
                           "internal_algorithm_comparisons": bundle["internal_comparisons"],
                           "metrics": metrics, "lifecycle_stages": _stage_metrics(rows)}

    root = artifact_root or MODEL_ROOT
    target = root / f"{training_start}_{training_end}"; target.mkdir(parents=True, exist_ok=True)
    for name, model in lifecycle_bundle["models"].items():
        joblib.dump(model, target / f"{name}_model.pkl")
    write_feature_quality_report(audit, target / "feature_quality_report.json")
    importance = {name: _importance(model, train.tail(min(500, len(train))), lifecycle_features) for name, model in lifecycle_bundle["models"].items()}
    (target / "shap_importance.json").write_text(json.dumps(importance, indent=2))
    lifecycle_rows.to_csv(target / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
    ingestion_audit_path = ROOT / "data" / "processed" / "paimana_monthly_ingestion_audit.json"
    ingestion_audit = json.loads(ingestion_audit_path.read_text()) if ingestion_audit_path.exists() else {}
    lifecycle_stages = _stage_metrics(lifecycle_rows)
    stage_distribution = _stage_distribution(lifecycle_rows)
    balanced_stage = _balanced_stage_summary(lifecycle_stages)
    provenance["artifact_fingerprints"] = artifact_fingerprints(target, [
        "cost_model.pkl", "delay_model.pkl", "risk_model.pkl",
        "feature_quality_report.json", "shap_importance.json", "prediction_validation.csv",
    ])

    metadata = {
        "model_version": f"monthly-{training_start}-{training_end}",
        "run_id": run_id,
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "training_period": [training_start, training_end],
        "testing_period": [training_end + 1, test_end],
        "unique_training_projects": int(train.canonical_project_id.nunique()), "training_snapshots": int(len(train)),
        "unique_test_projects": int(test.canonical_project_id.nunique()), "test_snapshots": int(len(test)),
        "features_used": lifecycle_features, "feature_availability": audit, "data_source": "Official PAIMANA/MoSPI monthly Flash Reports only",
        "raw_archive_coverage": {key: ingestion_audit.get(key) for key in ["reports_discovered", "financial_years", "missing_months", "reports_parsed", "monthly_observations"]},
        "parser_versions": ingestion_audit.get("parser_report_counts", {}),
        "identity_verified_training_only": True,
        "leakage_policy": "Direct features are same-snapshot values, trajectory features use only current/earlier project snapshots, historical priors require completion_date < snapshot_date, and the future holdout is excluded from selection.",
        "as_of_invariants": {"training": train_invariants, "testing": test_invariants},
        "snapshot_weighting_policy": "Quarterly last-observation sampling followed by per-project weights summing exactly to one in the final sampled cohort.",
        "selected_algorithms": lifecycle_bundle["selected_algorithms"],
        "hyperparameters": {name: lifecycle_bundle["models"][name].named_steps["model"].get_params() for name in ["cost", "delay", "risk"]},
        "internal_algorithm_comparisons": lifecycle_bundle["internal_comparisons"], "baseline_metrics": baseline_metrics,
        "lifecycle_metrics": lifecycle_metrics, "lifecycle_stage_metrics": lifecycle_stages,
        "lifecycle_stage_distribution": stage_distribution,
        "balanced_stage_summary": balanced_stage,
        "evaluation_policy": "Report overall weighted metrics together with equal-stage macro diagnostics and early/mid metrics; overall scores must not be presented as early-warning accuracy.",
        "ablation_results": ablations,
        "shap_available": {name: value["method"] == "mean_absolute_shap" for name, value in importance.items()},
        "provenance": provenance,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata = _json_safe(metadata)
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    evolution = []
    for project_id, group in lifecycle_rows.groupby("canonical_project_id"):
        if len(group) < 2:
            continue
        evolution = [{"project_id": str(project_id), "project_name": row.project_name,
                      "snapshot_date": pd.Timestamp(row.snapshot_date).strftime("%Y-%m-%d"),
                      "predicted_cost_overrun": round(float(row.predicted_cost_overrun), 3),
                      "actual_cost_overrun": round(float(row.actual_cost_overrun_percentage), 3),
                      "predicted_delay_days": round(float(row.predicted_delay_days), 3),
                      "actual_delay_days": round(float(row.actual_delay_days), 3),
                      "predicted_risk": row.predicted_risk, "actual_risk": row.actual_risk}
                     for _, row in group.sort_values("snapshot_date").iterrows()]
        break
    result = {
        "window": f"{training_start}_{training_end}", "metadata": metadata,
        "identity_resolution": {"rows": int(len(identity)), "verified_rows": int(identity.identity_verified.sum()), "ambiguous_rows": int(identity.identity_method.eq("ambiguous_exact_name").sum())},
        "baseline": {"features": baseline_features, "metrics": baseline_metrics, "lifecycle_stages": _stage_metrics(baseline_rows)},
        "lifecycle": {"features": lifecycle_features, "metrics": lifecycle_metrics, "lifecycle_stages": lifecycle_stages,
                      "stage_distribution": stage_distribution, "balanced_stage_summary": balanced_stage},
        "ablations": ablations, "shap": importance, "forecast_evolution_example": evolution,
    }
    result = _json_safe(result)
    (target / "evaluation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False)); return result


def train_required_windows(data: pd.DataFrame | None = None, identity: pd.DataFrame | None = None) -> dict:
    if data is None or identity is None:
        data, identity = build_training_dataset()
    windows = [train_window(2001, 2015, 2021, data, identity), train_window(2015, 2021, 2024, data, identity)]
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "windows": windows}
    payload = _json_safe(payload)
    REPORTS.mkdir(parents=True, exist_ok=True); COMPARISON_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Monthly PAIMANA lifecycle model comparison", "", "All results use official monthly snapshots, exact/audited identity linkage, final-cohort project-balanced weighting and out-of-time test periods.", ""]
    for item in windows:
        balanced = item["lifecycle"].get("balanced_stage_summary") or {}
        lines.extend([f"## Window {item['window']}", "", "| Model | Cost MAE | Cost R2 | Delay MAE | Delay R2 | Risk macro F1 |", "|---|---:|---:|---:|---:|---:|",
                      f"| Five-feature baseline | {item['baseline']['metrics']['cost']['MAE']} | {item['baseline']['metrics']['cost']['R2']} | {item['baseline']['metrics']['delay']['MAE']} | {item['baseline']['metrics']['delay']['R2']} | {item['baseline']['metrics']['risk']['macro_f1']} |",
                      f"| Monthly lifecycle | {item['lifecycle']['metrics']['cost']['MAE']} | {item['lifecycle']['metrics']['cost']['R2']} | {item['lifecycle']['metrics']['delay']['MAE']} | {item['lifecycle']['metrics']['delay']['R2']} | {item['lifecycle']['metrics']['risk']['macro_f1']} |", "",
                      f"Equal-stage diagnostic: cost MAE {balanced.get('cost_mae')}, delay MAE {balanced.get('delay_mae')}, risk macro-F1 {balanced.get('risk_macro_f1')}. Early and mid metrics remain separate in the JSON report.", ""])
    COMPARISON_MD.write_text("\n".join(lines)); return payload
