"""Experiment 4 v2: fairness-corrected and partially pooled early forecasting.

This module is deliberately separate from the frozen v1 artifact namespace.
All residual targets are made from expanding temporal predictions, never from
the in-sample prediction of the row being used as a training target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.ml.experiments.lifecycle_specialists import (
    EXPERIMENT_ROOT, STAGES, _metrics, _overall, _regression_metrics,
    _valid_features, add_lifecycle_stages, renormalize_stage_weights,
    train_lifecycle_specialists,
)
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES, CANDIDATE_FEATURES, IMPROVED_EARLY_FEATURES,
    IDENTITY_AUDIT, TARGETS, TRAINING_DATA, as_of_feature_evidence,
    build_training_dataset,
)
from backend.app.ml.monthly_training import _fit_pipeline, _regressors, _train_variant, temporal_project_split
from backend.app.ml.feature_audit import audit_features


IMPROVED_ROOT = EXPERIMENT_ROOT.parent / "lifecycle_specialists_improved"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _features(train: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    audit = audit_features(train, CANDIDATE_FEATURES, minimum_availability=10, minimum_year_coverage=2, as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES))
    return _valid_features(list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"])), train), audit


def _expanding_residuals(train: pd.DataFrame, features: list[str], selected: dict[str, str]) -> pd.DataFrame:
    """Create project-disjoint temporal OOF global predictions for residual targets."""
    rows = []
    years = sorted(pd.to_numeric(train.completion_year, errors="coerce").dropna().unique())
    # Three expanding folds are sufficient to establish temporal OOF
    # provenance while keeping the experiment deliberately small/reproducible.
    for year in years[-3:]:
        fitting = train[train.completion_year < year]
        fold = train[train.completion_year.eq(year)]
        if fitting.canonical_project_id.nunique() < 5 or fold.empty:
            continue
        prediction = fold[["canonical_project_id", "snapshot_date", "completion_year", "lifecycle_stage", "actual_cost_overrun_percentage", "actual_delay_days", "sample_weight"]].copy()
        for target, key in (("actual_cost_overrun_percentage", "cost"), ("actual_delay_days", "delay")):
            model = _regressors(71000 + int(year))[selected[key]]
            model = _fit_pipeline(model, fitting, features, target)
            prediction[f"global_{key}"] = model.predict(fold[features])
        rows.append(prediction)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _select_global_algorithms(train: pd.DataFrame, features: list[str]) -> dict[str, str]:
    # Selection happens once on the training period's final internal year.
    # The 2016-2025 holdout is never inspected here.
    from backend.app.ml.experiments.lifecycle_specialists import _strict_selection
    return {target: _strict_selection(train, features, target_name, 72000 + index)[0] for index, (target, target_name) in enumerate((("cost", "actual_cost_overrun_percentage"), ("delay", "actual_delay_days")))}


def _fit_residual_models(oof: pd.DataFrame, train: pd.DataFrame, features: list[str], selected: dict[str, str]) -> dict[str, Any]:
    early = train[train.lifecycle_stage.eq("early")].copy()
    residual = oof[oof.lifecycle_stage.eq("early")].copy()
    models = {}
    for key, target in (("cost", "cost_residual"), ("delay", "delay_residual")):
        if residual.empty:
            continue
        residual[target] = residual[f"actual_{key if key == 'cost' else 'delay'}" if False else ("actual_cost_overrun_percentage" if key == "cost" else "actual_delay_days")] - residual[f"global_{key}"]
        # Rejoin features by stable project/snapshot keys; no future row is used.
        fit = early.merge(residual[["canonical_project_id", "snapshot_date", target]], on=["canonical_project_id", "snapshot_date"], how="inner")
        fit = renormalize_stage_weights(fit)
        models[key] = _fit_pipeline(_regressors(73000)[selected[key]], fit, features, target)
    return models


def _choose_alphas(train: pd.DataFrame, features: list[str], selected: dict[str, str], oof: pd.DataFrame) -> tuple[dict[str, float], dict[str, list[dict[str, float]]]]:
    early_oof = oof[oof.lifecycle_stage.eq("early")].copy()
    if early_oof.empty:
        return {"cost": 0.0, "delay": 0.0}, {"cost": [], "delay": []}
    # Use the latest available internal year as validation and earlier OOF
    # residuals as correction training data.
    validation_year = int(early_oof.completion_year.max())
    # Fit corrections only on earlier temporal OOF folds and score alpha on
    # the final internal fold. This prevents residual-model overfit from
    # selecting the shrinkage strength.
    models = _fit_residual_models(oof[oof.completion_year.lt(validation_year)], train, features, selected)
    choices = {}
    diagnostics = {}
    for key, actual in (("cost", "actual_cost_overrun_percentage"), ("delay", "actual_delay_days")):
        if key not in models:
            choices[key] = 0.0; diagnostics[key] = []; continue
        fit = train[train.lifecycle_stage.eq("early")].merge(oof[oof.completion_year.eq(validation_year)][["canonical_project_id", "snapshot_date", f"global_{key}"]], on=["canonical_project_id", "snapshot_date"], how="inner")
        residual_pred = models[key].predict(fit[features])
        scores = []
        for alpha in ALPHAS:
            pred = fit[f"global_{key}"].to_numpy(float) + alpha * residual_pred
            scores.append({"alpha": alpha, "MAE": float(mean_abs_error(fit[actual], pred, fit.sample_weight))})
        diagnostics[key] = scores
        choices[key] = min(scores, key=lambda item: item["MAE"])["alpha"]
    return choices, diagnostics


def mean_abs_error(actual: pd.Series, predicted: np.ndarray, weights: pd.Series) -> float:
    mask = actual.notna() & np.isfinite(predicted)
    return float(np.average(np.abs(actual[mask].to_numpy(float) - np.asarray(predicted)[mask]), weights=weights[mask].to_numpy(float)))


def run_improved_experiment(training_start: int, training_end: int, test_end: int, data: pd.DataFrame | None = None, artifact_root: Path | None = None) -> dict[str, Any]:
    """Run v2A, improved independent specialists, partial pooling and hybrid."""
    if data is None:
        data, _ = build_training_dataset()
    data = add_lifecycle_stages(data.copy())
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    train, test = temporal_project_split(data, training_start, training_end, test_end)
    root = artifact_root or IMPROVED_ROOT
    root.mkdir(parents=True, exist_ok=True)
    v2a_path = root / "v2a" / f"{training_start}_{training_end}" / "comparison.json"
    v2a = json.loads(v2a_path.read_text()) if v2a_path.exists() else train_lifecycle_specialists(training_start, training_end, test_end, data=data, artifact_root=root / "v2a", include_improved_features=False)
    improved_features, audit = _features(train)
    independent_path = root / "independent" / f"{training_start}_{training_end}" / "comparison.json"
    improved = json.loads(independent_path.read_text()) if independent_path.exists() else train_lifecycle_specialists(training_start, training_end, test_end, data=data, artifact_root=root / "independent", include_improved_features=True)
    selected = _select_global_algorithms(train, improved_features)
    oof = _expanding_residuals(train, improved_features, selected)
    models = _fit_residual_models(oof, train, improved_features, selected)
    alphas, alpha_scores = _choose_alphas(train, improved_features, selected, oof)
    global_bundle, global_metrics, global_rows = _train_variant(train, test, improved_features, 74000, selected=selected)
    hybrid = improved["specialist_overall"]
    hybrid_rows = pd.read_csv(root / "independent" / f"{training_start}_{training_end}" / "routed_predictions.csv")
    hybrid_rows["snapshot_date"] = pd.to_datetime(hybrid_rows.snapshot_date)
    early = test[test.lifecycle_stage.eq("early")].copy()
    early["snapshot_date"] = pd.to_datetime(early.snapshot_date)
    early = early.merge(hybrid_rows[["canonical_project_id", "snapshot_date", "predicted_cost_overrun", "predicted_delay_days"]], on=["canonical_project_id", "snapshot_date"], suffixes=("", "_routed"))
    for key, actual, alpha in (("cost", "actual_cost_overrun_percentage", alphas["cost"]), ("delay", "actual_delay_days", alphas["delay"])):
        if key in models:
            correction = models[key].predict(early[improved_features])
            global_pred = global_bundle["models"][key].predict(early[improved_features])
            early[f"hybrid_{key}"] = global_pred + alpha * correction
            if key == "delay": early[f"hybrid_{key}"] = np.maximum(0, early[f"hybrid_{key}"])
        else:
            early[f"hybrid_{key}"] = early["predicted_cost_overrun" if key == "cost" else "predicted_delay_days"]
    hybrid_rows = hybrid_rows.merge(early[["canonical_project_id", "snapshot_date", "hybrid_cost", "hybrid_delay"]], on=["canonical_project_id", "snapshot_date"], how="left")
    hybrid_rows["predicted_cost_overrun"] = hybrid_rows.hybrid_cost.fillna(hybrid_rows.predicted_cost_overrun)
    hybrid_rows["predicted_delay_days"] = hybrid_rows.hybrid_delay.fillna(hybrid_rows.predicted_delay_days)
    hybrid = _overall(hybrid_rows)
    result = {"experiment": "experiment_4_improved", "v2a_fairness_only": v2a, "improved_independent": improved, "partial_pooling": {"alphas": alphas, "internal_alpha_scores": alpha_scores, "early": {"cost": _metrics(early.actual_cost_overrun_percentage, early.hybrid_cost.to_numpy(), early), "delay": _metrics(early.actual_delay_days, early.hybrid_delay.to_numpy(), early)}}, "hybrid": hybrid, "global_baseline": {"metrics": global_metrics}, "features": improved_features, "feature_audit": audit, "selected_algorithms": selected, "physical_progress_limitation": "Physical progress is retained as missing where unavailable; no synthetic progress is fabricated.", "holdout_policy": "2016-2025 is used only for final evaluation; algorithm and alpha selection use training-period temporal validation/OOF predictions."}
    (root / "comparison.json").write_text(json.dumps(result, indent=2, default=str))
    return result
