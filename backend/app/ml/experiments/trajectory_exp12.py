"""Experiment 12 legacy trajectory feature engineering and v1 implementation.

The active Experiment 12 adapter now points to ``trajectory_exp12_cost.py``.
This module is retained because v2/v3 reuse its leakage-safe trajectory helpers
and because it preserves the original cost+delay experiment history.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import build_experiment_context, experiment_run_directory, new_experiment_manifest
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split

EXPERIMENT_ID = "exp_12"
EXPERIMENT_NAME = "Trajectory-enhanced lifecycle forecasting"
EXPERIMENT_SCOPE = "cost_delay"
MIN_HISTORY = 2
WINDOW_DAYS = {3: 92, 6: 183, 12: 365}

EXP12_FEATURES = [
    "exp12_history_12m",
    "exp12_cost_velocity_12m",
    "exp12_cost_revisions_12m",
    "exp12_months_since_cost_revision",
    "exp12_cost_volatility_6m",
    "exp12_expenditure_velocity_3m",
    "exp12_expenditure_velocity_6m",
    "exp12_expenditure_velocity_12m",
    "exp12_expenditure_acceleration",
    "exp12_slippage_velocity_3m",
    "exp12_slippage_velocity_6m",
    "exp12_slippage_velocity_12m",
    "exp12_slippage_acceleration",
    "exp12_schedule_revisions_12m",
    "exp12_months_since_schedule_revision",
    "exp12_slippage_volatility_6m",
]


def _safe(obj):
    if isinstance(obj, dict): return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float): return None if not math.isfinite(obj) else obj
    return obj


def _values(group, column):
    if column not in group.columns:
        return np.full(len(group), np.nan)
    return pd.to_numeric(group[column], errors="coerce").to_numpy(float)


def _velocity(dates, values, days):
    out = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    for i in range(len(values)):
        if not valid[i]: continue
        target = dates[i] - days * 86_400_000_000_000
        earlier = np.where(valid[:i] & (dates[:i] <= target))[0]
        if not len(earlier): continue
        j = earlier[-1]
        months = max((dates[i] - dates[j]) / 86_400_000_000_000 / 30.4375, 1e-6)
        out[i] = (values[i] - values[j]) / months
    return out


def _delta(values):
    out = np.full(len(values), np.nan)
    previous = np.nan
    for i, value in enumerate(values):
        if np.isfinite(value) and np.isfinite(previous): out[i] = value - previous
        if np.isfinite(value): previous = value
    return out


def _rolling_std(dates, values, days):
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        start = dates[i] - days * 86_400_000_000_000
        sample = values[(dates >= start) & (dates <= dates[i]) & np.isfinite(values)]
        if len(sample) >= 2: out[i] = float(np.std(sample, ddof=0))
    return out


def _event_count(dates, events, days):
    out = np.zeros(len(dates), dtype=float)
    for i in range(len(dates)):
        start = dates[i] - days * 86_400_000_000_000
        out[i] = float(np.sum(events & (dates >= start) & (dates <= dates[i])))
    return out


def _months_since_event(dates, events):
    out = np.full(len(dates), np.nan)
    last = None
    for i in range(len(dates)):
        if events[i]: last = dates[i]
        if last is not None: out[i] = (dates[i] - last) / 86_400_000_000_000 / 30.4375
    return out


def engineer_history(history):
    frame = history.copy()
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame[frame.canonical_project_id.notna() & frame.snapshot_date.notna()].copy()
    frame = frame.sort_values(["canonical_project_id", "snapshot_date"]).reset_index(drop=True)
    for feature in EXP12_FEATURES: frame[feature] = np.nan
    for _, group in frame.groupby("canonical_project_id", sort=False):
        idx = group.index
        dates = group.snapshot_date.astype("int64").to_numpy(np.int64)
        cost = _values(group, "revised_cost_cr")
        spend = _values(group, "cumulative_expenditure_cr")
        slip = _values(group, "schedule_slippage_days")
        cost_delta = _delta(cost); slip_delta = _delta(slip)
        cost_events = np.isfinite(cost_delta) & (np.abs(cost_delta) > 1e-9)
        slip_events = np.isfinite(slip_delta) & (np.abs(slip_delta) > 1e-9)
        v3_spend, v6_spend, v12_spend = (_velocity(dates, spend, WINDOW_DAYS[w]) for w in (3, 6, 12))
        v3_slip, v6_slip, v12_slip = (_velocity(dates, slip, WINDOW_DAYS[w]) for w in (3, 6, 12))
        history_count = np.array([np.sum((dates >= d - WINDOW_DAYS[12] * 86_400_000_000_000) & (dates <= d)) for d in dates], float)
        frame.loc[idx, "exp12_history_12m"] = history_count
        frame.loc[idx, "exp12_cost_velocity_12m"] = _velocity(dates, cost, WINDOW_DAYS[12])
        frame.loc[idx, "exp12_cost_revisions_12m"] = _event_count(dates, cost_events, WINDOW_DAYS[12])
        frame.loc[idx, "exp12_months_since_cost_revision"] = _months_since_event(dates, cost_events)
        frame.loc[idx, "exp12_cost_volatility_6m"] = _rolling_std(dates, cost_delta, WINDOW_DAYS[6])
        frame.loc[idx, "exp12_expenditure_velocity_3m"] = v3_spend
        frame.loc[idx, "exp12_expenditure_velocity_6m"] = v6_spend
        frame.loc[idx, "exp12_expenditure_velocity_12m"] = v12_spend
        frame.loc[idx, "exp12_expenditure_acceleration"] = v3_spend - v6_spend
        frame.loc[idx, "exp12_slippage_velocity_3m"] = v3_slip
        frame.loc[idx, "exp12_slippage_velocity_6m"] = v6_slip
        frame.loc[idx, "exp12_slippage_velocity_12m"] = v12_slip
        frame.loc[idx, "exp12_slippage_acceleration"] = v3_slip - v6_slip
        frame.loc[idx, "exp12_schedule_revisions_12m"] = _event_count(dates, slip_events, WINDOW_DAYS[12])
        frame.loc[idx, "exp12_months_since_schedule_revision"] = _months_since_event(dates, slip_events)
        frame.loc[idx, "exp12_slippage_volatility_6m"] = _rolling_std(dates, slip_delta, WINDOW_DAYS[6])
    return frame


def enrich_rows(supervised, history=None):
    if history is None:
        if not TRAJECTORIES.exists(): raise FileNotFoundError("Experiment 12 requires paimana_project_trajectories.csv.")
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
    source = engineer_history(history)
    lookup = source[["canonical_project_id", "snapshot_date", *EXP12_FEATURES]].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    rows = supervised.copy(); rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    result = rows.merge(lookup, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")
    if len(result) != len(rows): raise AssertionError("Experiment 12 changed the supervised cohort.")
    return result


def _algorithm(bundle, receipt, target):
    selected = (bundle.get("metadata") or {}).get("selected_algorithms") or receipt.get("selected_algorithms") or {}
    return selected.get(target) or "extra_trees"


def _metric(frame, target, prediction):
    return _regression_metrics(frame[target], frame[prediction], frame.sample_weight, frame.canonical_project_id)


def _stage(frame, prefix):
    out = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = frame[frame.lifecycle_stage.eq(stage)]
        out[stage] = {"available": False} if part.empty else {
            "available": True,
            "cost": _metric(part, "actual_cost_overrun_percentage", f"{prefix}_cost"),
            "delay": _metric(part, "actual_delay_days", f"{prefix}_delay"),
        }
    return out


def _macro(stage, target):
    values = [v[target]["MAE"] for v in stage.values() if v.get("available")]
    return round(float(np.mean(values)), 4) if values else None


def _key(row): return (str(row.get("canonical_project_id")), pd.Timestamp(row.get("snapshot_date")).isoformat())


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None):
    frozen = data.copy(); frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce"); frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    base_train, base_test = temporal_project_split(frozen, training_start, training_end, test_end)
    enriched = enrich_rows(frozen, history)
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    production_features = list((production_bundle.get("metadata") or {}).get("features_used") or production_receipt.get("features_used") or [])
    usable, audit = [], {}
    for name in EXP12_FEATURES:
        values = pd.to_numeric(train[name], errors="coerce"); pct = float(values.notna().mean() * 100.0)
        ok = pct >= 10.0 and values.dropna().nunique() > 1
        audit[name] = {"availability_percentage": round(pct, 3), "selected": bool(ok)}
        if ok: usable.append(name)
    if not production_features or not usable: raise ValueError("Experiment 12 requires production features plus usable trajectory features.")
    features = list(dict.fromkeys(production_features + usable))
    cost_name = _algorithm(production_bundle, production_receipt, "cost"); delay_name = _algorithm(production_bundle, production_receipt, "delay")
    cost = _fit_pipeline(_regressors(26203)[cost_name], train, features, "actual_cost_overrun_percentage")
    delay = _fit_pipeline(_regressors(26204)[delay_name], train, features, "actual_delay_days")
    compare = test[pd.to_numeric(test.exp12_history_12m, errors="coerce").fillna(0).ge(MIN_HISTORY)].copy()
    if compare.canonical_project_id.nunique() < 2: raise ValueError("Experiment 12 has too few future projects with usable history.")
    compare = assign_project_balanced_weights(compare)
    compare["production_cost"] = production_bundle["cost"].predict(compare[production_features])
    compare["production_delay"] = np.maximum(0, production_bundle["delay"].predict(compare[production_features]))
    compare["experiment_cost"] = cost.predict(compare[features])
    compare["experiment_delay"] = np.maximum(0, delay.predict(compare[features]))
    pc, ec = _metric(compare, "actual_cost_overrun_percentage", "production_cost"), _metric(compare, "actual_cost_overrun_percentage", "experiment_cost")
    pdm, edm = _metric(compare, "actual_delay_days", "production_delay"), _metric(compare, "actual_delay_days", "experiment_delay")
    paired_cost = paired_project_mae_comparison(compare, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost", candidate_prediction="experiment_cost")
    paired_delay = paired_project_mae_comparison(compare, actual="actual_delay_days", baseline_prediction="production_delay", candidate_prediction="experiment_delay", seed=26104)
    ps, es = _stage(compare, "production"), _stage(compare, "experiment")
    cost_gain = (pc["MAE"] - ec["MAE"]) / pc["MAE"] * 100 if pc["MAE"] else None
    delay_gain = (pdm["MAE"] - edm["MAE"]) / pdm["MAE"] * 100 if pdm["MAE"] else None
    overall = {
        "production_cost_mae": pc["MAE"], "experiment_cost_mae": ec["MAE"],
        "absolute_mae_improvement_pp": round(pc["MAE"] - ec["MAE"], 4), "improvement_percentage": round(cost_gain, 4) if cost_gain is not None else None,
        "production_delay_mae": pdm["MAE"], "experiment_delay_mae": edm["MAE"],
        "absolute_delay_mae_improvement_days": round(pdm["MAE"] - edm["MAE"], 4), "delay_improvement_percentage": round(delay_gain, 4) if delay_gain is not None else None,
        "comparison_test_projects": int(compare.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(compare)),
        "paired_project_comparison": paired_cost, "paired_project_cost_comparison": paired_cost, "paired_project_delay_comparison": paired_delay,
        "production_stage_metrics": ps, "experiment_stage_metrics": es,
        "stage_balanced": {"production_cost_mae": _macro(ps, "cost"), "experiment_cost_mae": _macro(es, "cost"), "production_delay_mae": _macro(ps, "delay"), "experiment_delay_mae": _macro(es, "delay")},
    }
    context = build_experiment_context(experiment_id=EXPERIMENT_ID, full_data=frozen, train=base_train, test=base_test, features=features, training_start=training_start, training_end=training_end, testing_end=test_end, weighting_policy="project-balanced quarterly snapshots")
    manifest = new_experiment_manifest(context=context, name=EXPERIMENT_NAME, changed_dimension="feature_set", hypothesis="Richer past-only trajectories improve future cost and delay MAE without changing algorithms or targets.")
    manifest.update({"scope": EXPERIMENT_SCOPE, "production_run_id": production_receipt.get("run_id"), "production_features": production_features, "added_features": usable, "feature_availability": audit, "selected_algorithms": {"cost": cost_name, "delay": delay_name}, "comparison_filter": f">={MIN_HISTORY} official observations in trailing 12 months", "leakage_policy": "Every trajectory feature uses only the current or earlier official snapshot for the same canonical project; later snapshots and final outcomes are never consulted."})
    run_dir = experiment_run_directory(EXPERIMENT_ID, context.window, manifest["run_id"]); run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(cost, run_dir / "cost_model.pkl"); joblib.dump(delay, run_dir / "delay_model.pkl")
    (run_dir / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, allow_nan=False) + "\n")
    (run_dir / "evaluation_results.json").write_text(json.dumps(_safe(overall), indent=2, allow_nan=False) + "\n")
    record_experiment({"experiment_id": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "run_id": manifest["run_id"], "status": "COMPLETED", "decision": "PENDING", "model_role": "experiment", "promotion_allowed": False, "scope": EXPERIMENT_SCOPE, "window": context.window, "created_at": manifest["created_at"], "production_run_id": production_receipt.get("run_id"), "cost_improvement_percentage": overall["improvement_percentage"], "delay_improvement_percentage": overall["delay_improvement_percentage"]})
    lookup_rows = enriched[["canonical_project_id", "snapshot_date", *usable]].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    lookup = {(str(r.canonical_project_id), pd.Timestamp(r.snapshot_date).isoformat()): {f: r.get(f) for f in usable} for _, r in lookup_rows.iterrows() if pd.notna(r.canonical_project_id) and pd.notna(r.snapshot_date)}
    comparable = {(str(r.canonical_project_id), pd.Timestamp(r.snapshot_date).isoformat()) for _, r in compare.iterrows()}
    experiment = {"experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "run_id": manifest["run_id"], "model_role": "experiment", "scope": EXPERIMENT_SCOPE, "decision": "PENDING", "promotion_allowed": False, "feature_count": len(features), "production_feature_count": len(production_features), "added_feature_count": len(usable), "added_features": usable, "selected_algorithms": {"cost": cost_name, "delay": delay_name}, "metrics": {"cost": ec, "delay": edm}, "leakage_policy": manifest["leakage_policy"]}
    return {"experiment": experiment, "overall_comparison": overall, "runtime_state": {"cost_model": cost, "delay_model": delay, "features": features, "added": usable, "lookup": lookup, "comparable": comparable}}


def filter_comparable_rows(frame, state): return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row, state):
    key = _key(row)
    if key not in state["lookup"]: raise ValueError("No Experiment 12 trajectory is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items(): candidate[name] = value
    X = candidate.to_frame().T.reindex(columns=state["features"])
    return {"predicted_cost_overrun": round(float(state["cost_model"].predict(X)[0]), 4), "predicted_delay_days": round(max(0.0, float(state["delay_model"].predict(X)[0])), 4), "trajectory_features_available": int(sum(pd.notna(candidate.get(f)) for f in state["added"])), "trajectory_feature_count": len(state["added"])}
