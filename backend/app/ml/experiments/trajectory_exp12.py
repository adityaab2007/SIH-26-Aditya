"""Experiment 12: past-only trajectory-enhanced cost and delay forecasting."""
from __future__ import annotations

import json
import math

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
WINDOW_DAYS = {3: 93, 6: 186, 12: 366}
DAY_NS = 86_400_000_000_000
MONTH_NS = int(30.4375 * DAY_NS)

EXP12_FEATURES = [
    "exp12_history_12m", "exp12_cost_velocity_12m", "exp12_cost_revisions_12m",
    "exp12_months_since_cost_revision", "exp12_cost_volatility_6m",
    "exp12_expenditure_velocity_3m", "exp12_expenditure_velocity_6m",
    "exp12_expenditure_velocity_12m", "exp12_expenditure_acceleration",
    "exp12_slippage_velocity_3m", "exp12_slippage_velocity_6m",
    "exp12_slippage_velocity_12m", "exp12_slippage_acceleration",
    "exp12_schedule_revisions_12m", "exp12_months_since_schedule_revision",
    "exp12_slippage_volatility_6m",
]


def _values(group: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(group.get(name), errors="coerce").to_numpy(dtype=float)


def _left(dates: np.ndarray, days: int) -> np.ndarray:
    return np.searchsorted(dates, dates - days * DAY_NS, side="left")


def _velocity(dates: np.ndarray, values: np.ndarray, days: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values)); left = _left(dates, days)
    for i in valid:
        pos = int(np.searchsorted(valid, left[i], side="left"))
        if pos >= len(valid): continue
        j = int(valid[pos])
        if j >= i: continue
        months = (dates[i] - dates[j]) / MONTH_NS
        if months > 0: out[i] = (values[i] - values[j]) / months
    return out


def _events(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        good = np.isfinite(values[1:]) & np.isfinite(values[:-1])
        out[1:] = good & (np.abs(values[1:] - values[:-1]) > 1e-9)
    return out


def _event_count(dates: np.ndarray, events: np.ndarray, days: int) -> np.ndarray:
    left = _left(dates, days); prefix = np.r_[0, np.cumsum(events.astype(int))]; pos = np.arange(len(events))
    return prefix[pos + 1] - prefix[left]


def _months_since_event(dates: np.ndarray, events: np.ndarray) -> np.ndarray:
    out = np.full(len(events), np.nan); last = -1
    for i, changed in enumerate(events):
        if changed: last = i
        if last >= 0: out[i] = (dates[i] - dates[last]) / MONTH_NS
    return out


def _delta(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) > 1:
        good = np.isfinite(values[1:]) & np.isfinite(values[:-1]); pos = np.flatnonzero(good) + 1
        out[pos] = values[pos] - values[pos - 1]
    return out


def _rolling_std(dates: np.ndarray, values: np.ndarray, days: int) -> np.ndarray:
    finite = np.isfinite(values); safe = np.where(finite, values, 0.0)
    pc = np.r_[0, np.cumsum(finite.astype(int))]; ps = np.r_[0.0, np.cumsum(safe)]; pq = np.r_[0.0, np.cumsum(safe * safe)]
    left = _left(dates, days); pos = np.arange(len(values)); count = pc[pos + 1] - pc[left]
    total = ps[pos + 1] - ps[left]; square = pq[pos + 1] - pq[left]; out = np.full(len(values), np.nan); ok = count >= 2
    var = (square[ok] - total[ok] ** 2 / count[ok]) / (count[ok] - 1); out[ok] = np.sqrt(np.maximum(0.0, var))
    return out


def engineer_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"canonical_project_id", "snapshot_date"}; missing = required - set(history)
    if missing: raise ValueError("Experiment 12 history missing: " + ", ".join(sorted(missing)))
    frame = history.copy(); frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame.dropna(subset=["canonical_project_id", "snapshot_date"]).sort_values(["canonical_project_id", "snapshot_date"]).reset_index(drop=True)
    for feature in EXP12_FEATURES: frame[feature] = np.nan
    for _, group in frame.groupby("canonical_project_id", sort=False):
        idx = group.index; dates = group.snapshot_date.astype("int64").to_numpy(np.int64)
        cost = _values(group, "revised_cost_cr"); spend = _values(group, "cumulative_expenditure_cr"); slip = _values(group, "schedule_slippage_days")
        ce, se = _events(cost), _events(slip)
        cv3, cv6, cv12 = (_velocity(dates, cost, WINDOW_DAYS[w]) for w in (3, 6, 12))
        ev3, ev6, ev12 = (_velocity(dates, spend, WINDOW_DAYS[w]) for w in (3, 6, 12))
        sv3, sv6, sv12 = (_velocity(dates, slip, WINDOW_DAYS[w]) for w in (3, 6, 12))
        frame.loc[idx, "exp12_history_12m"] = np.arange(len(group)) - _left(dates, WINDOW_DAYS[12]) + 1
        frame.loc[idx, "exp12_cost_velocity_12m"] = cv12
        frame.loc[idx, "exp12_cost_revisions_12m"] = _event_count(dates, ce, WINDOW_DAYS[12])
        frame.loc[idx, "exp12_months_since_cost_revision"] = _months_since_event(dates, ce)
        frame.loc[idx, "exp12_cost_volatility_6m"] = _rolling_std(dates, _delta(cost), WINDOW_DAYS[6])
        frame.loc[idx, "exp12_expenditure_velocity_3m"] = ev3; frame.loc[idx, "exp12_expenditure_velocity_6m"] = ev6
        frame.loc[idx, "exp12_expenditure_velocity_12m"] = ev12; frame.loc[idx, "exp12_expenditure_acceleration"] = ev3 - ev6
        frame.loc[idx, "exp12_slippage_velocity_3m"] = sv3; frame.loc[idx, "exp12_slippage_velocity_6m"] = sv6
        frame.loc[idx, "exp12_slippage_velocity_12m"] = sv12; frame.loc[idx, "exp12_slippage_acceleration"] = sv3 - sv6
        frame.loc[idx, "exp12_schedule_revisions_12m"] = _event_count(dates, se, WINDOW_DAYS[12])
        frame.loc[idx, "exp12_months_since_schedule_revision"] = _months_since_event(dates, se)
        frame.loc[idx, "exp12_slippage_volatility_6m"] = _rolling_std(dates, _delta(slip), WINDOW_DAYS[6])
    return frame


def enrich_rows(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        if not TRAJECTORIES.exists(): raise FileNotFoundError("Experiment 12 requires paimana_project_trajectories.csv.")
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
    source = engineer_history(history)
    lookup = source[["canonical_project_id", "snapshot_date", *EXP12_FEATURES]].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    rows = supervised.copy(); rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    result = rows.merge(lookup, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")
    if len(result) != len(rows): raise AssertionError("Experiment 12 changed the supervised cohort.")
    return result


def _usable_features(train: pd.DataFrame) -> tuple[list[str], dict]:
    selected, audit = [], {}
    for name in EXP12_FEATURES:
        values = pd.to_numeric(train[name], errors="coerce"); pct = float(values.notna().mean() * 100.0)
        usable = pct >= 10.0 and values.dropna().nunique() > 1
        audit[name] = {"availability_percentage": round(pct, 3), "selected": bool(usable)}
        if usable: selected.append(name)
    return selected, audit


def _algorithm(bundle: dict, receipt: dict, target: str) -> str:
    name = ((bundle.get("metadata") or {}).get("selected_algorithms") or {}).get(target) or (receipt.get("selected_algorithms") or {}).get(target)
    if name in {"lightgbm", "xgboost", "extra_trees"}: return name
    lowered = type(bundle[target].named_steps["model"]).__name__.lower()
    if "lgbm" in lowered: return "lightgbm"
    if "xgb" in lowered: return "xgboost"
    if "extratrees" in lowered: return "extra_trees"
    raise ValueError(f"Cannot identify production {target} algorithm.")


def _metric(rows: pd.DataFrame, actual: str, predicted: str) -> dict:
    return _regression_metrics(rows[actual], rows[predicted].to_numpy(float), rows.sample_weight, rows.canonical_project_id)


def _stage(rows: pd.DataFrame, prefix: str) -> dict:
    result = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = rows[rows.lifecycle_stage.eq(stage)]
        result[stage] = {"available": False} if part.empty else {"available": True, "cost": _metric(part, "actual_cost_overrun_percentage", f"{prefix}_cost"), "delay": _metric(part, "actual_delay_days", f"{prefix}_delay")}
    return result


def _macro(stage: dict, target: str) -> float | None:
    values = [v[target]["MAE"] for v in stage.values() if v.get("available")]
    return round(float(np.mean(values)), 4) if values else None


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)): value = value.item()
    if isinstance(value, pd.Timestamp): return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None):
    frozen = data.copy(); frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce"); frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    base_train, base_test = temporal_project_split(frozen, training_start, training_end, test_end)
    enriched = enrich_rows(frozen, history); train, test = temporal_project_split(enriched, training_start, training_end, test_end)
    production_features = list((production_bundle.get("metadata") or {}).get("features_used") or production_receipt.get("features_used") or [])
    added, audit = _usable_features(train)
    if not production_features or not added: raise ValueError("Experiment 12 requires production features plus usable trajectory features.")
    features = list(dict.fromkeys(production_features + added)); cost_name = _algorithm(production_bundle, production_receipt, "cost"); delay_name = _algorithm(production_bundle, production_receipt, "delay")
    cost = _fit_pipeline(_regressors(26203)[cost_name], train, features, "actual_cost_overrun_percentage"); delay = _fit_pipeline(_regressors(26204)[delay_name], train, features, "actual_delay_days")
    compare = test[pd.to_numeric(test.exp12_history_12m, errors="coerce").fillna(0).ge(MIN_HISTORY)].copy()
    if compare.canonical_project_id.nunique() < 2: raise ValueError("Experiment 12 has too few future projects with usable history.")
    compare = assign_project_balanced_weights(compare)
    compare["production_cost"] = production_bundle["cost"].predict(compare[production_features]); compare["production_delay"] = np.maximum(0, production_bundle["delay"].predict(compare[production_features]))
    compare["experiment_cost"] = cost.predict(compare[features]); compare["experiment_delay"] = np.maximum(0, delay.predict(compare[features]))
    pc, ec = _metric(compare, "actual_cost_overrun_percentage", "production_cost"), _metric(compare, "actual_cost_overrun_percentage", "experiment_cost")
    pdm, edm = _metric(compare, "actual_delay_days", "production_delay"), _metric(compare, "actual_delay_days", "experiment_delay")
    paired_cost = paired_project_mae_comparison(compare, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost", candidate_prediction="experiment_cost")
    paired_delay = paired_project_mae_comparison(compare, actual="actual_delay_days", baseline_prediction="production_delay", candidate_prediction="experiment_delay", seed=26104)
    ps, es = _stage(compare, "production"), _stage(compare, "experiment")
    cost_gain = (pc["MAE"] - ec["MAE"]) / pc["MAE"] * 100 if pc["MAE"] else None; delay_gain = (pdm["MAE"] - edm["MAE"]) / pdm["MAE"] * 100 if pdm["MAE"] else None
    overall = {
        "production_cost_mae": pc["MAE"], "experiment_cost_mae": ec["MAE"], "absolute_mae_improvement_pp": round(pc["MAE"] - ec["MAE"], 4), "improvement_percentage": round(cost_gain, 4) if cost_gain is not None else None,
        "production_delay_mae": pdm["MAE"], "experiment_delay_mae": edm["MAE"], "absolute_delay_mae_improvement_days": round(pdm["MAE"] - edm["MAE"], 4), "delay_improvement_percentage": round(delay_gain, 4) if delay_gain is not None else None,
        "comparison_test_projects": int(compare.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(compare)), "paired_project_comparison": paired_cost,
        "paired_project_cost_comparison": paired_cost, "paired_project_delay_comparison": paired_delay, "production_stage_metrics": ps, "experiment_stage_metrics": es,
        "stage_balanced": {"production_cost_mae": _macro(ps, "cost"), "experiment_cost_mae": _macro(es, "cost"), "production_delay_mae": _macro(ps, "delay"), "experiment_delay_mae": _macro(es, "delay")},
    }
    context = build_experiment_context(experiment_id=EXPERIMENT_ID, full_data=frozen, train=base_train, test=base_test, features=features, training_start=training_start, training_end=training_end, testing_end=test_end, weighting_policy="project-balanced quarterly snapshots")
    manifest = new_experiment_manifest(context=context, name=EXPERIMENT_NAME, changed_dimension="feature_set", hypothesis="Past-only cost, expenditure and schedule trajectories reduce future cost and delay MAE.")
    manifest.update({"scope": EXPERIMENT_SCOPE, "production_run_id": production_receipt.get("run_id"), "production_features": production_features, "added_features": added, "feature_availability": audit, "selected_algorithms": {"cost": cost_name, "delay": delay_name}, "comparison_filter": f">={MIN_HISTORY} official observations in trailing 12 months", "leakage_policy": "Only current and earlier snapshots of the same canonical project are used."})
    run_dir = experiment_run_directory(EXPERIMENT_ID, context.window, manifest["run_id"]); run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(cost, run_dir / "cost_model.pkl"); joblib.dump(delay, run_dir / "delay_model.pkl"); (run_dir / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, allow_nan=False) + "\n"); (run_dir / "evaluation_results.json").write_text(json.dumps(_safe(overall), indent=2, allow_nan=False) + "\n")
    record_experiment({"experiment_id": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "run_id": manifest["run_id"], "status": "COMPLETED", "decision": "PENDING", "model_role": "experiment", "promotion_allowed": False, "scope": EXPERIMENT_SCOPE, "window": context.window, "created_at": manifest["created_at"], "production_run_id": production_receipt.get("run_id"), "cost_improvement_percentage": overall["improvement_percentage"], "delay_improvement_percentage": overall["delay_improvement_percentage"]})
    lookup_rows = enriched[["canonical_project_id", "snapshot_date", *added]].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    lookup = {(str(r.canonical_project_id), pd.Timestamp(r.snapshot_date).isoformat()): {f: r.get(f) for f in added} for _, r in lookup_rows.iterrows() if pd.notna(r.canonical_project_id) and pd.notna(r.snapshot_date)}
    comparable = {(str(r.canonical_project_id), pd.Timestamp(r.snapshot_date).isoformat()) for _, r in compare.iterrows()}
    experiment = {"experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "run_id": manifest["run_id"], "model_role": "experiment", "scope": EXPERIMENT_SCOPE, "decision": "PENDING", "promotion_allowed": False, "feature_count": len(features), "production_feature_count": len(production_features), "added_feature_count": len(added), "added_features": added, "selected_algorithms": {"cost": cost_name, "delay": delay_name}, "metrics": {"cost": ec, "delay": edm}, "leakage_policy": manifest["leakage_policy"]}
    return {"experiment": experiment, "overall_comparison": overall, "runtime_state": {"cost_model": cost, "delay_model": delay, "features": features, "added": added, "lookup": lookup, "comparable": comparable}}


def _key(row: pd.Series):
    date = pd.to_datetime(row.get("snapshot_date"), errors="coerce"); project = row.get("canonical_project_id")
    if pd.isna(project) or pd.isna(date): return None
    return str(project), pd.Timestamp(date).isoformat()


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]: raise ValueError("No Experiment 12 trajectory is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items(): candidate[name] = value
    X = candidate.to_frame().T.reindex(columns=state["features"])
    return {"predicted_cost_overrun": round(float(state["cost_model"].predict(X)[0]), 4), "predicted_delay_days": round(max(0.0, float(state["delay_model"].predict(X)[0])), 4), "trajectory_features_available": int(sum(pd.notna(candidate.get(f)) for f in state["added"])), "trajectory_feature_count": len(state["added"])}
