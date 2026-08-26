"""Experiment 12 v2: scale-aware, target-specific trajectory forecasting.

This module deliberately selects trajectory feature groups using only an
internal historical validation block inside the requested training window. The
future holdout is never consulted for feature-group selection.
"""
from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import build_experiment_context, experiment_run_directory, new_experiment_manifest
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.experiments.trajectory_exp12 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    MIN_HISTORY,
    WINDOW_DAYS,
    _algorithm,
    _delta,
    _event_count,
    _key,
    _macro,
    _metric,
    _months_since_event,
    _rolling_std,
    _safe,
    _stage,
    _values,
    _velocity,
    engineer_history as engineer_history_v1,
)
from backend.app.ml.monthly_lifecycle import TRAJECTORIES, assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split

V2_FEATURES = [
    "exp12_cost_growth_pct_3m",
    "exp12_cost_growth_pct_6m",
    "exp12_cost_growth_pct_12m",
    "exp12_cost_growth_pct_acceleration",
    "exp12_cost_revision_magnitude_12m_pct",
    "exp12_cost_worsening_streak",
    "exp12_expenditure_ratio_velocity_3m",
    "exp12_expenditure_ratio_velocity_6m",
    "exp12_expenditure_ratio_velocity_12m",
    "exp12_expenditure_ratio_acceleration",
    "exp12_spend_vs_expected_progress_gap",
    "exp12_slippage_ratio_velocity_3m",
    "exp12_slippage_ratio_velocity_6m",
    "exp12_slippage_ratio_velocity_12m",
    "exp12_slippage_ratio_acceleration",
    "exp12_schedule_revision_magnitude_12m_pct",
    "exp12_slippage_worsening_streak",
]

V1_COST = [
    "exp12_history_12m", "exp12_cost_velocity_12m", "exp12_cost_revisions_12m",
    "exp12_months_since_cost_revision", "exp12_cost_volatility_6m",
]
V1_SPEND = [
    "exp12_expenditure_velocity_3m", "exp12_expenditure_velocity_6m",
    "exp12_expenditure_velocity_12m", "exp12_expenditure_acceleration",
]
V1_SCHEDULE = [
    "exp12_slippage_velocity_3m", "exp12_slippage_velocity_6m",
    "exp12_slippage_velocity_12m", "exp12_slippage_acceleration",
    "exp12_schedule_revisions_12m", "exp12_months_since_schedule_revision",
    "exp12_slippage_volatility_6m",
]

COST_SIGNAL = V1_COST + [
    "exp12_cost_growth_pct_3m", "exp12_cost_growth_pct_6m",
    "exp12_cost_growth_pct_12m", "exp12_cost_growth_pct_acceleration",
    "exp12_cost_revision_magnitude_12m_pct", "exp12_cost_worsening_streak",
]
SPEND_SIGNAL = V1_SPEND + [
    "exp12_expenditure_ratio_velocity_3m", "exp12_expenditure_ratio_velocity_6m",
    "exp12_expenditure_ratio_velocity_12m", "exp12_expenditure_ratio_acceleration",
    "exp12_spend_vs_expected_progress_gap",
]
SCHEDULE_SIGNAL = V1_SCHEDULE + [
    "exp12_slippage_ratio_velocity_3m", "exp12_slippage_ratio_velocity_6m",
    "exp12_slippage_ratio_velocity_12m", "exp12_slippage_ratio_acceleration",
    "exp12_schedule_revision_magnitude_12m_pct", "exp12_slippage_worsening_streak",
]


def _rolling_sum(dates: np.ndarray, values: np.ndarray, days: int) -> np.ndarray:
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    prefix = np.r_[0.0, np.cumsum(safe)]
    left = np.searchsorted(dates, dates - days * 86_400_000_000_000, side="left")
    pos = np.arange(len(values))
    out = prefix[pos + 1] - prefix[left]
    out[(np.r_[0, np.cumsum(finite.astype(int))][pos + 1] - np.r_[0, np.cumsum(finite.astype(int))][left]) == 0] = np.nan
    return out


def _positive_streak(values: np.ndarray) -> np.ndarray:
    changes = _delta(values)
    out = np.zeros(len(values), dtype=float)
    streak = 0
    for i, change in enumerate(changes):
        if np.isfinite(change) and change > 1e-9:
            streak += 1
        elif np.isfinite(change):
            streak = 0
        out[i] = streak
    return out


def _effective_schedule_events(group: pd.DataFrame) -> np.ndarray:
    revised = pd.to_datetime(group.get("revised_completion_date"), errors="coerce")
    planned = pd.to_datetime(group.get("planned_completion_date"), errors="coerce")
    effective = revised.fillna(planned)
    out = np.zeros(len(group), dtype=bool)
    previous = None
    for i, value in enumerate(effective):
        if pd.isna(value):
            continue
        stamp = pd.Timestamp(value)
        if previous is not None and stamp != previous:
            out[i] = True
        previous = stamp
    return out


def engineer_history(history: pd.DataFrame) -> pd.DataFrame:
    """Extend v1 with normalized rates, persistence and revision magnitudes."""
    frame = engineer_history_v1(history)
    for feature in V2_FEATURES:
        frame[feature] = np.nan

    for _, group in frame.groupby("canonical_project_id", sort=False):
        idx = group.index
        dates = group.snapshot_date.astype("int64").to_numpy(np.int64)
        approved = _values(group, "approved_cost_cr")
        revised = _values(group, "revised_cost_cr")
        expenditure = _values(group, "cumulative_expenditure_cr")
        slippage = _values(group, "schedule_slippage_days")
        planned_duration = _values(group, "planned_duration_days")
        expected_progress = _values(group, "expected_progress_percentage")

        valid_cost = (approved > 0) & np.isfinite(revised)
        cost_pct = np.full(len(group), np.nan)
        cost_pct[valid_cost] = (revised[valid_cost] - approved[valid_cost]) / approved[valid_cost] * 100.0

        valid_spend = (approved > 0) & np.isfinite(expenditure)
        spend_pct = np.full(len(group), np.nan)
        spend_pct[valid_spend] = expenditure[valid_spend] / approved[valid_spend] * 100.0

        valid_schedule = (planned_duration > 0) & np.isfinite(slippage)
        slip_pct = np.full(len(group), np.nan)
        slip_pct[valid_schedule] = slippage[valid_schedule] / planned_duration[valid_schedule] * 100.0

        cost_v3, cost_v6, cost_v12 = (_velocity(dates, cost_pct, WINDOW_DAYS[w]) for w in (3, 6, 12))
        spend_v3, spend_v6, spend_v12 = (_velocity(dates, spend_pct, WINDOW_DAYS[w]) for w in (3, 6, 12))
        slip_v3, slip_v6, slip_v12 = (_velocity(dates, slip_pct, WINDOW_DAYS[w]) for w in (3, 6, 12))

        cost_change = np.abs(_delta(cost_pct))
        slip_change = np.abs(_delta(slip_pct))
        schedule_events = _effective_schedule_events(group)

        frame.loc[idx, "exp12_cost_growth_pct_3m"] = cost_v3
        frame.loc[idx, "exp12_cost_growth_pct_6m"] = cost_v6
        frame.loc[idx, "exp12_cost_growth_pct_12m"] = cost_v12
        frame.loc[idx, "exp12_cost_growth_pct_acceleration"] = cost_v3 - cost_v6
        frame.loc[idx, "exp12_cost_revision_magnitude_12m_pct"] = _rolling_sum(dates, cost_change, WINDOW_DAYS[12])
        frame.loc[idx, "exp12_cost_worsening_streak"] = _positive_streak(cost_pct)

        frame.loc[idx, "exp12_expenditure_ratio_velocity_3m"] = spend_v3
        frame.loc[idx, "exp12_expenditure_ratio_velocity_6m"] = spend_v6
        frame.loc[idx, "exp12_expenditure_ratio_velocity_12m"] = spend_v12
        frame.loc[idx, "exp12_expenditure_ratio_acceleration"] = spend_v3 - spend_v6
        gap = np.where(np.isfinite(spend_pct) & np.isfinite(expected_progress), spend_pct - expected_progress, np.nan)
        frame.loc[idx, "exp12_spend_vs_expected_progress_gap"] = gap

        frame.loc[idx, "exp12_slippage_ratio_velocity_3m"] = slip_v3
        frame.loc[idx, "exp12_slippage_ratio_velocity_6m"] = slip_v6
        frame.loc[idx, "exp12_slippage_ratio_velocity_12m"] = slip_v12
        frame.loc[idx, "exp12_slippage_ratio_acceleration"] = slip_v3 - slip_v6
        frame.loc[idx, "exp12_schedule_revision_magnitude_12m_pct"] = _rolling_sum(dates, slip_change, WINDOW_DAYS[12])
        frame.loc[idx, "exp12_slippage_worsening_streak"] = _positive_streak(slip_pct)

        # Replace v1's indirect "slippage changed" revision semantics with an
        # actual effective completion-date revision event when date fields exist.
        if schedule_events.any():
            frame.loc[idx, "exp12_schedule_revisions_12m"] = _event_count(dates, schedule_events, WINDOW_DAYS[12])
            frame.loc[idx, "exp12_months_since_schedule_revision"] = _months_since_event(dates, schedule_events)

    return frame


def enrich_rows(supervised: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    if history is None:
        if not TRAJECTORIES.exists():
            raise FileNotFoundError("Experiment 12 requires paimana_project_trajectories.csv.")
        history = pd.read_csv(TRAJECTORIES, dtype={"canonical_project_id": "string"}, low_memory=False)
    source = engineer_history(history)
    feature_names = list(dict.fromkeys(V1_COST + V1_SPEND + V1_SCHEDULE + V2_FEATURES))
    lookup = source[["canonical_project_id", "snapshot_date", *feature_names]].drop_duplicates(
        ["canonical_project_id", "snapshot_date"], keep="last"
    )
    rows = supervised.copy()
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce")
    result = rows.merge(lookup, on=["canonical_project_id", "snapshot_date"], how="left", validate="many_to_one")
    if len(result) != len(rows):
        raise AssertionError("Experiment 12 changed the supervised cohort.")
    return result


def _usable_features(train: pd.DataFrame) -> tuple[list[str], dict]:
    candidates = list(dict.fromkeys(V1_COST + V1_SPEND + V1_SCHEDULE + V2_FEATURES))
    selected, audit = [], {}
    for name in candidates:
        values = pd.to_numeric(train[name], errors="coerce")
        pct = float(values.notna().mean() * 100.0)
        usable = pct >= 10.0 and values.dropna().nunique() > 1
        audit[name] = {"availability_percentage": round(pct, 3), "selected": bool(usable)}
        if usable:
            selected.append(name)
    return selected, audit


def _candidate_groups(target: str, usable: list[str]) -> dict[str, list[str]]:
    allowed = set(usable)
    def keep(names: list[str]) -> list[str]:
        return [name for name in names if name in allowed]
    if target == "actual_cost_overrun_percentage":
        return {
            "production_only": [],
            "cost_only": keep(COST_SIGNAL),
            "cost_plus_spend": keep(COST_SIGNAL + SPEND_SIGNAL),
            "all_trajectory": list(usable),
        }
    return {
        "production_only": [],
        "schedule_only": keep(SCHEDULE_SIGNAL),
        "schedule_plus_spend": keep(SCHEDULE_SIGNAL + SPEND_SIGNAL),
        "all_trajectory": list(usable),
    }


def _select_target_features(
    train: pd.DataFrame,
    production_features: list[str],
    usable: list[str],
    target: str,
    algorithm: str,
    seed: int,
) -> tuple[list[str], str, list[dict]]:
    years = sorted(pd.to_numeric(train.completion_year, errors="coerce").dropna().astype(int).unique())
    validation_years = years[-2:] if len(years) >= 3 else years[-1:]
    fitting = train[~train.completion_year.isin(validation_years)].copy()
    validation = train[train.completion_year.isin(validation_years)].copy()
    if fitting.canonical_project_id.nunique() < 5 or validation.canonical_project_id.nunique() < 2:
        fitting = train.copy(); validation = train.copy(); validation_years = years

    comparisons = []
    groups = _candidate_groups(target, usable)
    for group_name, added in groups.items():
        features = list(dict.fromkeys(production_features + added))
        model = _fit_pipeline(_regressors(seed)[algorithm], fitting, features, target)
        pred = model.predict(validation[features])
        metrics = _regression_metrics(validation[target], pred, validation.sample_weight, validation.canonical_project_id)
        comparisons.append({
            "feature_group": group_name,
            "added_features": added,
            "feature_count": len(features),
            "validation_years": validation_years,
            **metrics,
        })
    winner = min(comparisons, key=lambda item: item["MAE"])
    return list(winner["added_features"]), str(winner["feature_group"]), comparisons


def fit_experiment(*, data, training_start, training_end, test_end, production_bundle, production_receipt, history=None):
    frozen = data.copy()
    frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce")
    frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    base_train, base_test = temporal_project_split(frozen, training_start, training_end, test_end)
    enriched = enrich_rows(frozen, history)
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    production_features = list((production_bundle.get("metadata") or {}).get("features_used") or production_receipt.get("features_used") or [])
    usable, audit = _usable_features(train)
    if not production_features or not usable:
        raise ValueError("Experiment 12 requires production features plus usable trajectory features.")

    cost_name = _algorithm(production_bundle, production_receipt, "cost")
    delay_name = _algorithm(production_bundle, production_receipt, "delay")
    cost_added, cost_group, cost_internal = _select_target_features(
        train, production_features, usable, "actual_cost_overrun_percentage", cost_name, 26203
    )
    delay_added, delay_group, delay_internal = _select_target_features(
        train, production_features, usable, "actual_delay_days", delay_name, 26204
    )
    cost_features = list(dict.fromkeys(production_features + cost_added))
    delay_features = list(dict.fromkeys(production_features + delay_added))
    union_added = list(dict.fromkeys(cost_added + delay_added))
    union_features = list(dict.fromkeys(production_features + union_added))

    cost = _fit_pipeline(_regressors(26203)[cost_name], train, cost_features, "actual_cost_overrun_percentage")
    delay = _fit_pipeline(_regressors(26204)[delay_name], train, delay_features, "actual_delay_days")

    compare = test[pd.to_numeric(test.exp12_history_12m, errors="coerce").fillna(0).ge(MIN_HISTORY)].copy()
    if compare.canonical_project_id.nunique() < 2:
        raise ValueError("Experiment 12 has too few future projects with usable history.")
    compare = assign_project_balanced_weights(compare)
    compare["production_cost"] = production_bundle["cost"].predict(compare[production_features])
    compare["production_delay"] = np.maximum(0, production_bundle["delay"].predict(compare[production_features]))
    compare["experiment_cost"] = cost.predict(compare[cost_features])
    compare["experiment_delay"] = np.maximum(0, delay.predict(compare[delay_features]))

    pc, ec = _metric(compare, "actual_cost_overrun_percentage", "production_cost"), _metric(compare, "actual_cost_overrun_percentage", "experiment_cost")
    pdm, edm = _metric(compare, "actual_delay_days", "production_delay"), _metric(compare, "actual_delay_days", "experiment_delay")
    paired_cost = paired_project_mae_comparison(compare, actual="actual_cost_overrun_percentage", baseline_prediction="production_cost", candidate_prediction="experiment_cost")
    paired_delay = paired_project_mae_comparison(compare, actual="actual_delay_days", baseline_prediction="production_delay", candidate_prediction="experiment_delay", seed=26104)
    ps, es = _stage(compare, "production"), _stage(compare, "experiment")
    cost_gain = (pc["MAE"] - ec["MAE"]) / pc["MAE"] * 100 if pc["MAE"] else None
    delay_gain = (pdm["MAE"] - edm["MAE"]) / pdm["MAE"] * 100 if pdm["MAE"] else None
    overall = {
        "production_cost_mae": pc["MAE"], "experiment_cost_mae": ec["MAE"],
        "absolute_mae_improvement_pp": round(pc["MAE"] - ec["MAE"], 4),
        "improvement_percentage": round(cost_gain, 4) if cost_gain is not None else None,
        "production_delay_mae": pdm["MAE"], "experiment_delay_mae": edm["MAE"],
        "absolute_delay_mae_improvement_days": round(pdm["MAE"] - edm["MAE"], 4),
        "delay_improvement_percentage": round(delay_gain, 4) if delay_gain is not None else None,
        "comparison_test_projects": int(compare.canonical_project_id.nunique()),
        "comparison_test_snapshots": int(len(compare)),
        "paired_project_comparison": paired_cost,
        "paired_project_cost_comparison": paired_cost,
        "paired_project_delay_comparison": paired_delay,
        "production_stage_metrics": ps,
        "experiment_stage_metrics": es,
        "stage_balanced": {
            "production_cost_mae": _macro(ps, "cost"), "experiment_cost_mae": _macro(es, "cost"),
            "production_delay_mae": _macro(ps, "delay"), "experiment_delay_mae": _macro(es, "delay"),
        },
        "internal_feature_selection": {
            "cost": {"selected_group": cost_group, "comparisons": cost_internal},
            "delay": {"selected_group": delay_group, "comparisons": delay_internal},
        },
    }

    context = build_experiment_context(
        experiment_id=EXPERIMENT_ID, full_data=frozen, train=base_train, test=base_test,
        features=union_features, training_start=training_start, training_end=training_end,
        testing_end=test_end, weighting_policy="project-balanced quarterly snapshots"
    )
    manifest = new_experiment_manifest(
        context=context, name=EXPERIMENT_NAME,
        changed_dimension="feature_set",
        hypothesis="Scale-aware past-only trajectories with target-specific internal feature-group selection reduce future cost and delay MAE."
    )
    manifest.update({
        "scope": EXPERIMENT_SCOPE,
        "implementation_revision": "v2",
        "production_run_id": production_receipt.get("run_id"),
        "production_features": production_features,
        "usable_trajectory_features": usable,
        "cost_added_features": cost_added,
        "delay_added_features": delay_added,
        "feature_availability": audit,
        "selected_algorithms": {"cost": cost_name, "delay": delay_name},
        "selected_feature_groups": {"cost": cost_group, "delay": delay_group},
        "internal_feature_selection": {"cost": cost_internal, "delay": delay_internal},
        "comparison_filter": f">={MIN_HISTORY} official observations in trailing 12 months",
        "leakage_policy": "Feature construction uses current/earlier project snapshots only; feature-group selection uses only the last two completion years inside the training window; the future holdout is never consulted for selection.",
    })

    run_dir = experiment_run_directory(EXPERIMENT_ID, context.window, manifest["run_id"])
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(cost, run_dir / "cost_model.pkl")
    joblib.dump(delay, run_dir / "delay_model.pkl")
    (run_dir / "manifest.json").write_text(json.dumps(_safe(manifest), indent=2, allow_nan=False) + "\n")
    (run_dir / "evaluation_results.json").write_text(json.dumps(_safe(overall), indent=2, allow_nan=False) + "\n")
    record_experiment({
        "experiment_id": EXPERIMENT_ID, "name": EXPERIMENT_NAME, "run_id": manifest["run_id"],
        "status": "COMPLETED", "decision": "PENDING", "model_role": "experiment",
        "promotion_allowed": False, "scope": EXPERIMENT_SCOPE, "window": context.window,
        "created_at": manifest["created_at"], "production_run_id": production_receipt.get("run_id"),
        "cost_improvement_percentage": overall["improvement_percentage"],
        "delay_improvement_percentage": overall["delay_improvement_percentage"],
        "implementation_revision": "v2",
    })

    lookup_rows = enriched[["canonical_project_id", "snapshot_date", *union_added]].drop_duplicates(
        ["canonical_project_id", "snapshot_date"], keep="last"
    )
    lookup = {
        (str(r.canonical_project_id), pd.Timestamp(r.snapshot_date).isoformat()): {f: r.get(f) for f in union_added}
        for _, r in lookup_rows.iterrows() if pd.notna(r.canonical_project_id) and pd.notna(r.snapshot_date)
    }
    comparable = {(str(r.canonical_project_id), pd.Timestamp(r.snapshot_date).isoformat()) for _, r in compare.iterrows()}
    experiment = {
        "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME, "run_id": manifest["run_id"],
        "model_role": "experiment", "scope": EXPERIMENT_SCOPE, "decision": "PENDING",
        "promotion_allowed": False, "implementation_revision": "v2",
        "feature_count": len(union_features), "production_feature_count": len(production_features),
        "added_feature_count": len(union_added), "added_features": union_added,
        "cost_added_features": cost_added, "delay_added_features": delay_added,
        "selected_feature_groups": {"cost": cost_group, "delay": delay_group},
        "selected_algorithms": {"cost": cost_name, "delay": delay_name},
        "metrics": {"cost": ec, "delay": edm}, "leakage_policy": manifest["leakage_policy"],
    }
    return {
        "experiment": experiment,
        "overall_comparison": overall,
        "runtime_state": {
            "cost_model": cost, "delay_model": delay,
            "production_features": production_features,
            "cost_features": cost_features, "delay_features": delay_features,
            "added": union_added, "lookup": lookup, "comparable": comparable,
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[frame.apply(lambda row: _key(row) in state["comparable"], axis=1)].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 12 trajectory is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    cost_X = candidate.to_frame().T.reindex(columns=state["cost_features"])
    delay_X = candidate.to_frame().T.reindex(columns=state["delay_features"])
    return {
        "predicted_cost_overrun": round(float(state["cost_model"].predict(cost_X)[0]), 4),
        "predicted_delay_days": round(max(0.0, float(state["delay_model"].predict(delay_X)[0])), 4),
        "trajectory_features_available": int(sum(pd.notna(candidate.get(f)) for f in state["added"])),
        "trajectory_feature_count": len(state["added"]),
        "implementation_revision": "v2",
    }
