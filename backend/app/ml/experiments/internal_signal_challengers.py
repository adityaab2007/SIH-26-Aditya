"""Shared causal internal-data challenger harness for Exp125-134.

All engineered features use only information available at or before each monthly
snapshot. No external datasets are required. Each experiment remains a bounded
residual challenger on top of freshly retrained current production.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import tempfile

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.experiments.prediction_ledger import build_prediction_ledger, write_prediction_ledger
from backend.app.ml.experiments.scientific_challenger_utils import (
    WINDOWS,
    attach_features,
    assert_same_keys,
    fit_bounded_residual_correction,
    fresh_production_window,
    lifecycle_metrics,
    paired_project_bootstrap,
    print_base_contract,
    production_hashes,
    rolling_production_oof,
    save_json,
    verdict_from_windows,
    weighted_mae,
)

ROOT = Path(__file__).resolve().parents[4]
KEYS = ["canonical_project_id", "snapshot_date"]

CORE_NUMERIC = [
    "approved_cost_cr",
    "revised_cost_cr",
    "cumulative_expenditure_cr",
    "expenditure_ratio",
    "schedule_slippage_days",
    "schedule_slippage_ratio",
    "elapsed_duration_days",
    "planned_duration_days",
    "duration_ratio",
    "expected_progress_percentage",
    "physical_progress",
    "cost_escalation_percentage",
    "cost_growth_velocity_3m",
    "cost_growth_velocity_6m",
    "cost_acceleration",
    "sector_average_delay",
    "sector_average_cost_overrun",
    "agency_average_delay",
    "agency_average_cost_overrun",
    "sector_delay_rate",
    "sector_cost_overrun_rate",
    "agency_delay_rate",
    "agency_cost_overrun_rate",
]

STRATEGIES = {
    "exp125": {
        "name": "Multi-Scale Trajectory Derivatives",
        "scope": "cost+delay",
        "features": [
            "is_d_cost_1", "is_d_cost_3", "is_d_cost_6", "is_cost_accel", "is_cost_vol3", "is_cost_vol6",
            "is_d_slip_1", "is_d_slip_3", "is_d_slip_6", "is_slip_accel", "is_slip_vol3", "is_slip_vol6",
            "is_d_exp_1", "is_d_exp_3", "is_d_exp_6", "is_exp_accel", "is_exp_vol3", "is_exp_vol6",
        ],
    },
    "exp126": {
        "name": "Reporting Cadence and Data-Quality Behavior",
        "scope": "cost+delay",
        "features": [
            "is_report_gap_days", "is_report_gap_mean3", "is_report_gap_std6", "is_snapshot_index",
            "is_missing_count", "is_missing_delta", "is_unchanged_core_count", "is_stale_report_flag",
            "is_gap_x_duration", "is_missing_x_duration",
        ],
    },
    "exp127": {
        "name": "Cost Revision Shock Persistence",
        "scope": "cost",
        "features": [
            "is_revision_ratio", "is_revision_jump", "is_revision_jump_abs", "is_revision_up_count",
            "is_cost_shock_count", "is_cost_shock_run", "is_months_since_cost_shock", "is_revision_velocity",
            "is_revision_x_size", "is_revision_x_duration",
        ],
    },
    "exp128": {
        "name": "Slippage Momentum and Recovery Dynamics",
        "scope": "delay",
        "features": [
            "is_slip_velocity", "is_slip_acceleration", "is_slip_pos_run", "is_slip_recovery_run",
            "is_slip_worsening_count", "is_slip_recovery_count", "is_months_since_slip_worsen",
            "is_slip_per_elapsed", "is_slip_x_duration", "is_recovery_fraction",
        ],
    },
    "exp129": {
        "name": "Expenditure-Time Efficiency Geometry",
        "scope": "cost+delay",
        "features": [
            "is_exec_gap", "is_exec_gap_abs", "is_burn_velocity", "is_burn_accel", "is_spend_efficiency",
            "is_expected_minus_spend", "is_exec_gap_trend", "is_exec_gap_vol", "is_burn_x_slip", "is_burn_x_cost",
        ],
    },
    "exp130": {
        "name": "Cross-Target Production Coupling",
        "scope": "cost+delay",
        "features": [
            "is_prod_cost", "is_prod_delay", "is_prod_cost_abs", "is_prod_delay_log",
            "is_prod_cost_x_delay", "is_prod_delay_per_duration", "is_prod_cost_x_revision",
            "is_prod_delay_x_slip", "is_joint_risk_norm", "is_prediction_divergence",
        ],
    },
    "exp131": {
        "name": "Cumulative Distress Burden",
        "scope": "cost+delay",
        "features": [
            "is_cost_burden_mean", "is_cost_burden_max", "is_slip_burden_mean", "is_slip_burden_max",
            "is_exec_gap_burden_mean", "is_worsening_fraction", "is_distress_area", "is_distress_peak",
            "is_distress_x_size", "is_distress_x_duration",
        ],
    },
    "exp132": {
        "name": "Causal Change-Point State",
        "scope": "cost+delay",
        "features": [
            "is_cost_change_z", "is_slip_change_z", "is_exp_change_z", "is_cost_shift_flag", "is_slip_shift_flag",
            "is_exp_shift_flag", "is_any_shift_count", "is_shift_run", "is_months_since_any_shift", "is_shift_x_duration",
        ],
    },
    "exp133": {
        "name": "Hierarchical Deviation Interactions",
        "scope": "cost+delay",
        "features": [
            "is_agency_sector_delay_gap", "is_agency_sector_cost_gap", "is_agency_sector_delay_rate_gap",
            "is_agency_sector_cost_rate_gap", "is_slip_vs_sector_delay", "is_cost_vs_sector_overrun",
            "is_slip_vs_agency_delay", "is_cost_vs_agency_overrun", "is_peer_gap_x_size", "is_peer_gap_x_duration",
        ],
    },
    "exp134": {
        "name": "Nonlinear Saturation and Tail Geometry",
        "scope": "cost+delay",
        "features": [
            "is_log_approved_cost", "is_log_planned_duration", "is_sqrt_slippage", "is_signed_log_cost_esc",
            "is_duration_tail", "is_slip_tail", "is_cost_tail", "is_size_x_slip", "is_size_x_cost",
            "is_duration_x_cost", "is_duration_x_slip", "is_tail_interaction",
        ],
    },
}


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    out = aa / bb.where(bb.abs() > eps)
    return out.replace([np.inf, -np.inf], np.nan)


def _run_length(mask: pd.Series) -> pd.Series:
    vals = mask.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(vals), dtype=float)
    run = 0
    for i, flag in enumerate(vals):
        run = run + 1 if flag else 0
        out[i] = run
    return pd.Series(out, index=mask.index)


def _months_since(mask: pd.Series, dates: pd.Series) -> pd.Series:
    out = []
    last = pd.NaT
    for flag, dt in zip(mask.fillna(False).astype(bool), pd.to_datetime(dates, errors="coerce")):
        if flag and pd.notna(dt):
            last = dt
        if pd.isna(dt) or pd.isna(last):
            out.append(99.0)
        else:
            out.append(max(0.0, float((dt - last).days) / 30.4375))
    return pd.Series(out, index=mask.index, dtype=float)


def _rolling_z(current: pd.Series, history: pd.Series) -> pd.Series:
    mean = history.shift(1).rolling(6, min_periods=3).mean()
    std = history.shift(1).rolling(6, min_periods=3).std().replace(0, np.nan)
    return ((current - mean) / std).replace([np.inf, -np.inf], np.nan)


def build_internal_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build all Exp125-134 causal features using current/past project history only."""
    data = frame.copy()
    data["canonical_project_id"] = data["canonical_project_id"].astype("string").str.strip()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    for col in CORE_NUMERIC:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values(KEYS, kind="mergesort").copy()
    pieces = []

    for _, g0 in data.groupby("canonical_project_id", sort=False):
        g = g0.sort_values("snapshot_date", kind="mergesort").copy()
        out = g[KEYS].copy()
        dates = g["snapshot_date"]
        approved = _num(g, "approved_cost_cr")
        revised = _num(g, "revised_cost_cr")
        cost = _num(g, "cost_escalation_percentage")
        slip = _num(g, "schedule_slippage_days")
        slip_ratio = _num(g, "schedule_slippage_ratio")
        exp = _num(g, "expenditure_ratio")
        cumexp = _num(g, "cumulative_expenditure_cr")
        duration = _num(g, "duration_ratio")
        elapsed = _num(g, "elapsed_duration_days")
        planned = _num(g, "planned_duration_days")
        expected = _num(g, "expected_progress_percentage") / 100.0

        # Exp125: multi-scale derivatives.
        for label, s in (("cost", cost), ("slip", slip), ("exp", exp)):
            d1 = s.diff(1)
            out[f"is_d_{label}_1"] = d1
            out[f"is_d_{label}_3"] = s - s.shift(3)
            out[f"is_d_{label}_6"] = s - s.shift(6)
            out[f"is_{label}_accel"] = d1.diff()
            out[f"is_{label}_vol3"] = d1.rolling(3, min_periods=2).std()
            out[f"is_{label}_vol6"] = d1.rolling(6, min_periods=3).std()

        # Exp126: cadence, staleness, missingness and reporting consistency.
        gap = dates.diff().dt.days.astype(float)
        out["is_report_gap_days"] = gap
        out["is_report_gap_mean3"] = gap.rolling(3, min_periods=1).mean()
        out["is_report_gap_std6"] = gap.rolling(6, min_periods=2).std()
        out["is_snapshot_index"] = np.arange(len(g), dtype=float)
        quality_cols = ["revised_cost_cr", "cumulative_expenditure_cr", "schedule_slippage_days", "expenditure_ratio", "expected_progress_percentage"]
        missing = g[quality_cols].isna().sum(axis=1).astype(float)
        out["is_missing_count"] = missing
        out["is_missing_delta"] = missing.diff()
        unchanged = pd.DataFrame({
            "cost": cost.eq(cost.shift()), "slip": slip.eq(slip.shift()), "exp": exp.eq(exp.shift()), "revised": revised.eq(revised.shift())
        }).sum(axis=1).astype(float)
        out["is_unchanged_core_count"] = unchanged
        out["is_stale_report_flag"] = gap.gt(45).astype(float)
        out["is_gap_x_duration"] = gap * duration
        out["is_missing_x_duration"] = missing * duration

        # Exp127: revision shock persistence.
        revision_ratio = _safe_div(revised, approved) - 1.0
        revision_jump = revision_ratio.diff()
        cost_jump = cost.diff()
        shock_threshold = max(float(cost_jump.abs().expanding(min_periods=3).median().shift(1).fillna(1.0).iloc[-1]) if len(g) else 1.0, 1.0)
        cost_shock = cost_jump.abs().ge(shock_threshold)
        out["is_revision_ratio"] = revision_ratio
        out["is_revision_jump"] = revision_jump
        out["is_revision_jump_abs"] = revision_jump.abs()
        out["is_revision_up_count"] = revision_jump.gt(0).cumsum().astype(float)
        out["is_cost_shock_count"] = cost_shock.cumsum().astype(float)
        out["is_cost_shock_run"] = _run_length(cost_jump.gt(0))
        out["is_months_since_cost_shock"] = _months_since(cost_shock, dates)
        out["is_revision_velocity"] = revision_jump.rolling(3, min_periods=1).mean()
        out["is_revision_x_size"] = revision_ratio * np.log1p(approved.clip(lower=0))
        out["is_revision_x_duration"] = revision_ratio * duration

        # Exp128: slippage momentum and recovery.
        slip_d = slip.diff()
        worsening = slip_d.gt(0)
        recovering = slip_d.lt(0)
        out["is_slip_velocity"] = slip_d.rolling(3, min_periods=1).mean()
        out["is_slip_acceleration"] = slip_d.diff()
        out["is_slip_pos_run"] = _run_length(worsening)
        out["is_slip_recovery_run"] = _run_length(recovering)
        out["is_slip_worsening_count"] = worsening.cumsum().astype(float)
        out["is_slip_recovery_count"] = recovering.cumsum().astype(float)
        out["is_months_since_slip_worsen"] = _months_since(worsening, dates)
        out["is_slip_per_elapsed"] = _safe_div(slip, elapsed.clip(lower=1))
        out["is_slip_x_duration"] = slip * duration
        out["is_recovery_fraction"] = _safe_div(recovering.cumsum().astype(float), pd.Series(np.arange(1, len(g)+1), index=g.index, dtype=float))

        # Exp129: expenditure-time geometry.
        exec_gap = exp - duration
        burn = exp.diff()
        out["is_exec_gap"] = exec_gap
        out["is_exec_gap_abs"] = exec_gap.abs()
        out["is_burn_velocity"] = burn.rolling(3, min_periods=1).mean()
        out["is_burn_accel"] = burn.diff()
        out["is_spend_efficiency"] = _safe_div(exp, duration.clip(lower=0.01))
        out["is_expected_minus_spend"] = expected - exp
        out["is_exec_gap_trend"] = exec_gap.diff().rolling(3, min_periods=1).mean()
        out["is_exec_gap_vol"] = exec_gap.rolling(6, min_periods=2).std()
        out["is_burn_x_slip"] = burn * slip
        out["is_burn_x_cost"] = burn * cost

        # Exp131: accumulated distress burden.
        cost_pos = cost.clip(lower=0)
        slip_pos = slip_ratio.clip(lower=0).fillna(_safe_div(slip.clip(lower=0), planned.clip(lower=1)))
        gap_bad = (-exec_gap).clip(lower=0)
        distress = cost_pos.fillna(0) / 100.0 + slip_pos.fillna(0) + gap_bad.fillna(0)
        worsening_any = (cost.diff().gt(0) | slip.diff().gt(0) | exec_gap.diff().lt(0)).astype(float)
        out["is_cost_burden_mean"] = cost_pos.expanding(min_periods=1).mean()
        out["is_cost_burden_max"] = cost_pos.expanding(min_periods=1).max()
        out["is_slip_burden_mean"] = slip_pos.expanding(min_periods=1).mean()
        out["is_slip_burden_max"] = slip_pos.expanding(min_periods=1).max()
        out["is_exec_gap_burden_mean"] = gap_bad.expanding(min_periods=1).mean()
        out["is_worsening_fraction"] = worsening_any.expanding(min_periods=1).mean()
        out["is_distress_area"] = distress.cumsum()
        out["is_distress_peak"] = distress.expanding(min_periods=1).max()
        out["is_distress_x_size"] = out["is_distress_area"] * np.log1p(approved.clip(lower=0))
        out["is_distress_x_duration"] = out["is_distress_area"] * duration

        # Exp132: causal change-point indicators based on prior local distribution.
        cost_z = _rolling_z(cost.diff(), cost.diff())
        slip_z = _rolling_z(slip.diff(), slip.diff())
        exp_z = _rolling_z(exp.diff(), exp.diff())
        shift_cost = cost_z.abs().ge(2.0)
        shift_slip = slip_z.abs().ge(2.0)
        shift_exp = exp_z.abs().ge(2.0)
        any_shift = shift_cost | shift_slip | shift_exp
        out["is_cost_change_z"] = cost_z
        out["is_slip_change_z"] = slip_z
        out["is_exp_change_z"] = exp_z
        out["is_cost_shift_flag"] = shift_cost.astype(float)
        out["is_slip_shift_flag"] = shift_slip.astype(float)
        out["is_exp_shift_flag"] = shift_exp.astype(float)
        out["is_any_shift_count"] = any_shift.cumsum().astype(float)
        out["is_shift_run"] = _run_length(any_shift)
        out["is_months_since_any_shift"] = _months_since(any_shift, dates)
        out["is_shift_x_duration"] = out["is_any_shift_count"] * duration

        # Exp133: interactions among already-causal hierarchical priors and current state.
        sad = _num(g, "sector_average_delay")
        scost = _num(g, "sector_average_cost_overrun")
        aad = _num(g, "agency_average_delay")
        acost = _num(g, "agency_average_cost_overrun")
        sdr = _num(g, "sector_delay_rate")
        scr = _num(g, "sector_cost_overrun_rate")
        adr = _num(g, "agency_delay_rate")
        acr = _num(g, "agency_cost_overrun_rate")
        out["is_agency_sector_delay_gap"] = aad - sad
        out["is_agency_sector_cost_gap"] = acost - scost
        out["is_agency_sector_delay_rate_gap"] = adr - sdr
        out["is_agency_sector_cost_rate_gap"] = acr - scr
        out["is_slip_vs_sector_delay"] = slip - sad
        out["is_cost_vs_sector_overrun"] = cost - scost
        out["is_slip_vs_agency_delay"] = slip - aad
        out["is_cost_vs_agency_overrun"] = cost - acost
        peer_gap = (aad - sad).abs().fillna(0) + (acost - scost).abs().fillna(0)
        out["is_peer_gap_x_size"] = peer_gap * np.log1p(approved.clip(lower=0))
        out["is_peer_gap_x_duration"] = peer_gap * duration

        # Exp134: deliberately bounded transforms to help tail behavior.
        out["is_log_approved_cost"] = np.log1p(approved.clip(lower=0))
        out["is_log_planned_duration"] = np.log1p(planned.clip(lower=0))
        out["is_sqrt_slippage"] = np.sqrt(slip.clip(lower=0))
        out["is_signed_log_cost_esc"] = np.sign(cost) * np.log1p(cost.abs())
        out["is_duration_tail"] = np.maximum(duration - 1.0, 0.0)
        out["is_slip_tail"] = np.log1p(slip.clip(lower=0))
        out["is_cost_tail"] = np.log1p(cost.clip(lower=0))
        out["is_size_x_slip"] = out["is_log_approved_cost"] * out["is_slip_tail"]
        out["is_size_x_cost"] = out["is_log_approved_cost"] * out["is_cost_tail"]
        out["is_duration_x_cost"] = duration * out["is_cost_tail"]
        out["is_duration_x_slip"] = duration * out["is_slip_tail"]
        out["is_tail_interaction"] = out["is_duration_tail"] * (out["is_slip_tail"] + out["is_cost_tail"])

        pieces.append(out)

    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=KEYS)
    if result.duplicated(KEYS).any():
        raise AssertionError("internal feature generation duplicated project/snapshot keys")
    return result


def _add_cross_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cost = pd.to_numeric(out["predicted_cost_overrun"], errors="coerce")
    delay = pd.to_numeric(out["predicted_delay_days"], errors="coerce")
    duration_days = pd.to_numeric(out.get("planned_duration_days"), errors="coerce")
    revision = pd.to_numeric(out.get("cost_escalation_percentage"), errors="coerce")
    slip = pd.to_numeric(out.get("schedule_slippage_days"), errors="coerce")
    out["is_prod_cost"] = cost
    out["is_prod_delay"] = delay
    out["is_prod_cost_abs"] = cost.abs()
    out["is_prod_delay_log"] = np.log1p(delay.clip(lower=0))
    out["is_prod_cost_x_delay"] = cost * np.log1p(delay.clip(lower=0))
    out["is_prod_delay_per_duration"] = _safe_div(delay, duration_days.clip(lower=1))
    out["is_prod_cost_x_revision"] = cost * revision
    out["is_prod_delay_x_slip"] = np.log1p(delay.clip(lower=0)) * slip
    out["is_joint_risk_norm"] = np.sqrt(cost.pow(2) + np.log1p(delay.clip(lower=0)).pow(2))
    out["is_prediction_divergence"] = cost.abs() - np.log1p(delay.clip(lower=0))
    return out


def _target_result(score: pd.DataFrame, baseline, candidate, actual: str, seed: int) -> dict:
    b = weighted_mae(score, actual, baseline)
    c = weighted_mae(score, actual, candidate)
    return {
        "base_mae": b,
        "experiment_mae": c,
        "improvement_pct": ((b - c) / b * 100.0) if b else 0.0,
        "bootstrap": paired_project_bootstrap(score, actual=actual, baseline=baseline, challenger=candidate, samples=5000, seed=seed),
        "lifecycle": lifecycle_metrics(score, actual=actual, baseline=baseline, challenger=candidate),
    }


def run_experiment(exp_id: str, output_dir: Path) -> dict:
    if exp_id not in STRATEGIES:
        raise KeyError(exp_id)
    config = STRATEGIES[exp_id]
    print_base_contract()
    before = production_hashes()
    data, identity = build_training_dataset()
    data = data.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    feature_table = build_internal_feature_table(data)
    windows = []
    exp_num = int(exp_id.replace("exp", ""))

    for start, end, test_end in WINDOWS:
        with tempfile.TemporaryDirectory(prefix=f"{exp_id}-{end}-") as td:
            tmp = Path(td)
            prod = fresh_production_window(data, identity, training_start=start, training_end=end, test_end=test_end, artifact_root=tmp / "baseline")
            oof = rolling_production_oof(data, identity, training_start=start, training_end=end, root=tmp / "oof")
            score = prod.comparable.copy()
            features = list(config["features"])

            if exp_id == "exp130":
                oof = _add_cross_target_features(oof)
                score = _add_cross_target_features(score)
            else:
                # Attach only experiment features; source production columns remain untouched.
                oof = attach_features(oof, feature_table, features)
                score = attach_features(score, feature_table, features)

            assert_same_keys(prod.comparable, score)
            oof["is_internal_available"] = oof[features].notna().any(axis=1)
            score["is_internal_available"] = score[features].notna().any(axis=1)

            cost_base = pd.to_numeric(score["predicted_cost_overrun"], errors="coerce").to_numpy(float)
            delay_base = pd.to_numeric(score["predicted_delay_days"], errors="coerce").to_numpy(float)
            cost_candidate = cost_base.copy()
            delay_candidate = delay_base.copy()
            cost_diag = {"status": "UNCHANGED BY SCOPE", "selected_scale": 0.0}
            delay_diag = {"status": "UNCHANGED BY SCOPE", "selected_scale": 0.0}
            cost_model = None
            delay_model = None

            if config["scope"] in {"cost", "cost+delay"}:
                cost_candidate, cost_diag, cost_model = fit_bounded_residual_correction(
                    oof, score, features=features, actual="actual_cost_overrun_percentage",
                    production_col="predicted_cost_overrun", available_col="is_internal_available",
                    seed=exp_num * 100 + end,
                )
            if config["scope"] in {"delay", "cost+delay"}:
                delay_candidate, delay_diag, delay_model = fit_bounded_residual_correction(
                    oof, score, features=features, actual="actual_delay_days",
                    production_col="predicted_delay_days", available_col="is_internal_available",
                    seed=exp_num * 100 + end + 1,
                )

            cost_result = _target_result(score, cost_base, cost_candidate, "actual_cost_overrun_percentage", exp_num * 1000 + end)
            delay_result = _target_result(score, delay_base, delay_candidate, "actual_delay_days", exp_num * 1000 + end + 1)
            window_name = f"{start}_{end}"
            ledger_dir = output_dir / window_name
            ledger = build_prediction_ledger(
                score, experiment_id=exp_id, window=window_name,
                production_cost_prediction=cost_base, experiment_cost_prediction=cost_candidate,
                production_delay_prediction=delay_base, experiment_delay_prediction=delay_candidate,
                extra_columns=["lifecycle_stage"],
            )
            write_prediction_ledger(ledger, ledger_dir, overwrite=True)
            if cost_model is not None:
                joblib.dump(cost_model, ledger_dir / "cost_residual_model.pkl")
            if delay_model is not None:
                joblib.dump(delay_model, ledger_dir / "delay_residual_model.pkl")

            windows.append({
                "window": window_name,
                "status": "EXECUTION VALID",
                "projects": int(score["canonical_project_id"].nunique()),
                "snapshots": int(len(score)),
                "base_feature_count": 25,
                "internal_feature_count": len(features),
                "internal_features": features,
                "scope": config["scope"],
                "cost": cost_result | {"residual_correction": cost_diag},
                "delay": delay_result | {"residual_correction": delay_diag},
                "production_cost_baseline": prod.result["metadata"].get("production_cost_baseline"),
                "production_delay_baseline": prod.result["metadata"].get("production_delay_baseline"),
            })

    after = production_hashes()
    if before != after:
        raise AssertionError(f"{exp_id} modified tracked production artifacts")

    cost_windows = [{"status": w["status"], "improvement_pct": w["cost"]["improvement_pct"], "bootstrap": w["cost"]["bootstrap"]} for w in windows]
    delay_windows = [{"status": w["status"], "improvement_pct": w["delay"]["improvement_pct"], "bootstrap": w["delay"]["bootstrap"]} for w in windows]
    cost_verdict = verdict_from_windows(cost_windows) if config["scope"] in {"cost", "cost+delay"} else "NOT IN SCOPE"
    delay_verdict = verdict_from_windows(delay_windows) if config["scope"] in {"delay", "cost+delay"} else "NOT IN SCOPE"
    payload = {
        "experiment": exp_id,
        "name": config["name"],
        "scope": config["scope"],
        "external_dataset_required": False,
        "causal_as_of_features_only": True,
        "base_pipeline": "25-FEATURE MONTHLY LIFECYCLE",
        "windows": windows,
        "cost_verdict": cost_verdict,
        "delay_verdict": delay_verdict,
        "production_artifacts_unchanged": True,
        "promotion_allowed": False,
    }
    save_json(ROOT / "reports" / f"{exp_id}_final_report.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return payload


def self_test(exp_id: str) -> None:
    if exp_id not in STRATEGIES:
        raise AssertionError("unknown experiment")
    sample = pd.DataFrame({
        "canonical_project_id": ["p1"] * 8,
        "snapshot_date": pd.date_range("2020-01-01", periods=8, freq="MS"),
        "approved_cost_cr": [100] * 8,
        "revised_cost_cr": [100, 100, 105, 105, 110, 110, 112, 115],
        "cumulative_expenditure_cr": [5, 10, 17, 25, 35, 48, 60, 72],
        "expenditure_ratio": [0.05, 0.10, 0.17, 0.25, 0.35, 0.48, 0.60, 0.72],
        "schedule_slippage_days": [0, 2, 5, 12, 20, 18, 15, 16],
        "schedule_slippage_ratio": [0, .01, .02, .04, .06, .05, .04, .04],
        "elapsed_duration_days": [30, 60, 90, 120, 150, 180, 210, 240],
        "planned_duration_days": [300] * 8,
        "duration_ratio": [.1, .2, .3, .4, .5, .6, .7, .8],
        "expected_progress_percentage": [10, 20, 30, 40, 50, 60, 70, 80],
        "cost_escalation_percentage": [0, 0, 5, 5, 10, 10, 12, 15],
        "sector_average_delay": [20] * 8,
        "sector_average_cost_overrun": [8] * 8,
        "agency_average_delay": [25] * 8,
        "agency_average_cost_overrun": [10] * 8,
        "sector_delay_rate": [.2] * 8,
        "sector_cost_overrun_rate": [.1] * 8,
        "agency_delay_rate": [.25] * 8,
        "agency_cost_overrun_rate": [.12] * 8,
    })
    features = build_internal_feature_table(sample)
    if features.duplicated(KEYS).any() or len(features) != len(sample):
        raise AssertionError("feature table key integrity failed")
    for col in STRATEGIES[exp_id]["features"]:
        if exp_id != "exp130" and col not in features:
            raise AssertionError(f"missing engineered feature: {col}")
    # Future-row append must not change already-computed earlier features.
    future = sample.iloc[[-1]].copy()
    future["snapshot_date"] = pd.Timestamp("2021-01-01")
    future["schedule_slippage_days"] = 9999
    future["cost_escalation_percentage"] = 999
    extended = build_internal_feature_table(pd.concat([sample, future], ignore_index=True)).iloc[: len(sample)].reset_index(drop=True)
    base = features.reset_index(drop=True)
    common = [c for c in base.columns if c in extended.columns]
    pd.testing.assert_frame_equal(base[common], extended[common], check_dtype=False)
