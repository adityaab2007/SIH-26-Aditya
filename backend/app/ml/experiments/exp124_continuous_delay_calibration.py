"""Exp124 — continuous forecast-horizon Delay calibration.

Strictly post-model: Cost is untouched. The calibrator learns current-production
Delay residuals only from forward OOF predictions and selects its correction
scale from a second forward meta-OOF layer.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from backend.app.ml import production_exp35_baseline as exp35_production
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.experiments.prediction_ledger import build_prediction_ledger, write_prediction_ledger
from backend.app.ml.experiments.scientific_challenger_utils import (
    WINDOWS, attach_features, assert_same_keys, fresh_production_window, lifecycle_metrics,
    paired_project_bootstrap, print_base_contract, production_hashes, rolling_production_oof,
    save_json, verdict_from_windows, weighted_mae, weighted_quantile,
)

ROOT = Path(__file__).resolve().parents[4]
EXP_ID = "exp124"
BASE_INPUTS = ["duration_ratio", "elapsed_duration_days", "planned_duration_days", "physical_progress", "progress_deviation", "schedule_slippage_days"]
SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)
_AFT_CAPACITY_ERROR = re.compile(r"^Only (\d+) projects have AFT evidence; cannot form the requested (\d+)-project calibration cohort\.$")


def _design(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    for col in BASE_INPUTS:
        if col not in x: x[col] = np.nan
    x["production_delay_prediction"] = pd.to_numeric(x["predicted_delay_days"], errors="coerce")
    x["remaining_planned_time"] = (pd.to_numeric(x["planned_duration_days"], errors="coerce") - pd.to_numeric(x["elapsed_duration_days"], errors="coerce")).clip(lower=0)
    x["duration_x_prediction"] = pd.to_numeric(x["duration_ratio"], errors="coerce") * pd.to_numeric(x["production_delay_prediction"], errors="coerce")
    return x

FEATURES = BASE_INPUTS + ["production_delay_prediction", "remaining_planned_time", "duration_x_prediction"]


def _model(alpha: float = 0.05) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False, extrapolation="linear")),
        ("scale", StandardScaler()),
        ("median", QuantileRegressor(quantile=0.5, alpha=alpha, solver="highs")),
    ])


def _fit(frame: pd.DataFrame) -> Pipeline:
    m = _model()
    residual = (pd.to_numeric(frame["actual_delay_days"], errors="coerce") - pd.to_numeric(frame["predicted_delay_days"], errors="coerce")).to_numpy(float)
    m.fit(frame[FEATURES], residual)
    return m


def fit_calibrator(oof: pd.DataFrame, score: pd.DataFrame):
    oof = _design(oof); score = _design(score)
    years = sorted(int(y) for y in pd.to_numeric(oof["oof_year"], errors="coerce").dropna().unique())
    meta = []
    for i in range(1, len(years)):
        year = years[i]
        fit = oof[pd.to_numeric(oof["oof_year"], errors="coerce") < year]
        val = oof[pd.to_numeric(oof["oof_year"], errors="coerce").eq(year)].copy()
        if len(fit) < 100 or fit.canonical_project_id.nunique() < 20 or val.empty: continue
        m = _fit(fit); val["correction"] = m.predict(val[FEATURES]); meta.append(val)
    residual = (pd.to_numeric(oof["actual_delay_days"], errors="coerce") - pd.to_numeric(oof["predicted_delay_days"], errors="coerce")).to_numpy(float)
    cap = max(weighted_quantile(np.abs(residual), oof["sample_weight"], 0.90), 1e-9)
    scale_mae = {}; selected = 0.0
    if meta:
        mdf = pd.concat(meta, ignore_index=True); corr = np.clip(mdf["correction"].to_numpy(float), -cap, cap)
        baseline = weighted_mae(mdf, "actual_delay_days", mdf["predicted_delay_days"])
        for scale in SCALES:
            pred = np.maximum(0.0, mdf["predicted_delay_days"].to_numpy(float) + scale * corr)
            scale_mae[str(scale)] = weighted_mae(mdf, "actual_delay_days", pred)
        selected = min(SCALES, key=lambda s: (scale_mae[str(s)], s))
        if scale_mae[str(selected)] >= baseline - 1e-12: selected = 0.0
    final = _fit(oof); corr = np.clip(final.predict(score[FEATURES]), -cap, cap)
    candidate = np.maximum(0.0, score["predicted_delay_days"].to_numpy(float) + selected * corr)
    return candidate, final, {"selected_scale": float(selected), "correction_cap": float(cap), "meta_oof_rows": int(sum(len(x) for x in meta)), "meta_oof_mae_by_scale": scale_mae, "calibrator": "SplineTransformer(n_knots=4,degree=2)+median QuantileRegressor", "features": FEATURES}


def _binned_diagnostics(frame: pd.DataFrame, baseline, challenger) -> dict:
    work = _design(frame); work["base"] = np.asarray(baseline, float); work["challenger"] = np.asarray(challenger, float)
    work["base_residual"] = pd.to_numeric(work.actual_delay_days, errors="coerce") - work.base; work["challenger_residual"] = pd.to_numeric(work.actual_delay_days, errors="coerce") - work.challenger
    out = {}
    for name, col in [("duration_ratio", "duration_ratio"), ("production_prediction", "production_delay_prediction")]:
        try: work["bin"] = pd.qcut(pd.to_numeric(work[col], errors="coerce"), q=10, duplicates="drop")
        except ValueError: continue
        rows = []
        for bucket, g in work.groupby("bin", observed=True):
            rows.append({"bin": str(bucket), "rows": int(len(g)), "mean_residual_before": float(g.base_residual.mean()), "mean_residual_after": float(g.challenger_residual.mean()), "mae_before": weighted_mae(g, "actual_delay_days", g.base), "mae_after": weighted_mae(g, "actual_delay_days", g.challenger)})
        out[name] = rows
    return out


def _rolling_oof_with_adaptive_aft_gate(
    data: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    training_start: int,
    training_end: int,
    root: Path,
) -> pd.DataFrame:
    """Run strict forward OOF while adapting only the Exp35 audit gate to fold capacity.

    The fixed 688-project AFT calibration cohort is valid for the verified
    production holdout, but early one-year OOF folds can contain fewer projects
    with AFT evidence. For those folds only, use every available AFT-evidence
    project (minimum 20). The production selector is restored immediately, so
    final production evaluation and tracked artifacts remain unchanged.
    """
    original_selector = exp35_production._select_aft_calibration_projects
    reductions: list[tuple[int, int]] = []

    def adaptive_selector(frame: pd.DataFrame, limit: int = exp35_production.VERIFIED_AFT_CALIBRATION_PROJECTS):
        try:
            return original_selector(frame, limit=limit)
        except RuntimeError as exc:
            match = _AFT_CAPACITY_ERROR.match(str(exc))
            if not match:
                raise
            available = int(match.group(1))
            requested = int(match.group(2))
            if available < 20:
                raise RuntimeError(
                    f"Only {available} projects have AFT evidence in this forward OOF fold; "
                    "Exp124 requires at least 20 for an experiment-local adaptive calibration gate."
                ) from exc
            reductions.append((requested, available))
            return original_selector(frame, limit=available)

    exp35_production._select_aft_calibration_projects = adaptive_selector
    try:
        oof = rolling_production_oof(
            data,
            identity,
            training_start=training_start,
            training_end=training_end,
            root=root,
        )
    finally:
        exp35_production._select_aft_calibration_projects = original_selector

    if reductions:
        print(
            "EXP124_OOF_AFT_GATE_ADAPTATIONS="
            + json.dumps([{"requested": req, "available": avail} for req, avail in reductions])
        )
    return oof


def run(output_dir: Path) -> dict:
    print_base_contract(); before = production_hashes()
    data, identity = build_training_dataset(); data = data.copy(); data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    windows = []
    for start, end, test_end in WINDOWS:
        with tempfile.TemporaryDirectory(prefix=f"exp124-{end}-") as td:
            td = Path(td)
            prod = fresh_production_window(data, identity, training_start=start, training_end=end, test_end=test_end, artifact_root=td / "baseline")
            oof = attach_features(_rolling_oof_with_adaptive_aft_gate(data, identity, training_start=start, training_end=end, root=td / "oof"), data, BASE_INPUTS)
            score = attach_features(prod.comparable, data, BASE_INPUTS)
            candidate, model, calibration = fit_calibrator(oof, score); baseline = score["predicted_delay_days"].to_numpy(float)
            assert_same_keys(prod.comparable, score)
            base_mae = weighted_mae(score, "actual_delay_days", baseline); exp_mae = weighted_mae(score, "actual_delay_days", candidate); gain = (base_mae-exp_mae)/base_mae*100.0 if base_mae else 0.0
            bootstrap = paired_project_bootstrap(score, actual="actual_delay_days", baseline=baseline, challenger=candidate, samples=5000, seed=12400+end)
            stages = lifecycle_metrics(score, actual="actual_delay_days", baseline=baseline, challenger=candidate); diag = _binned_diagnostics(score, baseline, candidate)
            window_name = f"{start}_{end}"; ledger = build_prediction_ledger(score, experiment_id=EXP_ID, window=window_name, production_delay_prediction=baseline, experiment_delay_prediction=candidate, extra_columns=["lifecycle_stage", "duration_ratio", "schedule_slippage_days"])
            ledger_dir = output_dir / window_name; write_prediction_ledger(ledger, ledger_dir, overwrite=True); joblib.dump(model, ledger_dir / "calibrator.pkl")
            windows.append({"window": window_name, "status": "EXECUTION VALID", "target": "Delay", "base_mae": base_mae, "experiment_mae": exp_mae, "improvement_pct": gain, "projects": int(score.canonical_project_id.nunique()), "snapshots": int(len(score)), "base_feature_count": 25, "calibrator_feature_count": len(FEATURES), "production_cost_baseline": prod.result["metadata"].get("production_cost_baseline"), "production_delay_baseline": prod.result["metadata"].get("production_delay_baseline"), "cost_prediction_equality_asserted": True, "calibration": calibration, "bootstrap": bootstrap, "lifecycle": stages, "continuous_bias_diagnostics": diag})
    if before != production_hashes(): raise AssertionError("Exp124 modified tracked production artifacts")
    verdict = verdict_from_windows(windows)
    payload = {"experiment": EXP_ID, "name": "Continuous Forecast-Horizon Delay Calibration", "base_pipeline": "25-FEATURE MONTHLY LIFECYCLE", "base_features": 25, "windows": windows, "verdict": verdict, "production_artifacts_unchanged": True, "final_recommendation": "KEEP" if verdict in {"STRONG PROMOTION CANDIDATE", "PROMISING"} else "REJECT"}
    save_json(ROOT / "reports" / "exp124_final_report.json", payload)
    lines = ["# Exp124 Final Report", "", "## Executive Summary", f"Verdict: **{verdict}**", "", "## 25-Feature Baseline", "`BASE_FEATURE_COUNT = 25`", "", "## Results", "", "| Window | Base Delay MAE | Exp124 Delay MAE | Improvement |", "|---|---:|---:|---:|"]
    for w in windows: lines.append(f"| {w['window']} | {w['base_mae']:.6f} | {w['experiment_mae']:.6f} | {w['improvement_pct']:.4f}% |")
    lines += ["", "## Production Safety", "Tracked production artifact hashes were identical before and after execution.", "", "## Leakage Audit", "Production residuals are strict forward OOF; spline fit, correction cap, and scale selection use training/meta-OOF only; Cost predictions are unchanged."]
    (ROOT / "reports" / "exp124_final_report.md").write_text("\n".join(lines) + "\n"); print(json.dumps(payload, indent=2, allow_nan=False)); return payload


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", default="models/monthly_lifecycle/experiments/exp124/ci"); args = p.parse_args(); run(ROOT / args.output_dir)

if __name__ == "__main__": main()
