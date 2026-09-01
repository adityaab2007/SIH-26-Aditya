"""Exp121 — actual official rainfall exposure (Delay only).

The current U1 production Delay forecast remains the anchor. Weather features
are used only when a deterministic project geography can be matched to a
versioned official IMD/OGD rainfall observation available by the snapshot date.
Unmatched/stale rows retain the exact U1 prediction.
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

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.experiments.prediction_ledger import build_prediction_ledger, write_prediction_ledger
from backend.app.ml.experiments.scientific_challenger_utils import (
    WINDOWS, assert_same_keys, fit_bounded_residual_correction, fresh_production_window,
    lifecycle_metrics, paired_project_bootstrap, print_base_contract, production_hashes,
    rolling_production_oof, save_json, verdict_from_windows, weighted_mae,
)

ROOT = Path(__file__).resolve().parents[4]
EXP_ID = "exp121"
EXTERNAL_ROOT = ROOT / "data" / "external" / "exp121_rainfall"
RAINFALL_FILE = EXTERNAL_ROOT / "rainfall_monthly.csv"
MANIFEST = EXTERNAL_ROOT / "source_manifest.json"
GEOGRAPHY_FIELDS = (("district", "district"), ("state", "state"), ("meteorological_subdivision", "subdivision"), ("subdivision", "subdivision"))
FEATURES = [
    "exp121_rain_30d", "exp121_rain_60d", "exp121_rain_90d", "exp121_rain_180d",
    "exp121_anomaly_90d", "exp121_volatility_6m", "exp121_acceleration",
    "exp121_monsoon_cumulative", "exp121_rain_x_slippage", "exp121_rain_x_progress_deviation",
    "exp121_rain_x_expenditure_slowdown",
]
MAX_WEATHER_STALENESS_DAYS = 45


def _norm(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def load_verified_rainfall() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if manifest.get("source_institution") != "India Meteorological Department (IMD), Ministry of Earth Sciences, Government of India":
        return pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "official IMD source manifest missing/invalid"}
    if not RAINFALL_FILE.exists():
        return pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "versioned cleaned official rainfall bundle not present"}
    rain = pd.read_csv(RAINFALL_FILE)
    required = {"period", "geography_type", "geography_name", "rainfall_mm"}
    if not required.issubset(rain.columns):
        return pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": f"rainfall bundle missing {sorted(required - set(rain.columns))}"}
    rain["period"] = pd.to_datetime(rain["period"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    rain["geography_type"] = rain["geography_type"].astype(str).str.lower().str.strip()
    rain["geography_key"] = rain["geography_name"].map(_norm)
    rain["rainfall_mm"] = pd.to_numeric(rain["rainfall_mm"], errors="coerce")
    if "normal_mm" not in rain:
        rain["normal_mm"] = np.nan
    rain["normal_mm"] = pd.to_numeric(rain["normal_mm"], errors="coerce")
    rain = rain.dropna(subset=["period", "rainfall_mm"])
    if rain.duplicated(["period", "geography_type", "geography_key"]).any():
        raise ValueError("official rainfall bundle contains duplicate geography/month rows")
    if rain.empty:
        return rain, {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "rainfall bundle has no valid observations"}
    return rain.sort_values(["geography_type", "geography_key", "period"]), {
        "status": "VERIFIED",
        "coverage_start": str(rain.period.min().date()),
        "coverage_end": str(rain.period.max().date()),
        "geographies": int(rain[["geography_type", "geography_key"]].drop_duplicates().shape[0]),
    }


def resolve_project_geography(row: pd.Series) -> tuple[str, str, str]:
    """Return (type, normalized key, method) without project-name guessing."""
    for field, geo_type in GEOGRAPHY_FIELDS:
        if field in row.index and _norm(row.get(field)):
            return geo_type, _norm(row.get(field)), f"exact_normalized_{field}"
    return "", "", "unavailable_no_reliable_geography"


def _weather_features_for_one(snapshot: pd.Timestamp, group: pd.DataFrame) -> dict:
    eligible = group[group.period.le(snapshot)].copy()
    if eligible.empty:
        return {}
    latest = eligible.iloc[-1]
    staleness = (snapshot - latest.period).days
    if staleness < 0 or staleness > MAX_WEATHER_STALENESS_DAYS:
        return {}
    def trailing(months: int) -> pd.DataFrame:
        cutoff = snapshot - pd.DateOffset(months=months)
        return eligible[eligible.period.gt(cutoff)]
    t1, t2, t3, t6 = trailing(1), trailing(2), trailing(3), trailing(6)
    r3 = float(t3.rainfall_mm.sum()) if len(t3) else np.nan
    n3 = float(t3.normal_mm.sum()) if t3.normal_mm.notna().any() else np.nan
    anomaly = ((r3 - n3) / n3 * 100.0) if np.isfinite(n3) and n3 > 0 else np.nan
    recent = t6.rainfall_mm.to_numpy(float)
    accel = float(t1.rainfall_mm.sum() - t2.iloc[:-1].rainfall_mm.sum()) if len(t2) >= 2 else np.nan
    year = snapshot.year
    monsoon = eligible[(eligible.period.dt.year.eq(year)) & (eligible.period.dt.month.between(6, 9)) & eligible.period.le(snapshot)]
    return {
        "exp121_rain_30d": float(t1.rainfall_mm.sum()) if len(t1) else np.nan,
        "exp121_rain_60d": float(t2.rainfall_mm.sum()) if len(t2) else np.nan,
        "exp121_rain_90d": r3,
        "exp121_rain_180d": float(t6.rainfall_mm.sum()) if len(t6) else np.nan,
        "exp121_anomaly_90d": anomaly,
        "exp121_volatility_6m": float(np.std(recent, ddof=1)) if len(recent) >= 2 else np.nan,
        "exp121_acceleration": accel,
        "exp121_monsoon_cumulative": float(monsoon.rainfall_mm.sum()) if len(monsoon) else 0.0,
        "external_feature_timestamp": latest.period,
    }


def build_rainfall_features(rows: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy(); data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    lookup = {(t, k): g.sort_values("period") for (t, k), g in rain.groupby(["geography_type", "geography_key"], sort=False)} if not rain.empty else {}
    out = []
    for _, row in data.iterrows():
        geo_type, geo_key, method = resolve_project_geography(row)
        item = {"canonical_project_id": row.canonical_project_id, "snapshot_date": row.snapshot_date, "external_data_available": False, "external_match_method": method, "external_geography_type": geo_type, "external_geography_key": geo_key, "external_feature_timestamp": pd.NaT}
        if geo_type and geo_key and (geo_type, geo_key) in lookup:
            item.update(_weather_features_for_one(row.snapshot_date, lookup[(geo_type, geo_key)]))
            item["external_data_available"] = pd.notna(item.get("external_feature_timestamp"))
        rain90 = item.get("exp121_rain_90d", np.nan)
        item["exp121_rain_x_slippage"] = rain90 * float(pd.to_numeric(pd.Series([row.get("schedule_slippage_days")]), errors="coerce").iloc[0]) if pd.notna(rain90) and pd.notna(row.get("schedule_slippage_days")) else np.nan
        item["exp121_rain_x_progress_deviation"] = rain90 * float(pd.to_numeric(pd.Series([row.get("progress_deviation")]), errors="coerce").iloc[0]) if pd.notna(rain90) and pd.notna(row.get("progress_deviation")) else np.nan
        slowdown = max(0.0, -float(pd.to_numeric(pd.Series([row.get("expenditure_ratio")]), errors="coerce").iloc[0])) if pd.notna(row.get("expenditure_ratio")) else np.nan
        item["exp121_rain_x_expenditure_slowdown"] = rain90 * slowdown if pd.notna(rain90) and pd.notna(slowdown) else np.nan
        for f in FEATURES:
            item.setdefault(f, np.nan)
        if item["external_data_available"] and item["external_feature_timestamp"] > row.snapshot_date:
            raise AssertionError("rainfall after snapshot entered Exp121 feature vector")
        out.append(item)
    result = pd.DataFrame(out)
    if result.duplicated(["canonical_project_id", "snapshot_date"]).any():
        raise AssertionError("Exp121 feature construction duplicated project/snapshot rows")
    return result


def _coverage_by_year(frame: pd.DataFrame) -> dict:
    years = pd.to_datetime(frame.snapshot_date, errors="coerce").dt.year
    return {str(int(y)): {"rows": int((years == y).sum()), "available": int(((years == y) & frame.external_data_available).sum())} for y in sorted(years.dropna().unique())}


def _matched_unmatched_mae(score: pd.DataFrame, baseline, candidate) -> dict:
    work = score.copy(); work["base"] = np.asarray(baseline, float); work["candidate"] = np.asarray(candidate, float); out = {}
    for label, mask in {"matched": work.external_data_available, "unmatched": ~work.external_data_available}.items():
        part = work[mask]
        out[label] = {"rows": int(len(part)), "projects": int(part.canonical_project_id.nunique()), "baseline_mae": weighted_mae(part, "actual_delay_days", part.base) if len(part) else None, "candidate_mae": weighted_mae(part, "actual_delay_days", part.candidate) if len(part) else None}
    return out


def run(output_dir: Path) -> dict:
    print_base_contract(); before = production_hashes(); rain, source = load_verified_rainfall(); data, identity = build_training_dataset(); data = data.copy(); data["snapshot_date"] = pd.to_datetime(data.snapshot_date, errors="coerce"); weather = build_rainfall_features(data, rain); windows = []
    weather_cols = ["canonical_project_id", "snapshot_date", *FEATURES, "external_data_available", "external_match_method", "external_geography_type", "external_geography_key", "external_feature_timestamp"]
    for start, end, test_end in WINDOWS:
        with tempfile.TemporaryDirectory(prefix=f"exp121-{end}-") as td:
            td = Path(td); prod = fresh_production_window(data, identity, training_start=start, training_end=end, test_end=test_end, artifact_root=td / "baseline")
            score = prod.comparable.merge(weather[weather_cols], on=["canonical_project_id", "snapshot_date"], how="left", validate="one_to_one"); score["external_data_available"] = score.external_data_available.fillna(False); score["external_match_method"] = score.external_match_method.fillna("unavailable_no_reliable_geography")
            assert_same_keys(prod.comparable, score); baseline = score.predicted_delay_days.to_numpy(float); cost_baseline = score.predicted_cost_overrun.to_numpy(float)
            coverage_projects = int(score.loc[score.external_data_available, "canonical_project_id"].nunique()); coverage_ratio = coverage_projects / max(int(score.canonical_project_id.nunique()), 1)
            candidate = baseline.copy(); model = None; diag = {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "selected_scale": 0.0, "reason": "verified recent rainfall project coverage below 10%"}
            if source.get("status") == "VERIFIED" and coverage_ratio >= 0.10:
                oof = rolling_production_oof(data, identity, training_start=start, training_end=end, root=td / "oof").merge(weather[weather_cols[:-1]], on=["canonical_project_id", "snapshot_date"], how="left", validate="one_to_one"); oof["external_data_available"] = oof.external_data_available.fillna(False)
                candidate, diag, model = fit_bounded_residual_correction(oof, score, features=FEATURES, actual="actual_delay_days", production_col="predicted_delay_days", available_col="external_data_available", seed=12100 + end)
            base_mae = weighted_mae(score, "actual_delay_days", baseline); exp_mae = weighted_mae(score, "actual_delay_days", candidate); gain = (base_mae - exp_mae) / base_mae * 100.0 if base_mae else 0.0
            bootstrap = paired_project_bootstrap(score, actual="actual_delay_days", baseline=baseline, challenger=candidate, samples=5000, seed=12100 + end); stages = lifecycle_metrics(score, actual="actual_delay_days", baseline=baseline, challenger=candidate); status = "EXECUTION VALID" if diag.get("status") == "EXECUTION VALID" else "INSUFFICIENT VERIFIED EXTERNAL DATA"
            window = f"{start}_{end}"; ledger = build_prediction_ledger(score, experiment_id=EXP_ID, window=window, production_delay_prediction=baseline, experiment_delay_prediction=candidate, extra_columns=["lifecycle_stage", "external_data_available", "external_match_method", "external_geography_type", "external_feature_timestamp"]); ledger_dir = output_dir / window; write_prediction_ledger(ledger, ledger_dir, overwrite=True)
            if model is not None: joblib.dump(model, ledger_dir / "delay_residual_model.pkl")
            if not np.array_equal(cost_baseline, score.predicted_cost_overrun.to_numpy(float)): raise AssertionError("Exp121 changed Cost")
            windows.append({"window": window, "status": status, "base_mae": base_mae, "experiment_mae": exp_mae, "improvement_pct": gain, "projects": int(score.canonical_project_id.nunique()), "snapshots": int(len(score)), "external_project_coverage": coverage_ratio, "external_rows": int(score.external_data_available.sum()), "yearly_coverage": _coverage_by_year(score), "geographic_resolution_counts": score.external_geography_type.value_counts(dropna=False).to_dict(), "match_method_counts": score.external_match_method.value_counts(dropna=False).to_dict(), "matched_unmatched_mae": _matched_unmatched_mae(score, baseline, candidate), "base_feature_count": 25, "experimental_feature_count": len(FEATURES), "cost_prediction_equality_asserted": True, "residual_correction": diag, "bootstrap": bootstrap, "lifecycle": stages, "production_cost_baseline": prod.result["metadata"].get("production_cost_baseline"), "production_delay_baseline": prod.result["metadata"].get("production_delay_baseline")})
    if before != production_hashes(): raise AssertionError("Exp121 modified tracked production artifacts")
    verdict = verdict_from_windows(windows); payload = {"experiment": EXP_ID, "name": "Actual Weather / Rainfall Exposure", "target": "Delay", "source": source, "windows": windows, "verdict": verdict, "production_artifacts_unchanged": True, "final_recommendation": "KEEP" if verdict in {"STRONG PROMOTION CANDIDATE", "PROMISING"} else ("NEEDS DATA" if verdict == "INVALID / INSUFFICIENT DATA" else "REJECT")}
    save_json(ROOT / "reports" / "exp121_final_report.json", payload); lines = ["# Exp121 Final Report", "", f"Verdict: **{verdict}**", "", f"External source status: **{source.get('status')}**", "", "| Window | Base Delay MAE | Exp121 Delay MAE | Improvement | Status |", "|---|---:|---:|---:|---|"]
    for w in windows: lines.append(f"| {w['window']} | {w['base_mae']:.6f} | {w['experiment_mae']:.6f} | {w['improvement_pct']:.4f}% | {w['status']} |")
    lines += ["", "Rows without a deterministic geography or recent official rainfall retain the exact U1 production prediction. Project names are never parsed to guess geography. Cost is unchanged."]; (ROOT / "reports" / "exp121_final_report.md").write_text("\n".join(lines) + "\n"); print(json.dumps(payload, indent=2, allow_nan=False)); return payload


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", default="models/monthly_lifecycle/experiments/exp121/ci"); a = p.parse_args(); run(ROOT / a.output_dir)


if __name__ == "__main__": main()
