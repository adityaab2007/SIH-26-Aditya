"""Strict scientific helpers for isolated Exp120-124 challengers.

This module is experiment-only. It trains the current production stack only
inside caller-provided temporary roots and never writes production artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import MODEL_ROOT
from backend.app.ml.production_u1_delay_baseline import train_window_with_promoted_cost_and_delay as train_current_production

ROOT = Path(__file__).resolve().parents[4]
BASE_CONTRACT_SOURCE = ROOT / "models" / "monthly_lifecycle" / "2001_2019" / "metadata.json"
BASE_25_FEATURES = [
    "approved_cost_cr", "sector_average_delay", "sector_average_cost_overrun", "sector", "project_size_category",
    "cumulative_expenditure_cr", "expenditure_ratio", "schedule_slippage_days", "schedule_slippage_ratio",
    "elapsed_duration_days", "planned_duration_days", "duration_ratio", "expected_progress_percentage",
    "revised_cost_cr", "cost_escalation_percentage", "implementing_agency", "cost_growth_velocity_3m",
    "cost_growth_velocity_6m", "cost_acceleration", "sector_delay_rate", "sector_cost_overrun_rate",
    "agency_average_delay", "agency_average_cost_overrun", "agency_delay_rate", "agency_cost_overrun_rate",
]
KEYS = ["canonical_project_id", "snapshot_date"]
WINDOWS = [(2001, 2019, 2025), (2001, 2021, 2025)]


def assert_base_25_contract(metadata: dict | None = None) -> list[str]:
    if len(BASE_25_FEATURES) != 25 or len(set(BASE_25_FEATURES)) != 25:
        raise AssertionError("BASE_25_FEATURES must contain exactly 25 unique features")
    if not BASE_CONTRACT_SOURCE.exists():
        raise FileNotFoundError(f"25-feature evidence artifact missing: {BASE_CONTRACT_SOURCE}")
    tracked = json.loads(BASE_CONTRACT_SOURCE.read_text())
    tracked_features = list(tracked.get("features_used") or [])
    if tracked_features != BASE_25_FEATURES:
        raise AssertionError("Tracked 25-feature production contract changed; refusing to guess a replacement")
    if metadata is not None and list(metadata.get("features_used") or []) != BASE_25_FEATURES:
        raise AssertionError("Fresh production retrain no longer reproduces the frozen 25-feature lifecycle contract")
    return list(BASE_25_FEATURES)


def print_base_contract() -> None:
    features = assert_base_25_contract()
    print("BASE PIPELINE = 25-FEATURE MONTHLY LIFECYCLE")
    print("BASE_FEATURE_COUNT = 25")
    print("BASE_FEATURES = " + json.dumps(features))
    print(f"BASE_CONTRACT_SOURCE = {BASE_CONTRACT_SOURCE.relative_to(ROOT)}")


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def production_hashes() -> dict[str, str]:
    names = {"cost_model.pkl", "delay_model.pkl", "risk_model.pkl", "metadata.json", "evaluation_results.json"}
    result: dict[str, str] = {}
    if not MODEL_ROOT.exists():
        return result
    for path in sorted(MODEL_ROOT.glob("*/*")):
        if path.is_file() and path.name in names and "experiments" not in path.parts:
            result[str(path.relative_to(ROOT))] = _sha(path)
    return result


def _eligible_mask(validation: pd.DataFrame) -> pd.Series:
    if "cost_evaluation_eligible" not in validation:
        raise AssertionError("Current production validation is missing cost_evaluation_eligible")
    raw = validation["cost_evaluation_eligible"]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"true", "1", "yes"})


@dataclass
class ProductionRun:
    result: dict
    validation: pd.DataFrame
    comparable: pd.DataFrame
    artifact_dir: Path


def fresh_production_window(data: pd.DataFrame, identity: pd.DataFrame, *, training_start: int, training_end: int, test_end: int, artifact_root: Path) -> ProductionRun:
    result = train_current_production(training_start, training_end, test_end, data=data, identity=identity, artifact_root=artifact_root)
    assert_base_25_contract(result.get("metadata") or {})
    target = artifact_root / f"{training_start}_{training_end}"
    validation = pd.read_csv(target / "prediction_validation.csv", low_memory=False)
    validation["snapshot_date"] = pd.to_datetime(validation["snapshot_date"], errors="coerce")
    comparable = assign_project_balanced_weights(validation.loc[_eligible_mask(validation)].copy())
    if comparable.duplicated(KEYS).any():
        raise AssertionError("Production comparison cohort contains duplicate project/snapshot keys")
    return ProductionRun(result=result, validation=validation, comparable=comparable, artifact_dir=target)


def rolling_production_oof(data: pd.DataFrame, identity: pd.DataFrame, *, training_start: int, training_end: int, root: Path, n_folds: int = 4) -> pd.DataFrame:
    years = sorted(int(y) for y in pd.to_numeric(data.get("completion_year"), errors="coerce").dropna().unique() if training_start < int(y) <= training_end)
    if len(years) < n_folds + 2:
        raise ValueError("Insufficient completion-year history for strict rolling production OOF")
    chunks: list[pd.DataFrame] = []
    for year in years[-n_folds:]:
        run = fresh_production_window(data, identity, training_start=training_start, training_end=year - 1, test_end=year, artifact_root=root / f"oof_{year}")
        part = run.comparable.copy()
        part["oof_year"] = int(year)
        chunks.append(part)
    oof = pd.concat(chunks, ignore_index=True)
    if oof.duplicated(KEYS).any():
        raise AssertionError("Rolling production OOF contains duplicate keys")
    return oof


def canonicalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["canonical_project_id"] = out["canonical_project_id"].astype("string").str.strip()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    if out[KEYS].isna().any().any():
        raise ValueError("canonical project/snapshot keys contain missing values")
    return out


def attach_features(rows: pd.DataFrame, feature_frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    left = canonicalize_keys(rows).reset_index(drop=True)
    right = canonicalize_keys(feature_frame)
    if right.duplicated(KEYS).any():
        raise AssertionError("Experimental feature table contains duplicate project/snapshot keys")
    wanted = KEYS + [c for c in columns if c in right.columns]
    merged = left.merge(right[wanted], on=KEYS, how="left", sort=False, validate="one_to_one", suffixes=("", "__exp"))
    if len(merged) != len(left) or not merged[KEYS].equals(left[KEYS]):
        raise AssertionError("Experimental join changed comparison keys")
    return merged


def assert_same_keys(baseline: pd.DataFrame, challenger: pd.DataFrame) -> None:
    a = canonicalize_keys(baseline)[KEYS].reset_index(drop=True)
    b = canonicalize_keys(challenger)[KEYS].reset_index(drop=True)
    if not a.equals(b):
        raise AssertionError("baseline_keys != challenger_keys")


def weighted_mae(frame: pd.DataFrame, actual: str, prediction) -> float:
    y = pd.to_numeric(frame[actual], errors="coerce").to_numpy(float)
    p = np.asarray(prediction, dtype=float)
    w = pd.to_numeric(frame["sample_weight"], errors="coerce").to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(w)
    return float(np.average(np.abs(y[mask] - p[mask]), weights=w[mask])) if mask.any() else float("nan")


def weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, dtype=float); weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0); values, weights = values[mask], weights[mask]
    if not len(values): return 0.0
    order = np.argsort(values); values, weights = values[order], weights[order]
    if float(weights.sum()) <= 0: return float(np.quantile(values, q))
    return float(values[min(np.searchsorted(np.cumsum(weights), q * float(weights.sum()), side="left"), len(values) - 1)])


def residual_pipeline(seed: int) -> Pipeline:
    model = LGBMRegressor(n_estimators=140, learning_rate=0.025, max_depth=2, num_leaves=7, min_child_samples=80, subsample=0.85, colsample_bytree=0.9, reg_alpha=8.0, reg_lambda=30.0, random_state=seed, verbosity=-1)
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", model)])


def _fit_residual_model(frame: pd.DataFrame, features: list[str], residual_col: str, seed: int) -> Pipeline:
    model = residual_pipeline(seed)
    weight = pd.to_numeric(frame["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    model.fit(frame[features], frame[residual_col], model__sample_weight=weight)
    return model


def fit_bounded_residual_correction(oof: pd.DataFrame, score: pd.DataFrame, *, features: list[str], actual: str, production_col: str, available_col: str, seed: int, scales: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)):
    work = oof.copy(); score = score.copy()
    for col in features:
        if col not in work: work[col] = np.nan
        if col not in score: score[col] = np.nan
    work[available_col] = work.get(available_col, False).fillna(False).astype(bool)
    score[available_col] = score.get(available_col, False).fillna(False).astype(bool)
    work["__residual"] = pd.to_numeric(work[actual], errors="coerce") - pd.to_numeric(work[production_col], errors="coerce")
    covered = work[work[available_col] & work["__residual"].notna()].copy()
    support_projects = int(covered["canonical_project_id"].nunique())
    if len(covered) < 100 or support_projects < 20:
        status = "INSUFFICIENT VERIFIED EXTERNAL DATA" if "external" in available_col else "INSUFFICIENT TRAINING SUPPORT"
        return np.asarray(score[production_col], dtype=float), {"status": status, "coverage_rows": int(len(covered)), "coverage_projects": support_projects, "selected_scale": 0.0, "correction_cap": 0.0, "meta_oof_rows": 0}, None
    years = sorted(int(y) for y in pd.to_numeric(work["oof_year"], errors="coerce").dropna().unique())
    meta_parts = []
    for idx in range(1, len(years)):
        year = years[idx]
        fit = covered[pd.to_numeric(covered["oof_year"], errors="coerce") < year]
        val_all = work[pd.to_numeric(work["oof_year"], errors="coerce").eq(year)].copy()
        val_cov = val_all[val_all[available_col]].copy()
        if len(fit) < 100 or int(fit["canonical_project_id"].nunique()) < 20 or val_all.empty: continue
        model = _fit_residual_model(fit, features, "__residual", seed + idx)
        correction = np.zeros(len(val_all), dtype=float)
        if not val_cov.empty:
            correction[np.flatnonzero(val_all[available_col].to_numpy(bool))] = np.asarray(model.predict(val_cov[features]), dtype=float)
        part = val_all[[*KEYS, "sample_weight", actual, production_col, "oof_year"]].copy(); part["__correction"] = correction; meta_parts.append(part)
    cap = max(weighted_quantile(np.abs(covered["__residual"].to_numpy(float)), pd.to_numeric(covered["sample_weight"], errors="coerce").fillna(0.0).to_numpy(float), 0.90), 1e-9)
    selected_scale = 0.0; meta_mae = {}
    if meta_parts:
        meta = pd.concat(meta_parts, ignore_index=True); clipped = np.clip(meta["__correction"].to_numpy(float), -cap, cap)
        base = weighted_mae(meta, actual, meta[production_col].to_numpy(float))
        for scale in scales:
            pred = np.asarray(meta[production_col], dtype=float) + float(scale) * clipped
            if actual == "actual_delay_days": pred = np.maximum(0.0, pred)
            meta_mae[str(scale)] = weighted_mae(meta, actual, pred)
        selected_scale = min(scales, key=lambda s: (meta_mae[str(s)], s))
        if meta_mae[str(selected_scale)] >= base - 1e-12: selected_scale = 0.0
    final_model = _fit_residual_model(covered, features, "__residual", seed + 99)
    correction = np.zeros(len(score), dtype=float); score_cov = score[score[available_col]].copy()
    if not score_cov.empty and selected_scale > 0:
        locs = np.flatnonzero(score[available_col].to_numpy(bool)); correction[locs] = np.clip(np.asarray(final_model.predict(score_cov[features]), dtype=float), -cap, cap)
    candidate = np.asarray(score[production_col], dtype=float) + float(selected_scale) * correction
    if actual == "actual_delay_days": candidate = np.maximum(0.0, candidate)
    diagnostics = {"status": "EXECUTION VALID", "coverage_rows": int(len(covered)), "coverage_projects": support_projects, "score_coverage_rows": int(score[available_col].sum()), "score_coverage_projects": int(score.loc[score[available_col], "canonical_project_id"].nunique()), "selected_scale": float(selected_scale), "correction_cap": float(cap), "meta_oof_rows": int(sum(len(p) for p in meta_parts)), "meta_oof_mae_by_scale": meta_mae, "feature_count": len(features)}
    return candidate, diagnostics, final_model


def lifecycle_metrics(frame: pd.DataFrame, *, actual: str, baseline, challenger) -> dict:
    work = frame.copy(); work["__base"] = np.asarray(baseline, dtype=float); work["__challenger"] = np.asarray(challenger, dtype=float)
    result = {}; maes = []
    for stage in ["early", "mid", "late", "very_late"]:
        part = work[work.get("lifecycle_stage", pd.Series(index=work.index, dtype=object)).eq(stage)]
        if part.empty: result[stage] = {"available": False}; continue
        b = weighted_mae(part, actual, part["__base"]); c = weighted_mae(part, actual, part["__challenger"])
        result[stage] = {"available": True, "projects": int(part["canonical_project_id"].nunique()), "snapshots": int(len(part)), "baseline_mae": b, "challenger_mae": c, "improvement_pct": ((b-c)/b*100.0) if b else 0.0}; maes.append((b,c))
    result["equal_stage_macro_mae"] = {"baseline": float(np.mean([x[0] for x in maes])) if maes else None, "challenger": float(np.mean([x[1] for x in maes])) if maes else None}
    return result


def paired_project_bootstrap(frame: pd.DataFrame, *, actual: str, baseline, challenger, samples: int = 5000, seed: int = 26103) -> dict:
    work = frame[["canonical_project_id", "sample_weight", actual]].copy(); work["baseline"] = np.asarray(baseline, dtype=float); work["challenger"] = np.asarray(challenger, dtype=float)
    work["base_err"] = np.abs(pd.to_numeric(work[actual], errors="coerce") - work["baseline"]); work["challenger_err"] = np.abs(pd.to_numeric(work[actual], errors="coerce") - work["challenger"])
    records = []
    for project_id, g in work.groupby("canonical_project_id", sort=False):
        records.append((project_id, float(np.average(g["base_err"], weights=g["sample_weight"])), float(np.average(g["challenger_err"], weights=g["sample_weight"]))))
    per = pd.DataFrame(records, columns=["project", "base", "challenger"])
    improvement = per["base"].to_numpy(float) - per["challenger"].to_numpy(float); rng = np.random.default_rng(seed); draws = rng.integers(0, len(improvement), size=(samples, len(improvement))); boot = improvement[draws].mean(axis=1)
    base_mae = float(per["base"].mean()); cand_mae = float(per["challenger"].mean()); absolute = base_mae - cand_mae
    return {"samples": int(samples), "projects": int(len(per)), "baseline_mae": base_mae, "challenger_mae": cand_mae, "absolute_improvement": absolute, "percentage_improvement": (absolute/base_mae*100.0) if base_mae else 0.0, "bootstrap_median_improvement": float(np.median(boot)), "ci95": [float(np.quantile(boot,0.025)), float(np.quantile(boot,0.975))], "probability_challenger_beats_baseline": float(np.mean(boot > 0)), "project_win_rate": float(np.mean(improvement > 0))}


def verdict_from_windows(window_results: list[dict]) -> str:
    usable = [w for w in window_results if w.get("status") == "EXECUTION VALID"]
    if len(usable) != len(window_results): return "INVALID / INSUFFICIENT DATA"
    gains = [float(w["improvement_pct"]) for w in usable]; probs = [float((w.get("bootstrap") or {}).get("probability_challenger_beats_baseline", 0.0)) for w in usable]
    if all(g > 0 for g in gains) and all(p >= 0.95 for p in probs): return "STRONG PROMOTION CANDIDATE"
    if all(g >= -0.25 for g in gains) and any(g > 0 for g in gains): return "PROMISING"
    if any(g > 0 for g in gains) and any(g < -0.25 for g in gains): return "UNSTABLE"
    return "DO NOT PROMOTE"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
