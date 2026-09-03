"""Exp135: dynamic local-competence ET/LGBM/XGB ensemble.

Batch-only scientific experiment. Production is never modified. Local competence
is estimated exclusively from strict rolling temporal OOF errors and weights are
computed per holdout snapshot in a training-fitted numeric representation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from backend.app.ml.experiments.nextgen_common import _compare, _gain, _metric, _prepare
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline, _regressors, temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production

EXPERIMENT_ID = "Exp135"
SEED = 13503
FAMILIES = ("extra_trees", "lightgbm", "xgboost")
MIN_NEIGHBORS = 12
K_NEIGHBORS = 48
FALLBACK_BLEND = 0.75


def _numeric_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [f for f in features if f in frame and pd.api.types.is_numeric_dtype(frame[f])]


def _fit_representation(train: pd.DataFrame, features: list[str]):
    cols = _numeric_features(train, features)
    if not cols:
        raise ValueError("Exp135 requires at least one numeric production feature")
    med = train[cols].apply(pd.to_numeric, errors="coerce").median()
    scaler = RobustScaler().fit(train[cols].apply(pd.to_numeric, errors="coerce").fillna(med))
    return cols, med, scaler


def _transform(frame: pd.DataFrame, state) -> np.ndarray:
    cols, med, scaler = state
    x = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    return np.asarray(scaler.transform(x), dtype=float)


def _strict_oof(train: pd.DataFrame, features: list[str], target: str, seed: int) -> pd.DataFrame:
    rows = []
    for fold_no, (fit, val, year) in enumerate(_rolling_folds(train)):
        part = val[["canonical_project_id", "snapshot_date", "completion_year", target, *features]].copy()
        for family in FAMILIES:
            model = _fit_pipeline(_regressors(seed + fold_no)[family], fit, features, target)
            part[f"pred_{family}"] = np.asarray(model.predict(val[features]), dtype=float)
            part[f"err_{family}"] = np.abs(pd.to_numeric(val[target], errors="coerce").to_numpy(float) - part[f"pred_{family}"].to_numpy(float))
        part["oof_fold_year"] = int(year)
        rows.append(part)
    if not rows:
        raise ValueError("No strict forward OOF folds available for Exp135")
    return pd.concat(rows, ignore_index=True)


def local_competence_weights(query: np.ndarray, query_year: int, history_x: np.ndarray,
                             history_year: np.ndarray, errors: np.ndarray,
                             min_neighbors: int = MIN_NEIGHBORS, k: int = K_NEIGHBORS) -> tuple[np.ndarray | None, int]:
    """Return non-negative normalized inverse-error weights from prior folds only."""
    eligible = np.flatnonzero(np.asarray(history_year, int) < int(query_year))
    if len(eligible) < min_neighbors:
        return None, int(len(eligible))
    dist = np.linalg.norm(history_x[eligible] - np.asarray(query, float), axis=1)
    finite = np.isfinite(dist)
    eligible, dist = eligible[finite], dist[finite]
    if len(eligible) < min_neighbors:
        return None, int(len(eligible))
    order = np.argsort(dist)[: min(k, len(dist))]
    idx, d = eligible[order], dist[order]
    kernel = 1.0 / (d + 0.25)
    local = errors[idx]
    valid = np.isfinite(local).all(axis=1)
    if int(valid.sum()) < min_neighbors:
        return None, int(valid.sum())
    kernel, local = kernel[valid], local[valid]
    mae = np.average(local, axis=0, weights=kernel)
    inv = 1.0 / np.maximum(mae, 1e-6)
    weights = inv / inv.sum()
    if not np.isfinite(weights).all() or np.any(weights < 0):
        return None, int(valid.sum())
    return weights, int(valid.sum())


def _target_prediction(train: pd.DataFrame, cohort: pd.DataFrame, features: list[str], target: str,
                       production_prediction: np.ndarray, seed: int):
    oof = _strict_oof(train, features, target, seed)
    state = _fit_representation(train, features)
    hx = _transform(oof, state)
    qx = _transform(cohort, state)
    hyears = pd.to_numeric(oof["completion_year"], errors="coerce").fillna(-1).to_numpy(int)
    errors = oof[[f"err_{f}" for f in FAMILIES]].to_numpy(float)
    final_models = {f: _fit_pipeline(_regressors(seed)[f], train, features, target) for f in FAMILIES}
    family_pred = np.column_stack([np.asarray(final_models[f].predict(cohort[features]), float) for f in FAMILIES])
    qyears = pd.to_numeric(cohort["completion_year"], errors="coerce").fillna(10**9).to_numpy(int)
    out = np.asarray(production_prediction, float).copy()
    weight_rows, neighbors, fallback = [], [], 0
    for i in range(len(cohort)):
        w, n = local_competence_weights(qx[i], int(qyears[i]), hx, hyears, errors)
        neighbors.append(n)
        if w is None or not np.isfinite(family_pred[i]).all():
            fallback += 1
            weight_rows.append([np.nan] * len(FAMILIES))
            continue
        local_pred = float(np.dot(w, family_pred[i]))
        out[i] = FALLBACK_BLEND * local_pred + (1.0 - FALLBACK_BLEND) * out[i]
        weight_rows.append(w.tolist())
    warr = np.asarray(weight_rows, float)
    diag = {
        "oof_rows": int(len(oof)), "fallback_percentage": float(100 * fallback / max(len(cohort), 1)),
        "neighbor_count_mean": float(np.mean(neighbors)) if neighbors else 0.0,
        "average_weights": {f: float(np.nanmean(warr[:, j])) if np.isfinite(warr[:, j]).any() else None for j, f in enumerate(FAMILIES)},
        "weight_std": {f: float(np.nanstd(warr[:, j])) if np.isfinite(warr[:, j]).any() else None for j, f in enumerate(FAMILIES)},
        "prior_fold_rule": "competence source completion_year < query completion_year",
        "production_shrinkage": 1.0 - FALLBACK_BLEND,
    }
    return out, diag


def fit_against_production(*, data: pd.DataFrame, training_start: int, training_end: int, test_end: int,
                           production_bundle: dict, production_receipt: dict | None = None) -> dict:
    frame = _prepare(data)
    train, test = temporal_project_split(frame, training_start, training_end, test_end)
    cohort = _compare(test)
    cost_model, delay_model = production_bundle["cost"], production_bundle["delay"]
    pc = np.asarray(cost_model.predict(cohort), float)
    pdly = np.maximum(0, np.asarray(delay_model.predict(cohort), float))
    ec, cost_diag = _target_prediction(train, cohort, list(cost_model.features), "actual_cost_overrun_percentage", pc, SEED)
    ed, delay_diag = _target_prediction(train, cohort, list(delay_model.features), "actual_delay_days", pdly, SEED + 100)
    ed = np.maximum(0, ed)
    pcm, ecm = _metric(cohort, "actual_cost_overrun_percentage", pc), _metric(cohort, "actual_cost_overrun_percentage", ec)
    pdm, edm = _metric(cohort, "actual_delay_days", pdly), _metric(cohort, "actual_delay_days", ed)
    overall = {"production_cost_mae": pcm, "experiment_cost_mae": ecm, "cost_improvement_percentage": _gain(pcm, ecm),
               "production_delay_mae": pdm, "experiment_delay_mae": edm, "delay_improvement_percentage": _gain(pdm, edm),
               "comparison_test_projects": int(cohort.canonical_project_id.nunique()), "comparison_test_snapshots": int(len(cohort))}
    return {"experiment": {"experiment_id": EXPERIMENT_ID, "diagnostics": {"cost": cost_diag, "delay": delay_diag}, "seed": SEED}, "overall_comparison": overall}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--start", type=int, required=True); p.add_argument("--end", type=int, required=True); p.add_argument("--test-end", type=int, required=True); p.add_argument("--output", required=True); a = p.parse_args()
    data, _ = build_training_dataset(); data = data.copy(); data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    bundle, receipt = fast_current_production(data, a.start, a.end, a.test_end)
    result = fit_against_production(data=data, training_start=a.start, training_end=a.end, test_end=a.test_end, production_bundle=bundle, production_receipt=receipt)
    m = result["overall_comparison"]; payload = {"experiment": EXPERIMENT_ID, "window": f"{a.start}-{a.end}", "production": {"cost_mae": m["production_cost_mae"], "delay_mae": m["production_delay_mae"]}, "experiment_metrics": {"cost_mae": m["experiment_cost_mae"], "delay_mae": m["experiment_delay_mae"]}, "improvement": {"cost_percent": m["cost_improvement_percentage"], "delay_percent": m["delay_improvement_percentage"]}, "cohort": {"projects": m["comparison_test_projects"], "snapshots": m["comparison_test_snapshots"]}, "diagnostics": result["experiment"]["diagnostics"], "verdict": {"cost": "IMPROVED" if m["cost_improvement_percentage"] > 0 else "REGRESSED" if m["cost_improvement_percentage"] < 0 else "UNCHANGED", "delay": "IMPROVED" if m["delay_improvement_percentage"] > 0 else "REGRESSED" if m["delay_improvement_percentage"] < 0 else "UNCHANGED"}}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"WINDOW={a.start}_{a.end}"); print(f"PRODUCTION_COST_MAE={m['production_cost_mae']:.6f}"); print(f"EXPERIMENT_COST_MAE={m['experiment_cost_mae']:.6f}"); print(f"COST_IMPROVEMENT_PERCENT={m['cost_improvement_percentage']:.6f}"); print(f"PRODUCTION_DELAY_MAE={m['production_delay_mae']:.6f}"); print(f"EXPERIMENT_DELAY_MAE={m['experiment_delay_mae']:.6f}"); print(f"DELAY_IMPROVEMENT_PERCENT={m['delay_improvement_percentage']:.6f}"); print(f"PROJECT_COUNT={m['comparison_test_projects']}"); print(f"SNAPSHOT_COUNT={m['comparison_test_snapshots']}"); print(f"VERDICT_COST={payload['verdict']['cost']}"); print(f"VERDICT_DELAY={payload['verdict']['delay']}")

if __name__ == "__main__": main()
