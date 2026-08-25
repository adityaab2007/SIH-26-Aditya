"""Shared statistical evaluation utilities for candidate experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_project_mae_comparison(
    rows: pd.DataFrame,
    *,
    actual: str,
    baseline_prediction: str,
    candidate_prediction: str,
    project_id: str = "canonical_project_id",
    bootstrap_samples: int = 1000,
    seed: int = 26103,
) -> dict:
    """Compare candidate vs baseline while respecting project-level dependence.

    Lifecycle snapshots from the same project are correlated.  We therefore
    reduce each project to its own MAE and bootstrap projects, not individual
    snapshots.  Positive improvement means the candidate has lower MAE.
    """
    required = [actual, baseline_prediction, candidate_prediction, project_id]
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError("paired comparison missing columns: " + ", ".join(missing))

    frame = rows[required].copy()
    for column in (actual, baseline_prediction, candidate_prediction):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    if frame.empty:
        raise ValueError("paired comparison has no complete rows")

    frame["baseline_abs_error"] = (frame[baseline_prediction] - frame[actual]).abs()
    frame["candidate_abs_error"] = (frame[candidate_prediction] - frame[actual]).abs()
    per_project = frame.groupby(project_id, sort=True).agg(
        baseline_mae=("baseline_abs_error", "mean"),
        candidate_mae=("candidate_abs_error", "mean"),
        snapshots=(actual, "size"),
    )
    if len(per_project) < 2:
        raise ValueError("paired comparison requires at least two projects")

    baseline_mae = float(per_project.baseline_mae.mean())
    candidate_mae = float(per_project.candidate_mae.mean())
    absolute_improvement = baseline_mae - candidate_mae
    percentage_improvement = (absolute_improvement / baseline_mae * 100.0) if baseline_mae else None

    bootstrap_samples = int(bootstrap_samples)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    rng = np.random.default_rng(seed)
    baseline_values = per_project.baseline_mae.to_numpy(dtype=float)
    candidate_values = per_project.candidate_mae.to_numpy(dtype=float)
    project_count = len(per_project)
    sampled = rng.integers(0, project_count, size=(bootstrap_samples, project_count))
    sampled_baseline = baseline_values[sampled].mean(axis=1)
    sampled_candidate = candidate_values[sampled].mean(axis=1)
    sampled_improvement = sampled_baseline - sampled_candidate
    sampled_pct = np.divide(
        sampled_improvement * 100.0,
        sampled_baseline,
        out=np.full_like(sampled_improvement, np.nan),
        where=sampled_baseline != 0,
    )
    finite_pct = sampled_pct[np.isfinite(sampled_pct)]

    return {
        "evaluation_unit": "project",
        "projects": int(project_count),
        "snapshots": int(len(frame)),
        "baseline_project_macro_mae": round(baseline_mae, 4),
        "candidate_project_macro_mae": round(candidate_mae, 4),
        "absolute_mae_improvement": round(absolute_improvement, 4),
        "percentage_mae_improvement": round(float(percentage_improvement), 4) if percentage_improvement is not None else None,
        "bootstrap_samples": bootstrap_samples,
        "improvement_95pct_ci": [
            round(float(np.quantile(finite_pct, 0.025)), 4),
            round(float(np.quantile(finite_pct, 0.975)), 4),
        ] if len(finite_pct) else [None, None],
        "probability_candidate_better": round(float(np.mean(sampled_improvement > 0)), 4),
        "interpretation": "Positive improvement favors the candidate. Confidence intervals resample whole projects so repeated lifecycle snapshots do not masquerade as independent evidence.",
    }
