from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from backend.app.ml.experiments.exp82_paimana_bottlenecks import (
    add_bottleneck_features,
    discover_text_columns,
    normalize_text,
)
from backend.app.ml.monthly_lifecycle import (
    OUTCOMES,
    assign_project_balanced_weights,
    build_training_dataset,
    load_monthly_snapshots,
    resolve_identities,
)
from backend.app.ml.production_u1_delay_baseline import (
    train_window_with_promoted_cost_and_delay as train_current_production,
)

KEYS = ["canonical_project_id", "snapshot_date"]
GENERIC_PATTERNS = {
    "delay": [r"\bdelay(?:ed)?\b", r"\bbehind schedule\b", r"\boverdue\b", r"\bslippage\b"],
    "on_track": [r"\bon schedule\b", r"\bon track\b", r"\bas per schedule\b", r"\bahead of schedule\b"],
    "pending": [r"\bpending\b", r"\bawait(?:ing)?\b", r"\bnot started\b"],
    "stalled": [r"\bstall(?:ed)?\b", r"\bhalt(?:ed)?\b", r"\bstopp(?:ed|age)\b", r"\bheld up\b"],
    "critical": [r"\bcritical\b", r"\bsevere\b", r"\bmajor issue\b"],
    "progressing": [r"\bongoing\b", r"\bin progress\b", r"\bunder implementation\b"],
    "completed": [r"\bcompleted\b", r"\bcommissioned\b"],
}
NEGATION = re.compile(r"\b(?:no|not|without|resolved|cleared)\b.{0,20}$")


def _keyword_active(text: str, patterns: list[str]) -> int:
    normalized = normalize_text(text)
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match and not NEGATION.search(normalized[max(0, match.start() - 25) : match.start()]):
            return 1
    return 0


def add_generic_status_features(frame: pd.DataFrame, text_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy().sort_values(KEYS, kind="stable")
    joined = result[text_columns].fillna("").astype(str).agg(" | ".join, axis=1).map(normalize_text)
    features = []
    for name, patterns in GENERIC_PATTERNS.items():
        active = f"exp82_status_{name}_active"
        seen = f"exp82_status_{name}_seen"
        count = f"exp82_status_{name}_count"
        result[active] = joined.map(lambda value, p=patterns: _keyword_active(value, p))
        result[seen] = result.groupby("canonical_project_id", sort=False)[active].cummax()
        result[count] = result.groupby("canonical_project_id", sort=False)[active].cumsum()
        features.extend([active, seen, count])
    result["exp82_status_text_length"] = joined.str.len().astype(float)
    result["exp82_status_changed"] = (
        joined.ne(joined.groupby(result["canonical_project_id"], sort=False).shift(1))
        & joined.groupby(result["canonical_project_id"], sort=False).shift(1).notna()
    ).astype(int)
    result["exp82_status_change_count"] = result.groupby("canonical_project_id", sort=False)[
        "exp82_status_changed"
    ].cumsum()
    features.extend(["exp82_status_text_length", "exp82_status_changed", "exp82_status_change_count"])
    return result, features


def _eligible_mask(frame: pd.DataFrame) -> pd.Series:
    raw = frame["cost_evaluation_eligible"]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_validation(root: Path, training_start: int, training_end: int) -> pd.DataFrame:
    frame = pd.read_csv(
        root / f"{training_start}_{training_end}" / "prediction_validation.csv",
        low_memory=False,
    )
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string").str.strip()
    comparable = assign_project_balanced_weights(frame.loc[_eligible_mask(frame)].copy())
    return comparable.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _weighted_mae(frame: pd.DataFrame, actual: str, prediction) -> float:
    actual_values = pd.to_numeric(frame[actual], errors="coerce").to_numpy(float)
    prediction_values = np.asarray(prediction, dtype=float)
    weight = pd.to_numeric(frame["sample_weight"], errors="coerce").to_numpy(float)
    mask = np.isfinite(actual_values) & np.isfinite(prediction_values) & np.isfinite(weight)
    return float(np.average(np.abs(actual_values[mask] - prediction_values[mask]), weights=weight[mask]))


def _attach(rows: pd.DataFrame, features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    left = rows.copy().reset_index(drop=True)
    right = features[KEYS + columns].copy()
    right["canonical_project_id"] = right["canonical_project_id"].astype("string").str.strip()
    right["snapshot_date"] = pd.to_datetime(right["snapshot_date"], errors="coerce")
    right = right.sort_values(KEYS, kind="stable").drop_duplicates(KEYS, keep="last")
    merged = left.merge(right, on=KEYS, how="left", sort=False, validate="one_to_one")
    if len(merged) != len(left) or not merged[KEYS].equals(left[KEYS]):
        raise AssertionError("Exp82 feature join changed evaluation keys")
    return merged


def _residual_model(seed: int) -> Pipeline:
    model = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=3,
        num_leaves=12,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=8.0,
        reg_lambda=35.0,
        random_state=seed,
        verbosity=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", model)])


def _weighted_q90(values, weights) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    if not len(values):
        return 0.0
    if weights.sum() <= 0:
        return float(np.quantile(values, 0.90))
    index = np.searchsorted(np.cumsum(weights), 0.90 * weights.sum(), side="left")
    return float(values[min(index, len(values) - 1)])


def _fit_correction(oof: pd.DataFrame, score: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict]:
    work = oof.copy()
    score = score.copy()
    work["residual"] = pd.to_numeric(work["actual_delay_days"], errors="coerce") - pd.to_numeric(
        work["predicted_delay_days"], errors="coerce"
    )
    available = work[features].notna().any(axis=1) & work["residual"].notna()
    covered = work.loc[available].copy()
    if len(covered) < 100 or covered["canonical_project_id"].nunique() < 20:
        return score["predicted_delay_days"].to_numpy(float), {
            "status": "INSUFFICIENT TRAINING SUPPORT",
            "coverage_rows": int(len(covered)),
            "coverage_projects": int(covered["canonical_project_id"].nunique()),
            "selected_scale": 0.0,
        }

    years = sorted(int(value) for value in pd.to_numeric(covered["oof_year"], errors="coerce").dropna().unique())
    validation_year = years[-1]
    fit = covered[pd.to_numeric(covered["oof_year"], errors="coerce").lt(validation_year)].copy()
    validation = covered[pd.to_numeric(covered["oof_year"], errors="coerce").eq(validation_year)].copy()
    selected_scale = 0.0
    scale_metrics = {}

    if len(fit) >= 100 and fit["canonical_project_id"].nunique() >= 20 and len(validation):
        selector = _residual_model(8201)
        selector.fit(
            fit[features],
            fit["residual"],
            model__sample_weight=pd.to_numeric(fit["sample_weight"], errors="coerce").fillna(0).to_numpy(float),
        )
        fit_weight = pd.to_numeric(fit["sample_weight"], errors="coerce").fillna(0).to_numpy(float)
        cap = max(_weighted_q90(np.abs(fit["residual"].to_numpy(float)), fit_weight), 1e-9)
        correction = np.clip(selector.predict(validation[features]), -cap, cap)
        base = validation["predicted_delay_days"].to_numpy(float)
        for scale in (0.0, 0.25, 0.5, 0.75, 1.0):
            prediction = np.maximum(0.0, base + scale * correction)
            scale_metrics[str(scale)] = _weighted_mae(validation, "actual_delay_days", prediction)
        selected_scale = min((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda value: (scale_metrics[str(value)], value))
        if scale_metrics[str(selected_scale)] >= scale_metrics["0.0"] - 1e-12:
            selected_scale = 0.0

    final = _residual_model(8299)
    final.fit(
        covered[features],
        covered["residual"],
        model__sample_weight=pd.to_numeric(covered["sample_weight"], errors="coerce").fillna(0).to_numpy(float),
    )
    weight = pd.to_numeric(covered["sample_weight"], errors="coerce").fillna(0).to_numpy(float)
    cap = max(_weighted_q90(np.abs(covered["residual"].to_numpy(float)), weight), 1e-9)
    correction = np.clip(final.predict(score[features]), -cap, cap) if selected_scale > 0 else np.zeros(len(score))
    prediction = np.maximum(0.0, score["predicted_delay_days"].to_numpy(float) + selected_scale * correction)
    return prediction, {
        "status": "EXECUTION VALID",
        "coverage_rows": int(len(covered)),
        "coverage_projects": int(covered["canonical_project_id"].nunique()),
        "selected_scale": float(selected_scale),
        "correction_cap": float(cap),
        "selection_mae_by_scale": scale_metrics,
        "feature_count": int(len(features)),
    }


def _bootstrap(frame: pd.DataFrame, challenger: np.ndarray) -> dict:
    work = frame[["canonical_project_id", "sample_weight", "actual_delay_days", "predicted_delay_days"]].copy()
    work["challenger"] = challenger
    records = []
    for project_id, group in work.groupby("canonical_project_id", sort=False):
        weight = group["sample_weight"].to_numpy(float)
        actual = group["actual_delay_days"].to_numpy(float)
        base = group["predicted_delay_days"].to_numpy(float)
        candidate = group["challenger"].to_numpy(float)
        records.append(
            (
                str(project_id),
                float(np.average(np.abs(actual - base), weights=weight)),
                float(np.average(np.abs(actual - candidate), weights=weight)),
            )
        )
    per = pd.DataFrame(records, columns=["project", "baseline", "challenger"])
    improvement = per["baseline"].to_numpy(float) - per["challenger"].to_numpy(float)
    rng = np.random.default_rng(8200)
    draws = rng.integers(0, len(improvement), size=(5000, len(improvement)))
    boot = improvement[draws].mean(axis=1)
    return {
        "samples": 5000,
        "projects": int(len(per)),
        "ci95_absolute_improvement": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "probability_challenger_beats_baseline": float(np.mean(boot > 0)),
        "project_win_rate": float(np.mean(improvement > 0)),
    }


def _feature_table() -> tuple[pd.DataFrame, list[str], list[str]]:
    snapshots = load_monthly_snapshots()
    outcomes = pd.read_csv(OUTCOMES, dtype={"project_id": "string"}, low_memory=False)
    resolved, _ = resolve_identities(snapshots, outcomes)
    text_columns = discover_text_columns(snapshots.columns)
    if not text_columns:
        raise ValueError("No tracked PAIMANA status/narrative columns are available for Exp82")
    enriched, _ = add_bottleneck_features(resolved, text_columns=text_columns)
    enriched, generic = add_generic_status_features(enriched, text_columns)
    issue_features = [
        column
        for column in enriched.columns
        if column.startswith("issue_")
        and (
            column.endswith("_active")
            or column.endswith("_seen_before")
            or column in {"issue_count_active", "issue_count_seen", "repeated_bottleneck_flag", "months_since_first_bottleneck"}
        )
    ]
    features = list(dict.fromkeys(issue_features + generic))
    return enriched, features, text_columns


def run(training_end: int, output: Path) -> dict:
    training_start, test_end = 2001, 2025
    data, identity = build_training_dataset()
    feature_table, feature_columns, text_columns = _feature_table()

    years = sorted(
        int(value)
        for value in pd.to_numeric(data["completion_year"], errors="coerce").dropna().unique()
        if training_start < int(value) <= training_end
    )
    fold_years = years[-3:]
    if len(fold_years) < 3:
        raise ValueError("Exp82 requires three rolling production OOF years")

    with tempfile.TemporaryDirectory(prefix="exp82_") as temp:
        root = Path(temp)
        production_root = root / "production"
        train_current_production(
            training_start,
            training_end,
            test_end,
            data=data,
            identity=identity,
            artifact_root=production_root,
        )
        score = _load_validation(production_root, training_start, training_end)

        chunks = []
        for fold_year in fold_years:
            fold_root = root / f"oof_{fold_year}"
            train_current_production(
                training_start,
                fold_year - 1,
                fold_year,
                data=data,
                identity=identity,
                artifact_root=fold_root,
            )
            part = _load_validation(fold_root, training_start, fold_year - 1)
            part["oof_year"] = fold_year
            chunks.append(part)
        oof = pd.concat(chunks, ignore_index=True)

    score = _attach(score, feature_table, feature_columns)
    oof = _attach(oof, feature_table, feature_columns)
    challenger_delay, diagnostics = _fit_correction(oof, score, feature_columns)

    production_cost = _weighted_mae(score, "actual_cost_overrun_percentage", score["predicted_cost_overrun"])
    production_delay = _weighted_mae(score, "actual_delay_days", score["predicted_delay_days"])
    experiment_delay = _weighted_mae(score, "actual_delay_days", challenger_delay)
    delay_gain = (production_delay - experiment_delay) / production_delay * 100.0 if production_delay else 0.0

    result = {
        "experiment_id": "exp_82",
        "experiment_sequence": 82,
        "window": f"2001-{training_end}",
        "status": diagnostics["status"],
        "production_cost_mae": production_cost,
        "experiment_cost_mae": production_cost,
        "cost_improvement_percent": 0.0,
        "production_delay_mae": production_delay,
        "experiment_delay_mae": experiment_delay,
        "delay_improvement_percent": delay_gain,
        "comparison_projects": int(score["canonical_project_id"].nunique()),
        "comparison_snapshots": int(len(score)),
        "actual_text_columns": text_columns,
        "feature_columns": feature_columns,
        "residual_correction": diagnostics,
        "delay_bootstrap": _bootstrap(score, challenger_delay),
        "verdict": "IMPROVED" if delay_gain > 0 else "DO NOT PROMOTE",
        "comparison_policy": "current production anchor plus OOF-selected bounded delay residual correction; Cost unchanged",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-end", type=int, required=True, choices=[2019, 2021])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.training_end, args.output)
