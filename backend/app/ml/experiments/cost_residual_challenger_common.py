"""Shared leakage-safe cost residual challenger harness for Exp75-79.

This module is experiment-only. It never writes production artifacts. Each
experiment learns its correction policy only from forward OOF training evidence,
then scores the untouched future cohort against the freshly retrained production
bundle. Delay predictions are copied from production byte-for-byte at the
prediction-value level so Cost-only challengers can still report both metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regression_metrics, _regressors, temporal_project_split
from backend.app.ml.production_cost_baseline import enrich_supervised_for_production, target_feature_contract

TARGET = "actual_cost_overrun_percentage"
DELAY_TARGET = "actual_delay_days"
SEED = 26203


@dataclass
class ChallengerConfig:
    experiment_id: str
    sequence: int
    name: str
    strategy: str


def _finite(value, default=0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _algorithm(bundle: dict, receipt: dict) -> str:
    name = ((bundle.get("metadata") or {}).get("selected_algorithms") or {}).get("cost")
    name = name or (receipt.get("selected_algorithms") or {}).get("cost")
    if name in _regressors(SEED):
        return name
    model = bundle["cost"].named_steps["model"]
    lowered = type(model).__name__.lower()
    if "lgbm" in lowered:
        return "lightgbm"
    if "xgb" in lowered:
        return "xgboost"
    if "extra" in lowered:
        return "extra_trees"
    raise ValueError("Unable to identify production cost estimator family.")


def _metric(frame: pd.DataFrame, actual: str, pred: str) -> dict:
    return _regression_metrics(
        frame[actual], frame[pred].to_numpy(float), frame.sample_weight,
        frame.canonical_project_id,
    )


def _improvement(baseline: float, candidate: float) -> float | None:
    if baseline > 0:
        return (baseline - candidate) / baseline * 100.0
    return 0.0 if candidate == 0 else None


def _verdict(cost_improvement: float | None, delay_improvement: float | None) -> str:
    if cost_improvement is None or delay_improvement is None:
        return "INVALID / INCOMPLETE"
    # Cost-only experiments must not be penalized because Delay is intentionally
    # identical. Positive Cost with unchanged/non-worse Delay is promotable.
    if cost_improvement > 0 and delay_improvement >= -1e-12:
        return "PROMOTION CANDIDATE"
    if cost_improvement <= 0:
        return "REGRESSION / DO NOT PROMOTE"
    return "MIXED / NEEDS REVIEW"


def _project_balanced(frame: pd.DataFrame) -> pd.DataFrame:
    return assign_project_balanced_weights(frame.copy())


def _rolling_oof(
    train: pd.DataFrame,
    features: list[str],
    algorithm: str,
    *,
    max_folds: int = 3,
) -> pd.DataFrame:
    years = sorted(int(y) for y in pd.to_numeric(train.completion_year, errors="coerce").dropna().unique())
    folds: list[pd.DataFrame] = []
    for year in years[-max_folds:]:
        fitting = train[train.completion_year.lt(year)].copy()
        validation = train[train.completion_year.eq(year)].copy()
        if fitting.canonical_project_id.nunique() < 20 or validation.canonical_project_id.nunique() < 5:
            continue
        model = _fit_pipeline(_regressors(SEED)[algorithm], fitting, features, TARGET)
        validation = _project_balanced(validation)
        validation["production_like_cost"] = model.predict(validation[features])
        validation["cost_residual"] = validation[TARGET] - validation["production_like_cost"]
        validation["oof_year"] = year
        folds.append(validation)
    if len(folds) < 2:
        raise ValueError("At least two forward OOF years are required for Cost challenger selection.")
    return pd.concat(folds, ignore_index=True)


def _numeric(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _approved(frame: pd.DataFrame) -> pd.Series:
    for name in ("approved_cost_cr", "original_cost_cr", "sanctioned_cost_cr"):
        if name in frame:
            return _numeric(frame, name)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _revised(frame: pd.DataFrame) -> pd.Series:
    for name in ("revised_cost_cr", "current_cost_cr"):
        if name in frame:
            return _numeric(frame, name)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _spend(frame: pd.DataFrame) -> pd.Series:
    for name in ("cumulative_expenditure_cr", "expenditure_cr", "cumulative_expenditure"):
        if name in frame:
            return _numeric(frame, name)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _stage(frame: pd.DataFrame) -> pd.Series:
    if "lifecycle_stage" not in frame:
        return pd.Series("missing", index=frame.index, dtype="object")
    out = frame["lifecycle_stage"].astype("string").fillna("missing").str.lower()
    return out.astype(str)


def _qedges(values: pd.Series, q: int = 4) -> list[float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.nunique() < 2:
        return [-np.inf, np.inf]
    quantiles = np.linspace(0, 1, q + 1)
    vals = np.unique(np.quantile(finite.to_numpy(float), quantiles))
    if len(vals) < 2:
        return [-np.inf, np.inf]
    vals[0] = -np.inf
    vals[-1] = np.inf
    return [float(x) for x in vals]


def _bins(values: pd.Series, edges: list[float]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=edges, labels=False, include_lowest=True).fillna(-1).astype(int)


def _shrunk_median(values: pd.Series, support: int, prior: float = 0.0, strength: float = 25.0) -> float:
    if support <= 0:
        return float(prior)
    med = float(pd.to_numeric(values, errors="coerce").median())
    if not math.isfinite(med):
        return float(prior)
    weight = support / (support + strength)
    return float(weight * med + (1.0 - weight) * prior)


def _fit_exp75(oof: pd.DataFrame) -> dict:
    approved = _approved(oof).replace(0, np.nan)
    revised = _revised(oof)
    revision_ratio = ((revised - approved).abs() / approved.abs()).replace([np.inf, -np.inf], np.nan)
    availability = revised.notna().astype(float)
    stage_score = _stage(oof).map({"early": 0.0, "mid": 0.35, "late": 0.7, "very_late": 1.0}).fillna(0.2)
    reliability = (0.5 * availability + 0.3 * revision_ratio.clip(0, 1).fillna(0) + 0.2 * stage_score).clip(0, 1)
    edges = _qedges(reliability, 4)
    bucket = _bins(reliability, edges)
    global_med = _finite(oof.cost_residual.median())
    corrections = {}
    for key, group in oof.groupby(bucket):
        corrections[str(int(key))] = _shrunk_median(group.cost_residual, len(group), global_med * 0.2, 35.0)
    cap = max(1.0, _finite(oof.cost_residual.abs().quantile(0.75), 5.0))
    return {"edges": edges, "corrections": corrections, "cap": cap}


def _apply_exp75(frame: pd.DataFrame, state: dict) -> np.ndarray:
    approved = _approved(frame).replace(0, np.nan)
    revised = _revised(frame)
    ratio = ((revised - approved).abs() / approved.abs()).replace([np.inf, -np.inf], np.nan)
    availability = revised.notna().astype(float)
    stage_score = _stage(frame).map({"early": 0.0, "mid": 0.35, "late": 0.7, "very_late": 1.0}).fillna(0.2)
    reliability = (0.5 * availability + 0.3 * ratio.clip(0, 1).fillna(0) + 0.2 * stage_score).clip(0, 1)
    bucket = _bins(reliability, state["edges"])
    corr = bucket.map(lambda x: state["corrections"].get(str(int(x)), 0.0)).to_numpy(float)
    return np.clip(corr, -state["cap"], state["cap"])


def _fit_exp76(oof: pd.DataFrame) -> dict:
    approved = _approved(oof)
    finite = approved.dropna()
    if finite.nunique() >= 4:
        low, high = [float(x) for x in finite.quantile([0.25, 0.75])]
    else:
        low, high = -np.inf, np.inf
    mask = approved.between(low, high, inclusive="both")
    global_med = _finite(oof.loc[mask, "cost_residual"].median()) if mask.any() else 0.0
    by_stage = {}
    for stage, group in oof.loc[mask].assign(_stage=_stage(oof.loc[mask])).groupby("_stage"):
        by_stage[str(stage)] = 0.4 * _shrunk_median(group.cost_residual, len(group), global_med, 30.0)
    return {"low": low, "high": high, "global": 0.4 * global_med, "by_stage": by_stage, "cap": 5.0}


def _apply_exp76(frame: pd.DataFrame, state: dict) -> np.ndarray:
    approved = _approved(frame)
    medium = approved.between(state["low"], state["high"], inclusive="both")
    stages = _stage(frame)
    corr = np.zeros(len(frame), dtype=float)
    for i, (is_medium, stage) in enumerate(zip(medium.to_numpy(), stages.to_numpy())):
        if is_medium:
            corr[i] = state["by_stage"].get(str(stage), state["global"])
    return np.clip(corr, -state["cap"], state["cap"])


def _fit_exp77(oof: pd.DataFrame) -> dict:
    approved = _approved(oof).replace(0, np.nan)
    spend_ratio = (_spend(oof) / approved).replace([np.inf, -np.inf], np.nan)
    stage = _stage(oof)
    earlyish = stage.isin(["early", "mid", "missing"])
    if not earlyish.any():
        earlyish = pd.Series(True, index=oof.index)
    edges = _qedges(spend_ratio.loc[earlyish], 4)
    bucket = _bins(spend_ratio, edges)
    global_med = _finite(oof.loc[earlyish, "cost_residual"].median())
    corrections = {}
    work = oof.loc[earlyish].copy()
    work["_bucket"] = bucket.loc[earlyish]
    work["_stage"] = stage.loc[earlyish]
    for (st, b), group in work.groupby(["_stage", "_bucket"]):
        corrections[f"{st}|{int(b)}"] = 0.5 * _shrunk_median(group.cost_residual, len(group), global_med * 0.25, 30.0)
    return {"edges": edges, "corrections": corrections, "global": 0.25 * global_med, "cap": 6.0}


def _apply_exp77(frame: pd.DataFrame, state: dict) -> np.ndarray:
    approved = _approved(frame).replace(0, np.nan)
    ratio = (_spend(frame) / approved).replace([np.inf, -np.inf], np.nan)
    bucket = _bins(ratio, state["edges"])
    stage = _stage(frame)
    corr = np.zeros(len(frame), dtype=float)
    for i, (st, b) in enumerate(zip(stage.to_numpy(), bucket.to_numpy())):
        if st in ("early", "mid", "missing"):
            corr[i] = state["corrections"].get(f"{st}|{int(b)}", state["global"])
    return np.clip(corr, -state["cap"], state["cap"])


def _fit_exp78(oof: pd.DataFrame) -> dict:
    # Learn correction direction separately in each OOF era and retain only
    # prediction-magnitude regions whose residual sign is temporally consistent.
    edges = _qedges(oof.production_like_cost, 4)
    oof = oof.copy()
    oof["_bucket"] = _bins(oof.production_like_cost, edges)
    corrections = {}
    diagnostics = {}
    for b, group in oof.groupby("_bucket"):
        yearly = group.groupby("oof_year").cost_residual.median().dropna()
        signs = np.sign(yearly.to_numpy(float))
        nonzero = signs[signs != 0]
        consistent = len(nonzero) >= 2 and (np.all(nonzero > 0) or np.all(nonzero < 0))
        correction = 0.35 * _finite(group.cost_residual.median()) if consistent else 0.0
        corrections[str(int(b))] = correction
        diagnostics[str(int(b))] = {"yearly_median_residual": {str(int(k)): float(v) for k, v in yearly.items()}, "consistent": bool(consistent)}
    return {"edges": edges, "corrections": corrections, "diagnostics": diagnostics, "cap": 5.0}


def _apply_exp78(frame: pd.DataFrame, state: dict, production_prediction: np.ndarray) -> np.ndarray:
    bucket = _bins(pd.Series(production_prediction, index=frame.index), state["edges"])
    corr = bucket.map(lambda x: state["corrections"].get(str(int(x)), 0.0)).to_numpy(float)
    return np.clip(corr, -state["cap"], state["cap"])


def _fit_exp79(train: pd.DataFrame, features: list[str], algorithm: str) -> dict:
    years = sorted(int(y) for y in pd.to_numeric(train.completion_year, errors="coerce").dropna().unique())
    validation_year = years[-1]
    fitting = train[train.completion_year.lt(validation_year)].copy()
    validation = _project_balanced(train[train.completion_year.eq(validation_year)].copy())
    if fitting.canonical_project_id.nunique() < 20 or validation.canonical_project_id.nunique() < 5:
        raise ValueError("Insufficient forward validation for uncertainty shrinkage.")
    models = []
    preds = []
    for offset in range(5):
        seed = SEED + offset * 97
        model = _fit_pipeline(_regressors(seed)[algorithm], fitting, features, TARGET)
        models.append(model)
        preds.append(model.predict(validation[features]))
    matrix = np.vstack(preds)
    center = matrix.mean(axis=0)
    uncertainty = matrix.std(axis=0)
    threshold = float(np.quantile(uncertainty, 0.75))
    target_median = float(pd.to_numeric(fitting[TARGET], errors="coerce").median())
    best_alpha, best_mae = 0.0, np.inf
    for alpha in (0.0, 0.05, 0.10, 0.15, 0.20):
        pred = center.copy()
        high = uncertainty >= threshold
        pred[high] = (1 - alpha) * pred[high] + alpha * target_median
        metric = _regression_metrics(validation[TARGET], pred, validation.sample_weight, validation.canonical_project_id)["MAE"]
        if metric < best_mae - 1e-12:
            best_alpha, best_mae = alpha, metric
    final_models = [_fit_pipeline(_regressors(SEED + offset * 97)[algorithm], train, features, TARGET) for offset in range(5)]
    return {"models": final_models, "threshold": threshold, "alpha": best_alpha, "target_median": target_median, "validation_mae": best_mae}


def fit_challenger(
    config: ChallengerConfig,
    *,
    data: pd.DataFrame,
    training_start: int,
    training_end: int,
    test_end: int,
    production_bundle: dict,
    production_receipt: dict,
) -> dict:
    frame = enrich_supervised_for_production(data.copy())
    frame["completion_year"] = pd.to_numeric(frame.completion_year, errors="coerce")
    train, test = temporal_project_split(frame, int(training_start), int(training_end), int(test_end))
    test = _project_balanced(test)

    contract = target_feature_contract(production_bundle.get("metadata") or {})
    cost_features = list(contract.get("cost") or production_receipt.get("features_used") or [])
    delay_features = list(contract.get("delay") or production_receipt.get("features_used") or [])
    if not cost_features or not delay_features:
        raise ValueError("Production target feature contract unavailable.")
    algorithm = _algorithm(production_bundle, production_receipt)

    production_cost = production_bundle["cost"].predict(test[cost_features])
    production_delay = np.maximum(0, production_bundle["delay"].predict(test[delay_features]))

    if config.strategy == "uncertainty_shrinkage":
        policy = _fit_exp79(train, cost_features, algorithm)
        disagreement = np.vstack([m.predict(test[cost_features]) for m in policy["models"]]).std(axis=0)
        experiment_cost = np.asarray(production_cost, dtype=float).copy()
        high = disagreement >= policy["threshold"]
        experiment_cost[high] = (1 - policy["alpha"]) * experiment_cost[high] + policy["alpha"] * policy["target_median"]
        serializable_policy = {k: v for k, v in policy.items() if k != "models"}
    else:
        oof = _rolling_oof(train, cost_features, algorithm, max_folds=3)
        if config.strategy == "revised_cost_reliability":
            policy = _fit_exp75(oof)
            correction = _apply_exp75(test, policy)
        elif config.strategy == "medium_project_specialist":
            policy = _fit_exp76(oof)
            correction = _apply_exp76(test, policy)
        elif config.strategy == "early_financial_surrogate":
            policy = _fit_exp77(oof)
            correction = _apply_exp77(test, policy)
        elif config.strategy == "cross_window_consensus":
            policy = _fit_exp78(oof)
            correction = _apply_exp78(test, policy, production_cost)
        else:
            raise ValueError(f"Unknown challenger strategy: {config.strategy}")
        experiment_cost = np.asarray(production_cost, dtype=float) + correction
        serializable_policy = policy

    experiment_delay = np.asarray(production_delay, dtype=float).copy()
    test["production_cost"] = production_cost
    test["experiment_cost"] = experiment_cost
    test["production_delay"] = production_delay
    test["experiment_delay"] = experiment_delay

    prod_cost_metric = _metric(test, TARGET, "production_cost")
    exp_cost_metric = _metric(test, TARGET, "experiment_cost")
    prod_delay_metric = _metric(test, DELAY_TARGET, "production_delay")
    exp_delay_metric = _metric(test, DELAY_TARGET, "experiment_delay")
    cost_imp = _improvement(prod_cost_metric["MAE"], exp_cost_metric["MAE"])
    delay_imp = _improvement(prod_delay_metric["MAE"], exp_delay_metric["MAE"])

    paired_cost = paired_project_mae_comparison(test, actual=TARGET, baseline_prediction="production_cost", candidate_prediction="experiment_cost")
    paired_delay = paired_project_mae_comparison(test, actual=DELAY_TARGET, baseline_prediction="production_delay", candidate_prediction="experiment_delay", seed=26104)

    runtime_state = {
        "strategy": config.strategy,
        "policy": policy,
        "serializable_policy": serializable_policy,
        "cost_features": cost_features,
        "delay_features": delay_features,
        "production_cost_model": production_bundle["cost"],
        "production_delay_model": production_bundle["delay"],
    }
    run_id = f"{config.experiment_id}-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    overall = {
        "production_cost_mae": prod_cost_metric["MAE"],
        "experiment_cost_mae": exp_cost_metric["MAE"],
        "cost_improvement_percentage": round(cost_imp, 6) if cost_imp is not None else None,
        "production_delay_mae": prod_delay_metric["MAE"],
        "experiment_delay_mae": exp_delay_metric["MAE"],
        "delay_improvement_percentage": round(delay_imp, 6) if delay_imp is not None else None,
        "comparison_test_projects": int(test.canonical_project_id.nunique()),
        "comparison_test_snapshots": int(len(test)),
        "paired_project_cost_comparison": paired_cost,
        "paired_project_delay_comparison": paired_delay,
        "cost_only_delay_predictions_identical": bool(np.array_equal(production_delay, experiment_delay)),
        "verdict": _verdict(cost_imp, delay_imp),
    }
    experiment = {
        "experiment_id": config.experiment_id,
        "experiment_name": config.name,
        "scope": "cost",
        "run_id": run_id,
        "model_role": "experiment",
        "promotion_allowed": False,
        "strategy": config.strategy,
        "training_only_policy": serializable_policy,
        "future_holdout_used_for_selection": False,
    }
    return {"experiment": experiment, "overall_comparison": overall, "runtime_state": runtime_state}


def filter_rows(held: pd.DataFrame, runtime_state: dict) -> pd.DataFrame:
    return held.copy()


def predict_row(row: pd.DataFrame, runtime_state: dict) -> dict:
    if not isinstance(row, pd.DataFrame):
        row = pd.DataFrame([row])
    cost_features = runtime_state["cost_features"]
    delay_features = runtime_state["delay_features"]
    prod_cost = runtime_state["production_cost_model"].predict(row[cost_features])
    prod_delay = np.maximum(0, runtime_state["production_delay_model"].predict(row[delay_features]))
    strategy = runtime_state["strategy"]
    policy = runtime_state["policy"]
    if strategy == "revised_cost_reliability":
        correction = _apply_exp75(row, policy)
        candidate_cost = prod_cost + correction
    elif strategy == "medium_project_specialist":
        correction = _apply_exp76(row, policy)
        candidate_cost = prod_cost + correction
    elif strategy == "early_financial_surrogate":
        correction = _apply_exp77(row, policy)
        candidate_cost = prod_cost + correction
    elif strategy == "cross_window_consensus":
        correction = _apply_exp78(row, policy, prod_cost)
        candidate_cost = prod_cost + correction
    elif strategy == "uncertainty_shrinkage":
        disagreement = np.vstack([m.predict(row[cost_features]) for m in policy["models"]]).std(axis=0)
        candidate_cost = np.asarray(prod_cost, dtype=float).copy()
        high = disagreement >= policy["threshold"]
        candidate_cost[high] = (1 - policy["alpha"]) * candidate_cost[high] + policy["alpha"] * policy["target_median"]
    else:
        raise ValueError(strategy)
    return {
        "predicted_cost_overrun": float(candidate_cost[0]),
        "predicted_delay_days": float(prod_delay[0]),
    }
