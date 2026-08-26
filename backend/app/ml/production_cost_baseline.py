"""Production cost baseline promoted from verified Experiment 12.

Experiment 12 demonstrated repeatable future-project cost-MAE reductions while
its delay variant did not generalize.  Production therefore adopts only the
trajectory-enhanced cost path.  Delay and risk retain the existing lifecycle
feature contract and fitted models.

The promotion deliberately preserves Experiment 12's scientific contract:
algorithm selection happens on the existing production feature set, trajectory
feature-group selection happens only inside the training period, and the final
cost model is then refit on the full training period.  Future holdout outcomes
are never used for feature selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.trajectory_exp12_v2 import (
    _select_target_features,
    _usable_features,
    engineer_history,
    enrich_rows,
)
from backend.app.ml.monthly_lifecycle import TRAJECTORIES
from backend.app.ml.monthly_training import (
    MODEL_ROOT,
    _balanced_stage_summary,
    _fit_pipeline,
    _importance,
    _regression_metrics,
    _regressors,
    _stage_metrics,
    temporal_project_split,
    train_window,
)
from backend.app.ml.provenance import artifact_fingerprints, feature_schema_fingerprint

PROMOTED_EXPERIMENT_ID = "exp_12"
PRODUCTION_COST_BASELINE = "exp12_trajectory_v3_cost_only"
PRODUCTION_COST_SEED = 26203

_FINGERPRINTED_ARTIFACTS = [
    "cost_model.pkl",
    "delay_model.pkl",
    "risk_model.pkl",
    "feature_quality_report.json",
    "shap_importance.json",
    "prediction_validation.csv",
]


def target_feature_contract(metadata: dict) -> dict[str, list[str]]:
    """Return explicit per-target production feature lists.

    Older artifacts predate target-specific feature contracts, so they continue
    to fall back to ``features_used`` for complete backwards compatibility.
    """
    common = list(metadata.get("features_used") or [])
    return {
        "cost": list(metadata.get("cost_features_used") or common),
        "delay": list(metadata.get("delay_features_used") or common),
        "risk": list(metadata.get("risk_features_used") or common),
    }


def enrich_supervised_for_production(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe promoted cost trajectory features to supervised rows."""
    history = None if TRAJECTORIES.exists() else frame
    return enrich_rows(frame, history=history)


def enrich_history_for_production(frame: pd.DataFrame) -> pd.DataFrame:
    """Engineer promoted trajectory features on a full project-history frame."""
    return engineer_history(frame)


def _prediction_rows(
    test: pd.DataFrame,
    *,
    cost_model,
    cost_features: list[str],
    delay_model,
    delay_features: list[str],
    risk_model,
    risk_features: list[str],
) -> tuple[dict, pd.DataFrame]:
    predicted_cost = cost_model.predict(test[cost_features])
    predicted_delay = np.maximum(0, delay_model.predict(test[delay_features]))
    predicted_risk = risk_model.predict(test[risk_features])

    cost_metrics = _regression_metrics(
        test.actual_cost_overrun_percentage,
        predicted_cost,
        test.sample_weight,
        test.canonical_project_id,
    )

    rows = test[
        [
            "canonical_project_id",
            "project_name",
            "snapshot_date",
            "completion_year",
            "lifecycle_stage",
            "actual_cost_overrun_percentage",
            "actual_delay_days",
            "actual_risk",
            "sample_weight",
        ]
    ].copy()
    rows["predicted_cost_overrun"] = predicted_cost
    rows["predicted_delay_days"] = predicted_delay
    rows["predicted_risk"] = predicted_risk
    rows["cost_error"] = rows.predicted_cost_overrun - rows.actual_cost_overrun_percentage
    rows["delay_error"] = rows.predicted_delay_days - rows.actual_delay_days
    return cost_metrics, rows


def train_window_with_promoted_cost(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
) -> dict:
    """Train production with Exp 12 as the cost baseline only.

    The existing lifecycle trainer is run first so delay/risk, model selection,
    audits, ablations and provenance remain unchanged.  We then replace only the
    cost artifact with the verified Exp 12 trajectory path and rewrite the cost
    evaluation/provenance fields to match the promoted artifact.
    """
    result = train_window(
        training_start,
        training_end,
        test_end,
        data=data,
        identity=identity,
        artifact_root=artifact_root,
    )
    if data is None:
        raise ValueError(
            "Promoted production training requires the frozen supervised frame so "
            "cost trajectory selection uses exactly the same evidence as production."
        )

    root = artifact_root or MODEL_ROOT
    target = root / f"{training_start}_{training_end}"
    metadata = dict(result.get("metadata") or {})
    production_features = list(metadata.get("features_used") or [])
    if not production_features:
        raise ValueError("Production metadata did not provide the lifecycle feature contract.")

    enriched = enrich_supervised_for_production(data.copy())
    enriched["completion_year"] = pd.to_numeric(enriched.completion_year, errors="coerce")
    train, test = temporal_project_split(enriched, training_start, training_end, test_end)

    usable, trajectory_audit = _usable_features(train)
    selected = dict(metadata.get("selected_algorithms") or {})
    cost_algorithm = selected.get("cost")
    if cost_algorithm not in _regressors(PRODUCTION_COST_SEED):
        raise ValueError(f"Unsupported production cost algorithm for Exp 12 promotion: {cost_algorithm}")

    cost_added, cost_group, feature_comparisons = _select_target_features(
        train,
        production_features,
        usable,
        "actual_cost_overrun_percentage",
        cost_algorithm,
        PRODUCTION_COST_SEED,
    )
    cost_features = list(dict.fromkeys(production_features + cost_added))
    cost_model = _fit_pipeline(
        _regressors(PRODUCTION_COST_SEED)[cost_algorithm],
        train,
        cost_features,
        "actual_cost_overrun_percentage",
    )

    delay_model = joblib.load(target / "delay_model.pkl")
    risk_model = joblib.load(target / "risk_model.pkl")
    cost_metrics, validation_rows = _prediction_rows(
        test,
        cost_model=cost_model,
        cost_features=cost_features,
        delay_model=delay_model,
        delay_features=production_features,
        risk_model=risk_model,
        risk_features=production_features,
    )

    # Replace only the production cost artifact. Delay and risk are byte-for-byte
    # the models produced by the existing lifecycle trainer.
    joblib.dump(cost_model, target / "cost_model.pkl")
    validation_rows.to_csv(
        target / "prediction_validation.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    importance_path = target / "shap_importance.json"
    importance = json.loads(importance_path.read_text()) if importance_path.exists() else {}
    importance["cost"] = _importance(
        cost_model,
        train.tail(min(500, len(train))),
        cost_features,
    )
    importance_path.write_text(json.dumps(importance, indent=2))

    lifecycle_stages = _stage_metrics(validation_rows)
    balanced_stage = _balanced_stage_summary(lifecycle_stages)

    metadata["cost_features_used"] = cost_features
    metadata["delay_features_used"] = production_features
    metadata["risk_features_used"] = production_features
    metadata["feature_count_by_target"] = {
        "cost": len(cost_features),
        "delay": len(production_features),
        "risk": len(production_features),
    }
    metadata["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    metadata["promoted_from_experiment"] = PROMOTED_EXPERIMENT_ID
    metadata["promotion_scope"] = "cost"
    metadata["cost_trajectory_feature_group"] = cost_group
    metadata["cost_trajectory_features"] = cost_added
    metadata["trajectory_feature_availability"] = trajectory_audit
    metadata["internal_cost_trajectory_feature_comparisons"] = feature_comparisons
    metadata["delay_policy"] = "existing_production_retained"
    metadata["risk_policy"] = "existing_production_retained"
    metadata["leakage_policy"] = (
        "Direct features are same-snapshot values; production cost trajectory features "
        "use only current/earlier snapshots; cost feature-group selection uses only an "
        "internal historical validation block inside the training period; historical "
        "priors require completion_date < snapshot_date; the future holdout is excluded "
        "from all selection."
    )
    metadata.setdefault("lifecycle_metrics", {})["cost"] = cost_metrics
    metadata["lifecycle_stage_metrics"] = lifecycle_stages
    metadata["balanced_stage_summary"] = balanced_stage
    metadata.setdefault("hyperparameters", {})["cost"] = cost_model.named_steps["model"].get_params()

    provenance = dict(metadata.get("provenance") or {})
    provenance["feature_schema_fingerprint"] = feature_schema_fingerprint(
        list(dict.fromkeys(cost_features + production_features))
    )
    provenance["artifact_fingerprints"] = artifact_fingerprints(
        target,
        _FINGERPRINTED_ARTIFACTS,
    )
    metadata["provenance"] = provenance

    result["metadata"] = metadata
    lifecycle = dict(result.get("lifecycle") or {})
    lifecycle.setdefault("metrics", {})["cost"] = cost_metrics
    lifecycle["features"] = production_features
    lifecycle["target_features"] = {
        "cost": cost_features,
        "delay": production_features,
        "risk": production_features,
    }
    lifecycle["lifecycle_stages"] = lifecycle_stages
    lifecycle["balanced_stage_summary"] = balanced_stage
    lifecycle["production_cost_baseline"] = PRODUCTION_COST_BASELINE
    lifecycle["cost_trajectory_feature_group"] = cost_group
    lifecycle["cost_trajectory_features"] = cost_added
    result["lifecycle"] = lifecycle
    result["promotion"] = {
        "experiment_id": PROMOTED_EXPERIMENT_ID,
        "scope": "cost",
        "production_cost_baseline": PRODUCTION_COST_BASELINE,
        "selected_feature_group": cost_group,
        "added_features": cost_added,
        "delay_retained": True,
        "risk_retained": True,
    }

    metadata_path = target / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    evaluation_path = target / "evaluation_results.json"
    evaluation_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
