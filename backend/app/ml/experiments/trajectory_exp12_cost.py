"""Experiment 12 final form: trajectory-enhanced cost forecasting only.

The two-window audit showed a repeatable cost-MAE improvement but no defensible
future delay improvement.  The active Experiment 12 challenger therefore changes
only the cost target.  Production delay remains untouched and is reported only
as retained-production context.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.experiments.evaluator import paired_project_mae_comparison
from backend.app.ml.experiments.framework import (
    build_experiment_context,
    experiment_run_directory,
    new_experiment_manifest,
)
from backend.app.ml.experiments.registry import record_experiment
from backend.app.ml.experiments.trajectory_exp12 import (
    EXPERIMENT_ID,
    MIN_HISTORY,
    _algorithm,
    _key,
    _metric,
    _safe,
)
from backend.app.ml.experiments.trajectory_exp12_v2 import (
    _select_target_features,
    _usable_features,
    enrich_rows,
)
from backend.app.ml.monthly_lifecycle import assign_project_balanced_weights
from backend.app.ml.monthly_training import _fit_pipeline, _regressors, temporal_project_split

EXPERIMENT_NAME = "Trajectory-enhanced cost forecasting"
EXPERIMENT_SCOPE = "cost"
IMPLEMENTATION_REVISION = "v3_cost_only"


def _cost_stage_metrics(rows: pd.DataFrame, prefix: str) -> dict:
    result = {}
    for stage in ("early", "mid", "late", "very_late"):
        part = rows[rows.lifecycle_stage.eq(stage)]
        if part.empty:
            result[stage] = {"available": False}
        else:
            result[stage] = {
                "available": True,
                "cost": _metric(
                    part,
                    "actual_cost_overrun_percentage",
                    f"{prefix}_cost",
                ),
            }
    return result


def _cost_macro(stage: dict) -> float | None:
    values = [item["cost"]["MAE"] for item in stage.values() if item.get("available")]
    return round(float(np.mean(values)), 4) if values else None


def fit_experiment(
    *,
    data,
    training_start,
    training_end,
    test_end,
    production_bundle,
    production_receipt,
    history=None,
):
    """Fit a cost-only Exp 12 challenger against a freshly trained production run."""
    frozen = data.copy()
    frozen["completion_year"] = pd.to_numeric(frozen.completion_year, errors="coerce")
    frozen["snapshot_date"] = pd.to_datetime(frozen.snapshot_date, errors="coerce")
    base_train, base_test = temporal_project_split(
        frozen, training_start, training_end, test_end
    )

    enriched = enrich_rows(frozen, history)
    train, test = temporal_project_split(
        enriched, training_start, training_end, test_end
    )

    production_features = list(
        (production_bundle.get("metadata") or {}).get("features_used")
        or production_receipt.get("features_used")
        or []
    )
    usable, audit = _usable_features(train)
    if not production_features or not usable:
        raise ValueError(
            "Experiment 12 requires production features plus usable trajectory features."
        )

    cost_name = _algorithm(production_bundle, production_receipt, "cost")
    retained_delay_name = _algorithm(production_bundle, production_receipt, "delay")
    cost_added, cost_group, cost_internal = _select_target_features(
        train,
        production_features,
        usable,
        "actual_cost_overrun_percentage",
        cost_name,
        26203,
    )
    cost_features = list(dict.fromkeys(production_features + cost_added))

    cost_model = _fit_pipeline(
        _regressors(26203)[cost_name],
        train,
        cost_features,
        "actual_cost_overrun_percentage",
    )

    compare = test[
        pd.to_numeric(test.exp12_history_12m, errors="coerce")
        .fillna(0)
        .ge(MIN_HISTORY)
    ].copy()
    if compare.canonical_project_id.nunique() < 2:
        raise ValueError(
            "Experiment 12 has too few future projects with usable history."
        )
    compare = assign_project_balanced_weights(compare)

    compare["production_cost"] = production_bundle["cost"].predict(
        compare[production_features]
    )
    compare["experiment_cost"] = cost_model.predict(compare[cost_features])
    # Delay is intentionally NOT changed by Experiment 12.  We compute the
    # production delay MAE only to make the retained target explicit in evidence.
    compare["production_delay"] = np.maximum(
        0, production_bundle["delay"].predict(compare[production_features])
    )

    production_cost = _metric(
        compare, "actual_cost_overrun_percentage", "production_cost"
    )
    experiment_cost = _metric(
        compare, "actual_cost_overrun_percentage", "experiment_cost"
    )
    retained_delay = _metric(compare, "actual_delay_days", "production_delay")
    paired_cost = paired_project_mae_comparison(
        compare,
        actual="actual_cost_overrun_percentage",
        baseline_prediction="production_cost",
        candidate_prediction="experiment_cost",
    )

    production_stage = _cost_stage_metrics(compare, "production")
    experiment_stage = _cost_stage_metrics(compare, "experiment")
    cost_gain = (
        (production_cost["MAE"] - experiment_cost["MAE"])
        / production_cost["MAE"]
        * 100
        if production_cost["MAE"]
        else None
    )

    overall = {
        "production_cost_mae": production_cost["MAE"],
        "experiment_cost_mae": experiment_cost["MAE"],
        "absolute_mae_improvement_pp": round(
            production_cost["MAE"] - experiment_cost["MAE"], 4
        ),
        "improvement_percentage": round(cost_gain, 4)
        if cost_gain is not None
        else None,
        "comparison_test_projects": int(compare.canonical_project_id.nunique()),
        "comparison_test_snapshots": int(len(compare)),
        "paired_project_comparison": paired_cost,
        "paired_project_cost_comparison": paired_cost,
        "production_stage_metrics": production_stage,
        "experiment_stage_metrics": experiment_stage,
        "stage_balanced": {
            "production_cost_mae": _cost_macro(production_stage),
            "experiment_cost_mae": _cost_macro(experiment_stage),
        },
        "internal_feature_selection": {
            "cost": {
                "selected_group": cost_group,
                "comparisons": cost_internal,
            }
        },
        "delay_policy": "production_retained",
        "production_delay_mae": retained_delay["MAE"],
        "retained_production_delay_algorithm": retained_delay_name,
        "delay_experiment_status": "rejected_after_two_window_audit",
    }

    context = build_experiment_context(
        experiment_id=EXPERIMENT_ID,
        full_data=frozen,
        train=base_train,
        test=base_test,
        features=cost_features,
        training_start=training_start,
        training_end=training_end,
        testing_end=test_end,
        weighting_policy="project-balanced quarterly snapshots",
    )
    manifest = new_experiment_manifest(
        context=context,
        name=EXPERIMENT_NAME,
        changed_dimension="feature_set",
        hypothesis=(
            "Scale-aware past-only trajectory features reduce future cost-overrun "
            "MAE while the production delay model remains unchanged."
        ),
    )
    leakage_policy = (
        "Feature construction uses current/earlier project snapshots only; "
        "cost feature-group selection uses only the last two completion years "
        "inside the training window; the future holdout is never consulted for selection."
    )
    manifest.update(
        {
            "scope": EXPERIMENT_SCOPE,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "production_run_id": production_receipt.get("run_id"),
            "production_features": production_features,
            "usable_trajectory_features": usable,
            "cost_added_features": cost_added,
            "feature_availability": audit,
            "selected_algorithms": {
                "cost": cost_name,
                "delay": retained_delay_name,
            },
            "selected_feature_groups": {"cost": cost_group},
            "internal_feature_selection": {"cost": cost_internal},
            "delay_policy": "production_retained",
            "delay_experiment_status": "rejected_after_two_window_audit",
            "comparison_filter": (
                f">={MIN_HISTORY} official observations in trailing 12 months"
            ),
            "leakage_policy": leakage_policy,
        }
    )

    run_dir = experiment_run_directory(
        EXPERIMENT_ID, context.window, manifest["run_id"]
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(cost_model, run_dir / "cost_model.pkl")
    (run_dir / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, allow_nan=False) + "\n"
    )
    (run_dir / "evaluation_results.json").write_text(
        json.dumps(_safe(overall), indent=2, allow_nan=False) + "\n"
    )
    record_experiment(
        {
            "experiment_id": EXPERIMENT_ID,
            "name": EXPERIMENT_NAME,
            "run_id": manifest["run_id"],
            "status": "COMPLETED",
            "decision": "PENDING",
            "model_role": "experiment",
            "promotion_allowed": False,
            "scope": EXPERIMENT_SCOPE,
            "window": context.window,
            "created_at": manifest["created_at"],
            "production_run_id": production_receipt.get("run_id"),
            "cost_improvement_percentage": overall["improvement_percentage"],
            "delay_policy": "production_retained",
            "implementation_revision": IMPLEMENTATION_REVISION,
        }
    )

    lookup_rows = enriched[
        ["canonical_project_id", "snapshot_date", *cost_added]
    ].drop_duplicates(["canonical_project_id", "snapshot_date"], keep="last")
    lookup = {
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat()): {
            feature: row.get(feature) for feature in cost_added
        }
        for _, row in lookup_rows.iterrows()
        if pd.notna(row.canonical_project_id) and pd.notna(row.snapshot_date)
    }
    comparable = {
        (str(row.canonical_project_id), pd.Timestamp(row.snapshot_date).isoformat())
        for _, row in compare.iterrows()
    }

    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_id": manifest["run_id"],
        "model_role": "experiment",
        "scope": EXPERIMENT_SCOPE,
        "decision": "PENDING",
        "promotion_allowed": False,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "feature_count": len(cost_features),
        "production_feature_count": len(production_features),
        "added_feature_count": len(cost_added),
        "added_features": cost_added,
        "cost_added_features": cost_added,
        "selected_feature_groups": {"cost": cost_group},
        "selected_algorithms": {
            "cost": cost_name,
            "delay": retained_delay_name,
        },
        "metrics": {"cost": experiment_cost},
        "delay_policy": "production_retained",
        "delay_experiment_status": "rejected_after_two_window_audit",
        "leakage_policy": leakage_policy,
    }
    return {
        "experiment": experiment,
        "overall_comparison": overall,
        "runtime_state": {
            "cost_model": cost_model,
            "cost_features": cost_features,
            "added": cost_added,
            "lookup": lookup,
            "comparable": comparable,
        },
    }


def filter_comparable_rows(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    return frame[
        frame.apply(lambda row: _key(row) in state["comparable"], axis=1)
    ].copy()


def predict_project(row: pd.Series, state: dict) -> dict:
    key = _key(row)
    if key not in state["lookup"]:
        raise ValueError("No Experiment 12 trajectory is available for this snapshot.")
    candidate = row.copy()
    for name, value in state["lookup"][key].items():
        candidate[name] = value
    cost_X = candidate.to_frame().T.reindex(columns=state["cost_features"])
    return {
        "predicted_cost_overrun": round(
            float(state["cost_model"].predict(cost_X)[0]), 4
        ),
        "trajectory_features_available": int(
            sum(pd.notna(candidate.get(feature)) for feature in state["added"])
        ),
        "trajectory_feature_count": len(state["added"]),
        "delay_policy": "production_retained",
        "implementation_revision": IMPLEMENTATION_REVISION,
    }
