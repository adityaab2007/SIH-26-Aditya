"""Fast audit-only production-vs-Exp12 cost/delay comparison.

This reproduces production's lifecycle feature audit, model selection, seeds,
weights and temporal split, but intentionally skips risk fitting, SHAP,
production ablations and artifact publication. It is for experiment evidence
only; the website continues to use the full Retrain & Compare service.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    as_of_feature_evidence,
    build_training_dataset,
)
from backend.app.ml.monthly_training import _select_regressor, temporal_project_split
from backend.app.ml.experiments.trajectory_exp12 import _safe
from backend.app.ml.experiments.trajectory_exp12_v2 import fit_experiment


def production_cost_delay(data: pd.DataFrame, start: int, end: int, test_end: int) -> tuple[dict, dict]:
    train, _ = temporal_project_split(data, start, end, test_end)
    audit = audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
        leakage_risks={
            "revised_cost_cr": "late-stage signal available in the same official snapshot; evaluated by production ablation",
            "cost_escalation_percentage": "late-stage signal derived from same-snapshot revised cost; evaluated by production ablation",
        },
    )
    features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
    cost_name, cost, cost_cmp = _select_regressor(train, features, "actual_cost_overrun_percentage", 26203)
    delay_name, delay, delay_cmp = _select_regressor(train, features, "actual_delay_days", 26204)
    bundle = {
        "metadata": {
            "features_used": features,
            "selected_algorithms": {"cost": cost_name, "delay": delay_name},
        },
        "cost": cost,
        "delay": delay,
    }
    receipt = {
        "run_id": f"fast-audit-production-{start}-{end}",
        "dataset_fingerprint": "ephemeral-fast-audit",
        "features_used": features,
        "selected_algorithms": {"cost": cost_name, "delay": delay_name},
    }
    receipt["internal_algorithm_comparisons"] = {"cost": cost_cmp, "delay": delay_cmp}
    return bundle, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data, _identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data.completion_year, errors="coerce")
    test_end = int(data.completion_year.dropna().max())
    bundle, receipt = production_cost_delay(data, args.start, args.end, test_end)
    fitted = fit_experiment(
        data=data,
        training_start=args.start,
        training_end=args.end,
        test_end=test_end,
        production_bundle=bundle,
        production_receipt=receipt,
    )
    overall = fitted["overall_comparison"]
    experiment = fitted["experiment"]
    payload = {
        "window": f"{args.start}_{args.end}",
        "test_end": test_end,
        "audit_mode": "exact production cost-delay training; skips risk/SHAP/ablations only",
        "production_selected_algorithms": receipt["selected_algorithms"],
        "production_internal_algorithm_comparisons": receipt["internal_algorithm_comparisons"],
        "production_feature_count": len(receipt["features_used"]),
        "production_features": receipt["features_used"],
        "experiment": experiment,
        "overall_comparison": overall,
    }
    safe_payload = _safe(payload)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_payload, indent=2, allow_nan=False) + "\n")
    summary = {
        "window": payload["window"],
        "test_end": test_end,
        "production_cost_mae": overall.get("production_cost_mae"),
        "experiment_cost_mae": overall.get("experiment_cost_mae"),
        "cost_improvement_percentage": overall.get("improvement_percentage"),
        "production_delay_mae": overall.get("production_delay_mae"),
        "experiment_delay_mae": overall.get("experiment_delay_mae"),
        "delay_improvement_percentage": overall.get("delay_improvement_percentage"),
        "comparison_test_projects": overall.get("comparison_test_projects"),
        "comparison_test_snapshots": overall.get("comparison_test_snapshots"),
        "paired_project_cost_comparison": overall.get("paired_project_cost_comparison"),
        "paired_project_delay_comparison": overall.get("paired_project_delay_comparison"),
        "stage_balanced": overall.get("stage_balanced"),
        "internal_feature_selection": overall.get("internal_feature_selection"),
        "production_selected_algorithms": receipt["selected_algorithms"],
        "experiment_selected_feature_groups": experiment.get("selected_feature_groups"),
        "cost_added_features": experiment.get("cost_added_features"),
        "delay_added_features": experiment.get("delay_added_features"),
    }
    print("EXP12_FAST_COMPARISON=" + json.dumps(_safe(summary), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
