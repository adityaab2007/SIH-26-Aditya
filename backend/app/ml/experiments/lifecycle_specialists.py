"""Experiment 4: genuinely separate lifecycle-stage specialist models.

This experiment is intentionally isolated from the production lifecycle model.
Each stage receives its own independently fitted cost, delay, and risk models;
stage labels are therefore no longer evaluation buckets around one global model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.monthly_lifecycle import (
    BASELINE_FEATURES, CANDIDATE_FEATURES, as_of_feature_evidence,
    build_training_dataset,
)
from backend.app.ml.monthly_training import (
    MODEL_ROOT, _stage_metrics, _train_variant, temporal_project_split,
)
from backend.app.ml.provenance import feature_schema_fingerprint, frame_fingerprint, git_commit_sha, new_run_id

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = MODEL_ROOT / "experiments" / "lifecycle_specialists"
STAGES = ("early", "mid", "late", "very_late")


def train_lifecycle_specialists(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame | None = None,
    identity: pd.DataFrame | None = None,
    artifact_root: Path | None = None,
) -> dict:
    """Fit independent stage models and compare them with one global lifecycle model."""
    if data is None or identity is None:
        data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    train, test = temporal_project_split(data, int(training_start), int(training_end), int(test_end))
    if train.canonical_project_id.nunique() < 10 or test.canonical_project_id.nunique() < 2:
        raise ValueError("Experiment 4 requires at least 10 training projects and 2 future holdout projects.")

    audit = audit_features(
        train,
        CANDIDATE_FEATURES,
        minimum_availability=10,
        minimum_year_coverage=2,
        as_of_evidence=as_of_feature_evidence(CANDIDATE_FEATURES),
    )
    features = list(dict.fromkeys(BASELINE_FEATURES + audit["features_used"]))
    run_id = new_run_id()
    destination = (artifact_root or EXPERIMENT_ROOT) / f"{training_start}_{training_end}"
    destination.mkdir(parents=True, exist_ok=True)

    global_bundle, global_metrics, global_rows = _train_variant(train, test, features, 26400)
    global_stage_metrics = _stage_metrics(global_rows)
    specialists: dict[str, dict] = {}

    for index, stage in enumerate(STAGES):
        stage_train = train[train.lifecycle_stage.eq(stage)].copy()
        stage_test = test[test.lifecycle_stage.eq(stage)].copy()
        train_projects = int(stage_train.canonical_project_id.nunique())
        test_projects = int(stage_test.canonical_project_id.nunique())
        if train_projects < 10 or test_projects < 2:
            specialists[stage] = {
                "available": False,
                "reason": f"insufficient stage cohort: train_projects={train_projects}, test_projects={test_projects}",
                "training_rows": int(len(stage_train)),
                "testing_rows": int(len(stage_test)),
            }
            continue

        bundle, metrics, rows = _train_variant(stage_train, stage_test, features, 26500 + index * 10)
        stage_dir = destination / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        for target_name, model in bundle["models"].items():
            joblib.dump(model, stage_dir / f"{target_name}_model.pkl")
        rows.to_csv(stage_dir / "prediction_validation.csv", index=False, date_format="%Y-%m-%d")
        specialists[stage] = {
            "available": True,
            "training_rows": int(len(stage_train)),
            "training_projects": train_projects,
            "testing_rows": int(len(stage_test)),
            "testing_projects": test_projects,
            "selected_algorithms": {**bundle["selected_algorithms"], "risk": "random_forest"},
            "metrics": metrics,
            "global_model_same_stage_metrics": global_stage_metrics.get(stage),
            "artifact_directory": str(stage_dir.relative_to(destination)),
        }

    report = {
        "experiment": "experiment_4_lifecycle_specialists",
        "implementation": "independent_stage_models",
        "run_id": run_id,
        "training_period": [int(training_start), int(training_end)],
        "testing_period": [int(training_end) + 1, int(test_end)],
        "features_used": features,
        "feature_schema_fingerprint": feature_schema_fingerprint(features),
        "dataset_fingerprint": frame_fingerprint(data),
        "training_fingerprint": frame_fingerprint(train),
        "test_fingerprint": frame_fingerprint(test),
        "source_commit": git_commit_sha(ROOT),
        "global_model": {
            "selected_algorithms": {**global_bundle["selected_algorithms"], "risk": "random_forest"},
            "metrics": global_metrics,
            "stage_metrics": global_stage_metrics,
        },
        "specialists": specialists,
        "interpretation": "Each available lifecycle stage is fitted independently. Compare specialist metrics with global_model_same_stage_metrics to test whether specialization improves early/mid/late forecasting.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "experiment_4_results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
