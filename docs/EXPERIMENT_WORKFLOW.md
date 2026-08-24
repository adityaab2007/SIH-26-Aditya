# Lifecycle ML experiment workflow

This repository separates the judge-facing production lifecycle model from research experiments.

## Non-negotiable production boundary

- Production lifecycle models live at `models/monthly_lifecycle/<YYYY_YYYY>/` and are stamped `model_role: production` on fresh retrains.
- Experiment artifacts live below `models/monthly_lifecycle/experiments/<experiment_id>/<window>/<run_id>/` and must use `model_role: experiment`.
- `backend/app/services/lifecycle_run_service.py` excludes anything explicitly marked `experiment` from the production run registry.
- Prediction Accuracy and Model Simulation consume the production lifecycle registry/retrain flow; an experiment is not promoted merely because its code or artifacts exist.

## Before starting a new experiment

1. Pick an unused experiment ID/name with the team. Do not reuse another person's active experiment number.
2. Put the implementation under `backend/app/ml/experiments/`.
3. Declare exactly one primary changed dimension when possible: target, feature set, algorithm, loss, model routing, sampling, or weighting.
4. Build the candidate and baseline from the same explicit `ExperimentContext` unless the changed dimension itself is the cohort/sampling policy.
5. Preserve temporal/project-group leakage rules and as-of feature evidence.
6. Save candidate artifacts only under the experiment run directory returned by `experiment_run_directory(...)`.
7. Use a new immutable `run_id` for every execution.

## Required evidence for a fair comparison

Every experiment report should include:

- `model_role: experiment`
- experiment ID, name, hypothesis, and changed dimension
- training/test periods
- dataset fingerprint
- training cohort fingerprint
- test cohort fingerprint
- feature schema fingerprint
- weighting policy
- baseline and candidate metrics on the same final user-facing target
- project counts and snapshot counts
- run ID and source commit
- decision (`PENDING`, `ACCEPTED`, or `REJECTED`)
- `promotion_allowed: false` while it is only a research result

Use `paired_project_mae_comparison(...)` for cost/delay candidate comparisons where repeated lifecycle snapshots exist. It bootstraps whole projects instead of pretending every snapshot is statistically independent.

## Registry and collaboration

`reports/experiments/registry.json` contains policy plus historical verified decisions. New runs should use `record_experiment(...)`, which writes one immutable file under:

`reports/experiments/registry_entries/<experiment_id>/<run_id>.json`

This prevents Experiment A and Experiment B branches from repeatedly editing the same registry file and reduces merge conflicts.

The existing `reports/experiments/model_evolution.json` is legacy history only. Its older MAEs used a different baseline/model path and are not a valid denominator for current monthly-lifecycle experiment percentages.

## Running a heavy experiment

Normal PR CI runs syntax/build/tests only. Heavy ML experiments are manual using `.github/workflows/experiment.yml`.

The manual workflow accepts a callable in this restricted namespace:

`backend.app.ml.experiments.<module>:<function>`

The callable should accept `training_start`, `training_end`, and `test_end` (or a subset of those names). The workflow uploads `reports/experiments/**` and `models/monthly_lifecycle/experiments/**` as GitHub Actions artifacts.

## Decision and promotion

Training does not imply promotion.

Recommended sequence:

`NOT RUN -> run candidate -> COMPLETED/PENDING -> review same-cohort evidence -> ACCEPTED or REJECTED`

A rejected experiment stays in the repository/registry as evidence and production remains unchanged.

An accepted experiment still must not overwrite production automatically. The shared `promotion_guard(...)` requires both `decision: ACCEPTED` and `promotion_allowed: true`. A separate, deliberate production integration PR should then implement the winning method in the production trainer and rerun the full provenance-safe validation pipeline.

## Current Experiment 3 decision

The verified PR #23 CI comparison for 2001-2015 training and 2016-2025 holdout found:

- direct final-overrun MAE: 42.413 pp
- residual/reconstructed final-overrun MAE: 55.475 pp
- relative change: -30.797%
- decision: **REJECTED**

Experiment 3 therefore remains research evidence and is not the model used by Model Simulation or Prediction Accuracy.
