# Lifecycle ML experiment workflow

This repository separates the judge-facing production lifecycle model from research experiments and provides one reusable **Retrain & Compare** harness.

## Production boundary

- Production lifecycle models live at `models/monthly_lifecycle/<YYYY_YYYY>/` and fresh retrains are stamped `model_role: production`.
- Experiment artifacts live below `models/monthly_lifecycle/experiments/<experiment_id>/<window>/<run_id>/` and use `model_role: experiment`.
- Prediction Accuracy remains production-only.
- Comparison never auto-promotes or replaces production.

## Generic Retrain & Compare flow

`POST /api/model-simulations/custom/retrain-compare`

For the selected training window the harness:

1. builds/reuses one identity-verified PAIMANA lifecycle dataset;
2. freshly retrains the current production lifecycle stack;
3. loads that exact production run by immutable `run_id`;
4. asks the selected experiment adapter to fit against the same prepared data and production contract;
5. evaluates production and challenger on one comparable future cohort;
6. binds both run IDs into one judge session;
7. offers only projects scoreable by both models;
8. generates both predictions before the actual final outcome is sent to the browser;
9. reveals the official outcome once and compares project-level errors.

The harness itself contains **no experiment implementation**.

## How an experiment becomes the challenger

An experiment PR adds one module named:

`backend/app/ml/experiments/adapter_expXX.py`

The adapter declares:

- `EXPERIMENT_ID`
- `EXPERIMENT_SEQUENCE`
- `EXPERIMENT_NAME`
- `EXPERIMENT_SCOPE`
- `fit_against_production(...)`
- `filter_comparable_rows(...)`
- `predict_project(...)`

`backend/app/ml/experiments/adapters.py` discovers these modules automatically. The registered adapter with the highest `EXPERIMENT_SEQUENCE` is the default challenger shown in Model Simulation. A caller may still request a specific registered experiment ID.

This means the generic harness can be merged first. Afterwards an Experiment 3, Experiment 6, or later PR only needs to add its adapter and experiment-specific implementation. No Model Simulation or comparison-service rewrite is required just to activate the new challenger.

## Stacked PR workflow

Recommended sequence:

1. Merge the generic comparison-harness PR into `main`.
2. Keep the experiment PR based on the harness branch while the harness is under review.
3. After the harness merges, retarget the experiment PR to `main`.
4. Merge the experiment PR only when its own CI/evidence is acceptable.
5. The adapter is then discovered automatically and becomes the active challenger if its sequence number is the highest installed.

Future experiments should follow the same pattern.

## Fair-comparison requirements

Every adapter must preserve the experiment's declared scientific controls. Unless the experiment explicitly changes one of these dimensions, baseline and candidate should share:

- prepared dataset and temporal boundary;
- training/test project identities;
- feature schema;
- project-balanced weighting policy;
- leakage/as-of constraints;
- comparable final user-facing target.

Use `paired_project_mae_comparison(...)` when repeated lifecycle snapshots exist so uncertainty is assessed at project level rather than pretending snapshots are independent.

## Experiment evidence and promotion

`reports/experiments/registry.json` stores production policy. New experiment runs should write immutable evidence under:

`reports/experiments/registry_entries/<experiment_id>/<run_id>.json`

A completed experiment may be `ACCEPTED` or `REJECTED`, but comparison status alone never changes production. `promotion_guard(...)` requires an explicit accepted/promotion decision, and a separate deliberate production-integration PR is still required.

When an experiment is definitively rejected, its active `adapter_expXX.py` should be removed from the comparison harness so it no longer occupies the challenger slot. Its isolated implementation, immutable run evidence, manifests, fingerprints, and historical Git commits may remain for reproducibility. With no installed adapter, Model Simulation correctly reports that no challenger is installed; the next experiment PR can add its own higher-sequence adapter and become the challenger without modifying the generic harness.

## Starting the next experiment

1. Always create the experiment branch from current `main`.
2. Keep the implementation isolated and add `backend/app/ml/experiments/adapter_expXX.py`; do not rewrite the generic comparison infrastructure.
3. Keep production unchanged while developing the challenger.
4. Compare the challenger with current production on the same data and comparable project cohort, at minimum for 2001–2019 and 2001–2021.
5. Record the dataset fingerprint, production run ID, experiment run ID, comparable project count, MAE values, and percent improvement.
6. A green CI run means the pipeline completed; it does not establish scientific acceptance.
7. Promote only through a separate deliberate production PR.

When an experiment is rejected, remove its active adapter, preserve enough evidence to explain the decision, and leave the generic harness ready for the next isolated experiment.

## Heavy experiment execution

Normal PR CI should run syntax/build/tests. Heavy experiments can use `.github/workflows/experiment.yml` with a restricted callable under:

`backend.app.ml.experiments.<module>:<function>`

The manual workflow uploads experiment reports and isolated model artifacts without modifying the production registry.
