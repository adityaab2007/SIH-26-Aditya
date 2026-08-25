# Lifecycle ML experiment workflow

This repository separates the judge-facing production lifecycle model from research experiments while allowing Model Simulation to compare the fresh production run with the latest isolated experiment.

## Non-negotiable production boundary

- Production lifecycle models live at `models/monthly_lifecycle/<YYYY_YYYY>/` and are stamped `model_role: production` on fresh retrains.
- Experiment artifacts live below `models/monthly_lifecycle/experiments/<experiment_id>/<window>/<run_id>/` and must use `model_role: experiment`.
- `backend/app/services/lifecycle_run_service.py` excludes anything explicitly marked `experiment` from the production run registry.
- Prediction Accuracy remains production-only.
- Model Simulation may show production and experiment results side-by-side, but that comparison never promotes or replaces production.
- An experiment is not promoted merely because its code, artifacts, or comparison result exist.

## Live Retrain & Compare flow

Model Simulation uses one controlled backend orchestration call:

`POST /api/model-simulations/custom/retrain-compare`

For the selected training window it:

1. Loads/builds the prepared identity-verified PAIMANA lifecycle dataset once.
2. Freshly retrains the real production lifecycle stack and publishes its normal provenance-safe production artifacts.
3. Loads that exact production run by immutable `run_id`.
4. Fits the latest experiment using the same prepared dataset and the experiment's declared comparison contract.
5. Evaluates production and experiment on one comparable future holdout cohort.
6. Calculates absolute and percentage cost-MAE improvement plus a paired project-bootstrap comparison.
7. Creates one comparison session that binds both the production `run_id` and experiment `run_id`.
8. Offers only future held-out projects that both models can score.
9. Generates both predictions before any final outcome is sent to the browser.
10. Reveals the single official final outcome once, then calculates both project errors and the experiment's individual-project error improvement.

The latest comparison adapter is currently `exp_03` (remaining-overrun forecasting). Experiment 3 changes only cost forecasting. Delay and risk shown in Model Simulation remain the freshly retrained production outputs.

## Experiment 3 under Retrain & Compare

The production side of the Experiment 3 comparison is now the **actual freshly retrained production cost model**, not a second separately fitted direct model.

Experiment 3 reuses the fresh production run's:

- feature contract
- selected cost algorithm family
- prepared PAIMANA dataset
- training/test year boundary

Its target remains:

`remaining_cost_overrun_percentage = actual_cost_overrun_percentage - cost_escalation_percentage`

and its user-facing prediction remains:

`predicted_final_cost_overrun = current_cost_escalation + predicted_remaining_cost_overrun`

Because Experiment 3 requires a current cost-escalation anchor, the live comparison evaluates and offers only rows/projects where that anchor exists. Production itself is still trained normally; it is not weakened or modified to imitate the experiment.

## Before starting a new experiment

1. Pick an unused experiment ID/name with the team. Do not reuse another person's active experiment number.
2. Put the implementation under `backend/app/ml/experiments/`.
3. Declare exactly one primary changed dimension when possible: target, feature set, algorithm, loss, model routing, sampling, or weighting.
4. Build the candidate and baseline from the same explicit `ExperimentContext` unless the changed dimension itself is the cohort/sampling policy.
5. Preserve temporal/project-group leakage rules and as-of feature evidence.
6. Save candidate artifacts only under the experiment run directory returned by `experiment_run_directory(...)`.
7. Use a new immutable `run_id` for every execution.
8. When that experiment becomes the repository's latest experiment, add/update its adapter in `lifecycle_model_comparison_service.py`; do not change the production trainer merely to make comparison possible.

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

For live Model Simulation comparison, distinguish two metric sets clearly:

- the normal production holdout metrics from the full production evaluation contract; and
- the production-vs-experiment same-comparable-cohort metrics used to calculate experiment improvement.

Do not divide an experiment MAE by an unrelated historical production MAE from a different cohort.

## Registry and collaboration

`reports/experiments/registry.json` contains policy plus historical verified decisions. New runs should use `record_experiment(...)`, which writes one immutable file under:

`reports/experiments/registry_entries/<experiment_id>/<run_id>.json`

This prevents Experiment A and Experiment B branches from repeatedly editing the same registry file and reduces merge conflicts.

The existing `reports/experiments/model_evolution.json` is legacy history only. Its older MAEs used a different baseline/model path and are not a valid denominator for current monthly-lifecycle experiment percentages.

## Running a heavy experiment

Normal PR CI runs syntax/build/tests only. Heavy standalone ML experiments are manual using `.github/workflows/experiment.yml`.

The manual workflow accepts a callable in this restricted namespace:

`backend.app.ml.experiments.<module>:<function>`

The callable should accept `training_start`, `training_end`, and `test_end` (or a subset of those names). The workflow uploads `reports/experiments/**` and `models/monthly_lifecycle/experiments/**` as GitHub Actions artifacts.

For judge/demo verification of the latest experiment, use Model Simulation's **Retrain & Compare Production vs Latest Experiment** button instead. That is the path that binds the actual fresh production run and candidate run together for both aggregate and individual-project comparison.

## Decision and promotion

Training or winning a comparison does not imply promotion.

Recommended sequence:

`NOT RUN -> run candidate -> COMPLETED/PENDING -> review same-cohort evidence -> ACCEPTED or REJECTED`

A rejected experiment stays in the repository/registry as evidence and production remains unchanged.

An accepted experiment still must not overwrite production automatically. The shared `promotion_guard(...)` requires both `decision: ACCEPTED` and `promotion_allowed: true`. A separate, deliberate production integration PR should then implement the winning method in the production trainer and rerun the full provenance-safe validation pipeline.

## Historical Experiment 3 evidence

The verified PR #23 CI comparison for 2001-2015 training and 2016-2025 holdout found:

- direct final-overrun MAE: 42.413 pp
- residual/reconstructed final-overrun MAE: 55.475 pp
- relative change: -30.797%
- decision: **REJECTED**

That historical evidence remains preserved. A new Model Simulation Retrain & Compare run produces a new immutable Experiment 3 `run_id` and compares it against the production model freshly retrained in the same button click. Production remains the judge-facing model unless a later explicit promotion PR changes that.