# Experiment 12 — Trajectory-enhanced lifecycle forecasting

## Hypothesis

The production lifecycle model already uses current-state PAIMANA features plus a small number of 3/6-month trajectory features. Experiment 12 tests whether a richer **past-only** representation of how cost, expenditure, and schedule are changing improves future **cost-overrun MAE** and **delay MAE**.

## Changed dimension

Only the **feature representation** changes.

The experiment keeps the production controls fixed:

- same identity-verified supervised quarterly snapshots;
- same training and future-holdout project split;
- same final cost-overrun and delay targets;
- same project-balanced weighting policy;
- same production-selected regressor family for cost and delay;
- same LightGBM/XGBoost/ExtraTrees constructors and deterministic production seeds;
- no risk-model promotion or production artifact replacement.

## Added trajectory signals

The challenger derives additional features from the full monthly trajectory table and joins them back onto the exact supervised snapshot keys:

- trailing-12-month history observation count;
- 12-month revised-cost velocity;
- 12-month cost revision count;
- months since the latest cost revision;
- 6-month cost-change volatility;
- 3/6/12-month expenditure velocity;
- expenditure acceleration;
- 3/6/12-month schedule-slippage velocity;
- slippage acceleration;
- 12-month schedule revision count;
- months since the latest schedule revision;
- 6-month slippage-change volatility.

Features are admitted only when they have at least 10% availability in the selected training window and more than one observed value.

## Leakage rule

Every Experiment 12 feature uses only:

1. the current official PAIMANA snapshot, and
2. strictly earlier snapshots for the same canonical project.

Later reports, actual final expenditure, actual completion date, and derived final targets are never consulted while constructing trajectory features. Unit tests prove that appending an extreme future report cannot change any earlier Experiment 12 feature values.

## Comparable evaluation

The production and challenger predictions are evaluated on the same future snapshots that have at least two official observations in the trailing 12 months. Project-balanced weights are recalculated after this common comparability filter.

Reported evidence includes:

- production vs challenger cost MAE;
- production vs challenger delay MAE;
- absolute and percentage MAE improvement;
- project-level paired bootstrap comparison for cost and delay;
- early/mid/late/very-late lifecycle metrics;
- stage-balanced cost and delay MAE;
- project-level cost and delay errors in Model Simulation after reveal.

## Retrain & Compare integration

`backend/app/ml/experiments/adapter_exp12.py` registers:

- `EXPERIMENT_ID = exp_12`
- `EXPERIMENT_SEQUENCE = 12`
- scope `cost_delay`

The existing generic endpoint automatically discovers it:

`POST /api/model-simulations/custom/retrain-compare`

Model Simulation then freshly retrains production, fits Experiment 12 on the same frozen evidence, offers only jointly scoreable held-out projects, generates both cost and delay predictions before reveal, and compares both errors against the official outcome.

## Artifact isolation

Experiment models and evidence are written only below:

`models/monthly_lifecycle/experiments/exp_12/<window>/<run_id>/`

Registry evidence is written as an immutable run entry. Experiment 12 remains `PENDING` and `promotion_allowed: false`; it never replaces production automatically.
