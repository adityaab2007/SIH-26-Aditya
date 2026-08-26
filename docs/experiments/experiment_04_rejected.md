# Experiment 4 — Lifecycle-Specific Specialist Models

## Status

REJECTED

Promotion status:

DO NOT PROMOTE

Active challenger:

NO

Production impact:

NONE

## Hypothesis

Experiment 4 tested whether splitting lifecycle snapshots into separate stage-specific specialist models could improve forecasting accuracy compared with the current production model.

Lifecycle stages included variants of:

- early
- early-mid / mid
- late-mid
- late

## Controlled Current-Production Result

### 2001–2019

Production Cost MAE: 29.638 percentage points

Experiment 4 Cost MAE: 30.226 percentage points

Relative cost improvement: -1.9839%

Experiment 4 was approximately 1.98% worse on cost MAE.

Production Delay MAE: 550.670 days

Experiment 4 Delay MAE: 545.649 days

Relative delay improvement: +0.9118%

Experiment 4 produced a small delay improvement but regressed the primary cost objective.

## 2001–2021

No valid final Experiment 4 result was produced from the attempted comparison because the workflow exceeded its execution timeout. Do not interpret this as either improvement or regression.

## Decision

Experiment 4 is rejected. The small delay improvement does not justify the cost regression. The current production model remains unchanged.

Experiment 4 must not be promoted, automatically loaded, selected as active challenger, used as production, or treated as successful because CI completed.

## Important Scientific Note

A successful/green CI execution means the comparison pipeline completed technically. It does not mean the experiment improved model accuracy; scientific promotion decisions must be based on the actual comparison metrics.

## Future Experiments

Future experiments must start from main, use the generic adapter architecture, remain isolated from production, and compare against a freshly evaluated current production baseline on the same comparable test cohort. Validate at minimum 2001–2019 and 2001–2021, preserve dataset/model provenance, record fingerprints and run IDs, and promote only through a separate deliberate production PR.
