# Model Methodology

## Current baseline tasks

The included May 2026 snapshot supports observed-state labels:

- schedule overrun: revised completion date > original completion date by 90 days;
- cost overrun: revised cost > original cost by 5%;
- schedule extension regression: revised minus original completion date in days;
- cost escalation regression: `(revised - original) / original * 100`.

These are baseline tasks, not a substitute for the future-horizon target requested by SIH26103.

## Feature engineering

Useful fields are derived rather than relying only on raw rupee values:

- original cost;
- current revised cost where available;
- cumulative expenditure;
- physical progress;
- days relative to original deadline;
- expenditure as % of original cost;
- financial progress against revised/original cost basis;
- observed cost escalation (used for schedule baseline only);
- observed schedule extension (used for cost baseline only);
- sector;
- ministry.

Targets are never included directly as features for their own task.

## Algorithms

Each task is evaluated across four families:

Classification:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. CatBoost

Regression:
1. Linear Regression
2. Random Forest
3. XGBoost
4. CatBoost

Categorical values are imputed/one-hot encoded for a common comparison pipeline. CatBoost is evaluated with its own fold loop because the installed CatBoost/sklearn versions expose incompatible estimator tags; it is not skipped.

## Validation

The current baseline uses cross-validation because there is one principal current snapshot per project. When the full longitudinal archive is available, this must be replaced with **time-aware/grouped validation**. Random snapshot splitting would leak the same project's future/past behavior across folds.

## Explainability

The selected tree classifiers are explained with SHAP. One-hot sector/ministry contributions are aggregated into `Sector context` and `Ministry context` to avoid confusing a user with the contribution of an inactive one-hot category.

SHAP answers: **which features moved this model output?**
It does not answer: **which feature caused the real-world delay?**

## Priority score

The prototype review-priority score is a transparent triage index:

```text
45% schedule risk signal
35% cost risk signal
20% project financial-exposure percentile
```

It is not an official MoSPI score and is labelled accordingly.

## Forward forecasting

`backend/app/ml/forward_labels.py` creates labels from month `T` to `T+h` only when a later official snapshot exists. The intended production targets include:

- future deadline shift > 90 days in next 6 months;
- future revised-cost increase > 5% in next 6 months;
- days of future schedule shift;
- percentage future cost escalation.
