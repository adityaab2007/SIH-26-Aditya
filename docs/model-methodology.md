# Model Methodology

## Temporal forecasting demonstration

`data/project_history.csv` provides a documented synthetic monthly timeline for demonstrating the SIH26103 forecasting workflow. The train script creates a future label only when the snapshot month precedes the project's eventual completion date:

- cost escalation: `(actual_final_cost - original_cost) / original_cost * 100`;
- schedule extension: `actual_completion_date - planned_completion_date`.

Projects are assigned once to a **time-based, project-level** cohort by planned-start year: training through 2023, validation in 2024-25, and test from 2026. This prevents one project's later snapshots from appearing in an earlier evaluation cohort. Features use only the snapshot and preceding observations; sector/agency outcomes are calculated solely from other projects completed before the snapshot month.

The selected XGBoost, Random Forest, and CatBoost regressors are written to `models/cost_model.pkl` and `models/delay_model.pkl`. `models/model_metrics.json` records MAE, RMSE and R2 for every candidate, plus a held-out prediction-versus-actual example.

## Historical prediction verification

`backend/app/ml/backtest.py` creates a project-level historical verification record with a strict temporal holdout. Verification models are fitted only on projects completed through 2024. Projects completed in 2025–26 are excluded from fitting, frozen at least 90 days (or the final quarter of the lifecycle) before actual completion, predicted from that snapshot only, and compared with final cost and completion afterwards.

The generated `data/processed/prediction_validation.csv` contains the frozen prediction date, predicted/actual cost overrun and delay, and signed errors. `models/validation_report.json` records cost/delay MAE, RMSE, R2, average per-project accuracy, and elevated-risk classification metrics. Dashboard confidence is the cutoff feature-completeness percentage; it is not a probability of correctness.

## Legacy current-snapshot baselines

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

## Operational replacement path

The pipeline is intentionally schema-compatible with an authorised PAIMANA/OCMS monthly export. Replace the demonstration history file and rerun training before treating forecasts as operational. The intended production targets include:

- future deadline shift > 90 days in next 6 months;
- future revised-cost increase > 5% in next 6 months;
- days of future schedule shift;
- percentage future cost escalation.
