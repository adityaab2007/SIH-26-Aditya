# Advanced PAIMANA forecasting audit

## Current pipeline

InfraSight trains a versioned model for a user-selected completion-year window, validates it only on later completed projects, and stores models and evaluation rows under `models/{start_year}_{end_year}/`. Expanding-year rolling validation trains each fold only on earlier completion years.

Targets are official completed-project outcomes:

- cost overrun: `(reported completion expenditure - approved cost) / approved cost * 100`;
- delay: `completion date - planned commissioning date` in days;
- risk: the delay-severity class derived from the cleaned delay target.

The current advanced baseline benchmarks CatBoost, LightGBM, XGBoost, Random Forest, a temporal stacking experiment, and a two-stage delay experiment. Selection uses later-year MAE, never fitted-row accuracy.

## Current feature contract

The as-of feature builder supports approved cost, project size, planned date, cost/expenditure behaviour, progress behaviour, duration/slippage, milestone, sector, ministry, agency, state, and complexity fields. Missing outcome fields are never filled with zero.

The official completed-project archive has 800 outcome rows, but only five published project IDs and two published agencies. The three-month progress archive has 4,692 rows, but no safe exact project match to the completed-outcome rows. Consequently:

- static approved-cost, planned-date and sector features are available for training;
- progress, cost-revision and duration-history features remain unavailable for those unmatched rows;
- fuzzy project-name joins are rejected because similarly named road/rail packages can refer to different projects;
- lifecycle features are calculated only when a real as-of snapshot publishes the required values.

## Target quality and validation baseline

The raw archive included 294 invalid cost labels, dominated by malformed values near -100%, and 276 negative delay labels. The cleaned pipeline rejects invalid labels but retains valid positive extremes with robust loss, log-delay transformation, and sample weighting.

For the regenerated 2001-2015 model, 279 projects are used for training and 110 later completed projects for evaluation. The verified baseline is:

- cost MAE: 28.743 percentage points;
- delay MAE: 944.980 days;
- binary risk was previously reported and was not comparable with the required four-level severity task.

After the lifecycle-aware branch was implemented, the same 2001-2015/2016-2021 boundary reports:

- cost MAE: 32.759 percentage points;
- delay MAE: 715.037 days;
- four-level risk: 60.0% accuracy and 0.1875 macro F1.

The delay result improved by 229.943 days, while cost MAE worsened by 4.016 percentage points. The registry keeps these honest results: it does not claim that the desired 10-20 pp cost or 100-300 day delay targets were reached. The sector correction experiment is separately recorded and activated per target only if its nested later-year validation MAE improves.

These results remain above the desired target because real lifecycle histories cannot yet be joined to the completed outcomes. This limitation is surfaced in model metadata and must not be hidden with synthetic values.

## Leakage controls

- final expenditure, completion date and derived targets are forbidden features;
- validation years are strictly later than training years;
- agency and sector historical statistics are fitted from the selected training data only, with leave-one-out values for fitting rows;
- rolling folds rebuild historical priors using each fold's training rows;
- missing progress history remains unavailable rather than being fabricated.
