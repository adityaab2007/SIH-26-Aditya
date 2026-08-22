# SIH26103 Scope Mapping

InfraSight is deliberately designed around the statement's move from descriptive monitoring to predictive/prescriptive decision support.

| SIH26103 direction | Implemented prototype |
|---|---|
| Statistical/predictive models | Leakage-safe temporal RF/XGBoost/CatBoost regressors plus retained snapshot baselines |
| Compare AI/ML against conventional methods | Model Performance page with time-based validation/test MAE, RMSE and R2 |
| Cost overrun | Classifier + regression |
| Time overrun | Classifier + regression |
| Risk score | Transparent project-priority score |
| Early warning | Portfolio attention queue |
| Benchmarking | Same-sector peer comparison |
| Driver analysis | Per-forecast SHAP contributions from the selected temporal artifacts |
| AI monitoring dashboard | Dashboard + drilldown |
| Project intelligence assistant | Grounded local analytics assistant |
| Open-source tooling | Python/FastAPI/sklearn/XGBoost/CatBoost/SHAP |

The next full forecasting milestone is archive expansion followed by time-aware T→T+6/T+12 labels and historical forecast backtests.
