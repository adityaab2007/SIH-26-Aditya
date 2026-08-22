# SIH26103 model validation and leakage policy

Infrastructure monitoring is a forecasting problem. A valid backtest must reproduce what an officer would have known **at the prediction date** and keep later outcomes hidden until evaluation.

## What is hidden from the model

For a project snapshot at time `t`, feature engineering may use only values available at `t` (for example original cost, expenditure to date, physical progress, sector/ministry and time remaining to the original deadline). Future-only fields such as the later revised completion date, later revised cost, future schedule shift and future cost jump are labels, never features.

`backend/app/ml/temporal_validation.py` enforces this with `assert_no_target_leakage()` and a list of future-only columns.

## Temporal backtesting

When the dataset contains multiple monthly snapshot dates, `chronological_holdout()` creates a strict split:

- **train:** older snapshots only;
- **test:** later snapshots only;
- the latest training date must be earlier than the earliest test date.

The training pipeline records `validation_method=temporal_holdout`, plus `train_end` and `test_start`, in `models/metrics.json`.

### Current public-data limitation

The reproducible project-wide training table in this repository is a curated **May 2026 PAIMANA snapshot**. One snapshot cannot honestly produce a future temporal split. Therefore the current observed-overrun models are evaluated with cross-validation and are explicitly labelled `*_single_snapshot_baseline`.

The repository also rebuilds a smaller official monthly replay sample in `paimana_high_value_history.csv`. `forward_archive_status()` verifies that future-horizon label generation works, but it refuses to report this small replay set as statistically credible forecasting accuracy. The full longitudinal OCMS/PAIMANA archive remains the dataset required for final forward-model training.

## Metrics

Regression tasks report:

- MAE;
- RMSE;
- R².

Classification tasks report:

- Accuracy;
- F1;
- ROC-AUC.

All best-model selections are made using out-of-sample evaluation: ROC-AUC for classifiers and MAE for regressors.

## Backtest artifact

`python scripts/train_models.py` writes `models/backtest.json`. For each selected model it contains the out-of-sample rows used for evaluation:

- project code/name;
- snapshot date;
- hidden actual outcome;
- model prediction;
- absolute error;
- probability for classifiers.

The Model Validation page uses this artifact for prediction-vs-actual charts, error distributions and a sample hidden-outcome demonstration.

## Explainability

SHAP is already integrated through `backend/app/services/explanation_service.py`. The service loads the selected trained artifact, applies the same preprocessing pipeline used during training, and returns grouped feature contributions with direction (`raises risk` / `reduces risk`).

The validation API exposes this at:

- `GET /api/models/explain/{project_code}`

## Validation APIs

- `GET /api/models/validation` — methodology, best-model metrics and forward-archive status
- `GET /api/models/backtest` — out-of-sample prediction rows
- `GET /api/models/comparison` — XGBoost vs Random Forest vs CatBoost
- `GET /api/models/explain/{project_code}` — SHAP explanation for a real project

## Judge-facing interpretation

The site deliberately separates two claims:

1. **What is proven now:** real PAIMANA data ingestion, observed-overrun baseline models, out-of-sample baseline metrics, SHAP explanations and end-to-end prediction APIs.
2. **What becomes a genuine future forecast:** models trained and tested on the expanded monthly archive with strict older-snapshot → newer-snapshot temporal holdout.

This prevents inflated accuracy claims and makes the path from current prototype to production forecasting auditable.
