# SIH26103 Model Validation Workflow

## Temporal Backtesting

Projects are split chronologically to avoid future data leakage.

Training:
- Older project snapshots
- Historical progress updates

Testing:
- Future project snapshots
- Actual completion outcomes hidden during prediction

## Evaluation

Cost prediction:
- MAE
- RMSE
- R2 score

Schedule prediction:
- Accuracy
- F1 score
- Delay error in days

## Explainability

SHAP values explain why a project receives a high risk score:

- progress deviation
- historical sector performance
- cost revision trend
- agency delay patterns

The dashboard should show prediction, actual outcome and contributing factors together.
