# SIH26103 AI Validation Methodology

## Temporal Backtesting

Infrastructure monitoring is a forecasting problem. The model must only use information available at a prediction date.

Training:
- Earlier project snapshots
- Cost history
- Progress history
- Timeline information

Testing:
- Later historical snapshots
- Final actual cost
- Actual completion date

The final outcomes are hidden during prediction and revealed only for evaluation.

## Metrics

Regression:
- MAE
- RMSE
- R2 score

Classification:
- Accuracy
- F1 score

## Explainability

Each prediction should provide risk factors such as:
- delayed progress
- rising cost trend
- historical implementation delays
- abnormal milestone movement
