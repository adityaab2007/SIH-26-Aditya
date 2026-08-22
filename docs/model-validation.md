# SIH26103 AI Model Validation

## Why validation is required

The system predicts future cost and schedule overruns. To prove correctness, the
model must be tested on historical projects where the final outcome is known.

## Temporal Backtesting Approach

1. Sort project snapshots by date.
2. Train models only on earlier project states.
3. Hide future cost and completion outcomes.
4. Predict the future state.
5. Compare prediction with actual observed outcome.

Example:

Input available at June 2024:
- original cost
- revised cost history
- physical progress
- sector
- ministry
- implementation duration

Hidden during prediction:
- final cost escalation
- final completion date

After prediction, actual values are revealed and error metrics are calculated.

## Metrics

Regression:
- MAE
- RMSE
- R2 score

Classification:
- Accuracy
- F1 score

## Demo Requirement

The dashboard should show:
- predicted overrun
- actual overrun
- prediction error
- model accuracy on historical projects
