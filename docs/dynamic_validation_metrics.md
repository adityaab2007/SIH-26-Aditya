# Dynamic Validation Metrics

Training-window models must always regenerate validation metrics after retraining.

The evaluation contract is:

1. Select training window.
2. Train only on that historical period.
3. Evaluate on the configured future holdout period.
4. Persist metrics and project-level evaluation rows for that model version.

Never reuse a previous validation report after changing the training window.
