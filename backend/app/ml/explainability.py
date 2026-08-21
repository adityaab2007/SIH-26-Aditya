"""Explainability helpers for SIH26103 predictions.

Designed to expose why a project is predicted as high risk.
"""


def build_risk_explanation(features):
    factors = []
    if features.get("progress_delay", 0) > 0:
        factors.append("Current progress is behind planned schedule")
    if features.get("cost_growth", 0) > 0:
        factors.append("Cost escalation trend detected")
    if features.get("historical_delay_rate", 0) > 0.5:
        factors.append("Implementing agency has previous delay history")
    return factors
