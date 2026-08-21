"""Explain SIH26103 predictions using SHAP when trained models are available."""

from typing import Any


def explain_prediction(model: Any, features: Any) -> dict:
    """Return human readable feature contributions.

    The function safely falls back when SHAP or tree explainers are not
    available, allowing the API to run before production model training.
    """
    try:
        import shap

        explainer = shap.Explainer(model)
        values = explainer(features)
        contributions = []

        for name, value in zip(features.columns, values.values[0]):
            contributions.append({
                "feature": name,
                "impact": float(value)
            })

        contributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return {
            "method": "SHAP",
            "factors": contributions[:5]
        }
    except Exception:
        return {
            "method": "fallback",
            "factors": [],
            "message": "Train a compatible tree model to enable SHAP explanations."
        }
