"""Model comparison helpers for SIH26103.

Provides a consistent response format for comparing regression/classification
models such as XGBoost, Random Forest and CatBoost.
"""


def compare_models(results: dict) -> list:
    ranked = []
    for model, metrics in results.items():
        ranked.append({
            "model": model,
            **metrics
        })
    return sorted(ranked, key=lambda item: item.get("score", 0), reverse=True)
