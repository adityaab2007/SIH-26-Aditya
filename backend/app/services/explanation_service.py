from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
import shap

from backend.app.services.model_service import best_model_info, load_artifact


def _clean_feature(name: str) -> str:
    raw = name.replace("num__", "").replace("cat__", "")
    if raw.startswith("missingindicator_"):
        return "Missingness: " + raw.replace("missingindicator_", "").replace("_", " ").title()
    replacements = {
        "days_to_original_deadline": "Time relative to original deadline",
        "expenditure_to_original_pct": "Expenditure vs original cost",
        "financial_progress_pct": "Financial progress",
        "physical_progress_pct": "Physical progress",
        "cost_escalation_pct": "Observed cost escalation",
        "schedule_extension_days": "Observed schedule extension",
        "original_cost_cr": "Original project cost",
        "revised_cost_cr": "Current revised cost",
        "expenditure_cr": "Cumulative expenditure",
    }
    return replacements.get(raw, raw.replace("_", " ").title())


def _group_name(name: str) -> str:
    if name.startswith("cat__sector_"):
        return "Sector context"
    if name.startswith("cat__ministry_"):
        return "Ministry context"
    return _clean_feature(name)


def _shap_values(model, Xt: np.ndarray) -> np.ndarray:
    """Return SHAP values for one transformed row across supported model families.

    The registry can legitimately select a tree model or a linear baseline. Using a
    tree-only explainer made explanations disappear whenever LogisticRegression won
    a classifier comparison, so select the SHAP explainer from the trained model.
    """
    if hasattr(model, "tree_" ) or hasattr(model, "estimators_") or model.__class__.__module__.startswith(("xgboost", "catboost")):
        values = shap.TreeExplainer(model).shap_values(Xt)
    elif hasattr(model, "coef_"):
        values = shap.LinearExplainer(model, Xt).shap_values(Xt)
    else:
        explanation = shap.Explainer(model, Xt)(Xt)
        values = explanation.values

    arr = np.asarray(values)
    # Some binary classifiers return (samples, features, classes).
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    if arr.ndim == 2:
        arr = arr[0]
    return np.asarray(arr, dtype=float).reshape(-1)


def local_shap(task: str, frame: pd.DataFrame, limit: int = 7) -> list[dict]:
    info = best_model_info(task)
    artifact = load_artifact(info["path"])
    X = frame.reindex(columns=info["features"])

    try:
        if isinstance(artifact, dict):
            pre = artifact["preprocess"]
            model = artifact["model"]
        else:
            pre = artifact.named_steps["preprocess"]
            model = artifact.named_steps["model"]
        Xt = np.asarray(pre.transform(X))
        names = list(pre.get_feature_names_out())
        arr = _shap_values(model, Xt)
        if len(arr) != len(names):
            return []

        grouped = defaultdict(float)
        for name, value in zip(names, arr):
            grouped[_group_name(name)] += float(value)
        pairs = sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)[:limit]
        total = sum(abs(v) for _, v in pairs) or 1.0
        return [
            {
                "feature": n,
                "impact": round(v, 4),
                "share": round(abs(v) / total, 4),
                "direction": "raises risk" if v > 0 else "reduces risk",
            }
            for n, v in pairs
        ]
    except Exception:
        return []
