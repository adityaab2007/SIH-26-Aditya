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
        Xt = pre.transform(X)
        names = list(pre.get_feature_names_out())
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xt)
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        if arr.ndim == 2:
            arr = arr[0]
        arr = np.asarray(arr).reshape(-1)

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
