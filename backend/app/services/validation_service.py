from __future__ import annotations

import json
from typing import Any

from backend.app.core.config import MODELS_DIR
from backend.app.services.model_service import metrics
from backend.app.services.prediction_service import project_prediction

BACKTEST_PATH = MODELS_DIR / "backtest.json"


def _best(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    block = payload.get(task, {})
    name = block.get("best_model")
    values = block.get(name, {}) if name else {}
    return {"task": task, "model": name, **values}


def validation_summary() -> dict[str, Any]:
    payload = metrics()
    meta = payload.get("metadata", {})
    return {
        "methodology": {
            "forecasting_rule": "Features are taken only from the as-of snapshot; future outcomes (later revised cost and completion) are labels and are never exposed as features.",
            "single_snapshot_baseline": "When only one snapshot exists, evaluation is explicitly labelled cross-validation baseline rather than future forecasting.",
            "temporal_rule": "When multiple snapshot dates exist, training uses older snapshots and evaluation uses later snapshots.",
        },
        "metadata": meta,
        "best_models": {
            "schedule_classifier": _best("schedule_classifier", payload),
            "cost_classifier": _best("cost_classifier", payload),
            "schedule_regressor": _best("schedule_regressor", payload),
            "cost_regressor": _best("cost_regressor", payload),
        },
        "forward_validation": meta.get("forward_validation", {"available": False, "reason": "Training output predates forward-validation metadata. Retrain models."}),
    }


def model_comparison() -> dict[str, Any]:
    payload = metrics()
    families = {"random_forest", "xgboost", "catboost"}
    tasks: dict[str, list[dict[str, Any]]] = {}
    for task in ("schedule_classifier", "cost_classifier", "schedule_regressor", "cost_regressor"):
        block = payload.get(task, {})
        rows = []
        for name, values in block.items():
            if name not in families or not isinstance(values, dict):
                continue
            rows.append({"model": name, "is_best": name == block.get("best_model"), **values})
        tasks[task] = rows
    return {"tasks": tasks, "note": "Models are compared on the same evaluation split/folds for each task."}


def backtest_payload() -> dict[str, Any]:
    if not BACKTEST_PATH.exists():
        return {
            "available": False,
            "reason": "Backtest artifact has not been generated. Run python scripts/train_models.py.",
            "rows": [],
        }
    return json.loads(BACKTEST_PATH.read_text())


def explain_project(project_code: str) -> dict[str, Any]:
    prediction = project_prediction(project_code, include_explanations=True)
    return {
        "project_code": prediction["project_code"],
        "project_name": prediction["project_name"],
        "scope": prediction["model_scope"],
        "schedule_drivers": prediction.get("schedule_drivers", []),
        "cost_drivers": prediction.get("cost_drivers", []),
        "best_models": prediction.get("best_models", {}),
    }
