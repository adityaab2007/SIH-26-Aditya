from __future__ import annotations

import json
from functools import lru_cache
from typing import Any
import joblib
import numpy as np
import pandas as pd
from backend.app.core.config import MODELS_DIR
from backend.app.ml.real_time_windows import active_version

@lru_cache(maxsize=1)
def registry() -> dict[str, Any]: return json.loads((MODELS_DIR / "registry.json").read_text())
@lru_cache(maxsize=1)
def metrics() -> dict[str, Any]: return json.loads((MODELS_DIR / "model_metrics.json").read_text())
def global_importances() -> dict[str, Any]:
    version = active_version()
    if version:
        shap_dir = MODELS_DIR / version / "shap"
        paths = {name: shap_dir / f"{name}_shap_importance.json" for name in ("cost", "delay", "risk")}
        if all(path.exists() for path in paths.values()):
            return {"model_version": version, **{f"{name}_model": json.loads(path.read_text())["features"] for name, path in paths.items()}}
    return json.loads((MODELS_DIR / "global_feature_importance.json").read_text())
@lru_cache(maxsize=32)
def load_artifact(filename: str): return joblib.load(MODELS_DIR / filename)
def best_model_info(task: str) -> dict[str, Any]: return registry()[f"{task}:best"]
def predict(task: str, frame: pd.DataFrame) -> np.ndarray:
    info=best_model_info(task); artifact=load_artifact(info["path"]); X=frame.reindex(columns=info["features"])
    if isinstance(artifact,dict) and "preprocess" in artifact:
        Xt=artifact["preprocess"].transform(X); return np.asarray(artifact["model"].predict(Xt))
    return np.asarray(artifact.predict(X))
def predict_proba(task: str, frame: pd.DataFrame) -> np.ndarray:
    info=best_model_info(task); artifact=load_artifact(info["path"]); X=frame.reindex(columns=info["features"])
    if isinstance(artifact,dict) and "preprocess" in artifact:
        Xt=artifact["preprocess"].transform(X); return np.asarray(artifact["model"].predict_proba(Xt))[:,1]
    return np.asarray(artifact.predict_proba(X))[:,1]
def model_table() -> dict[str, Any]: return metrics()
