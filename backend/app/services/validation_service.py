from __future__ import annotations

import json
import pandas as pd

from backend.app.core.config import MODELS_DIR, PROCESSED_DIR
from backend.app.ml.real_time_windows import active_version


def _version(version: str | None = None) -> str | None:
    return version or active_version()


def validation_report(version: str | None = None) -> dict:
    selected = _version(version)
    path = MODELS_DIR / selected / "evaluation_results.json" if selected else None
    if path and path.exists():
        return json.loads(path.read_text())
    return json.loads((MODELS_DIR / "validation_report.json").read_text())


def validation_rows(version: str | None = None) -> pd.DataFrame:
    selected = _version(version)
    if selected:
        for name in ("prediction_validation.csv", "evaluation_results.csv"):
            path = MODELS_DIR / selected / name
            if path.exists():
                return pd.read_csv(path, dtype={"project_id": str})
    return pd.read_csv(PROCESSED_DIR / "prediction_validation.csv", dtype={"project_id": str})


def validation_payload(limit: int = 100, version: str | None = None) -> dict:
    all_rows = validation_rows(version)
    frame = all_rows.head(max(1, min(limit, 500)))
    safe = frame.astype(object)
    safe = safe.where(~frame.isin([float("inf"), float("-inf")]), None)
    safe = safe.where(pd.notna(safe), None)
    return {"model_version": _version(version), "items": safe.to_dict(orient="records"), "total": int(len(all_rows))}


def rolling_validation_report(version: str | None = None) -> dict:
    selected = _version(version)
    path = MODELS_DIR / selected / "rolling_validation_results.json" if selected else None
    if not path or not path.exists():
        return {"model_version": selected, "folds": [], "fold_count": 0, "status": "not_generated"}
    return json.loads(path.read_text())
