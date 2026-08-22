from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.app.ml.real_time_windows import WINDOWS, model_dir


def _resolve_window(key: str | None) -> str:
    if key and key in WINDOWS:
        return key
    return next(iter(WINDOWS))


def validation_report(model: str | None = None) -> dict:
    key = _resolve_window(model)
    path = model_dir(key) / "evaluation_results.json"
    if not path.exists():
        return {
            "model": key,
            "status": "not_evaluated",
            "message": "Run evaluation for this model window first."
        }
    return {"model": key, **json.loads(path.read_text())}


def validation_rows(model: str | None = None) -> pd.DataFrame:
    key = _resolve_window(model)
    path = model_dir(key) / "evaluation_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"project_id": str})


def validation_payload(model: str | None = None, limit: int = 100) -> dict:
    rows = validation_rows(model)
    frame = rows.head(max(1, min(limit, 500)))
    return {
        "model": _resolve_window(model),
        "items": frame.where(pd.notna(frame), None).to_dict(orient="records"),
        "total": int(len(rows)),
    }
