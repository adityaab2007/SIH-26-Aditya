from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from backend.app.core.config import MODELS_DIR, PROCESSED_DIR


@lru_cache(maxsize=1)
def validation_report() -> dict:
    return json.loads((MODELS_DIR / "validation_report.json").read_text())


@lru_cache(maxsize=1)
def validation_rows() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "prediction_validation.csv", dtype={"project_id": str})


def validation_payload(limit: int = 100) -> dict:
    frame = validation_rows().head(max(1, min(limit, 500)))
    return {"items": frame.where(pd.notna(frame), None).to_dict(orient="records"), "total": int(len(validation_rows()))}
