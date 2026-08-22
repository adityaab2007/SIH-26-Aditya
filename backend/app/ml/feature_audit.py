"""Feature-quality auditing for the official PAIMANA forecasting pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PLACEHOLDERS = {"", "not reported", "not published", "unknown", "nan", "none"}


def _available(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.notna() & np.isfinite(pd.to_numeric(series, errors="coerce"))
    normalized = series.astype("string").str.strip().str.lower()
    return series.notna() & ~normalized.isin(PLACEHOLDERS)


def audit_features(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    invalid_sources: dict[str, str] | None = None,
    minimum_availability: float = 5.0,
) -> dict:
    """Return deterministic per-feature quality decisions for one training frame."""
    invalid_sources = invalid_sources or {}
    rows = []
    for feature in feature_names:
        series = frame[feature] if feature in frame else pd.Series(np.nan, index=frame.index)
        valid = _available(series)
        available = series[valid]
        availability = float(valid.mean() * 100) if len(series) else 0.0
        missing = 100.0 - availability
        numeric = pd.to_numeric(available, errors="coerce")
        zero_percentage = float(numeric.eq(0).sum() / len(series) * 100) if len(series) else 0.0
        unique = int(available.nunique(dropna=True))
        constant = unique <= 1
        reason = invalid_sources.get(feature)
        if reason:
            decision = "remove"
        elif availability < minimum_availability:
            decision, reason = "remove", f"availability below {minimum_availability:.1f}%"
        elif constant:
            decision, reason = "remove", "constant or empty in the training window"
        else:
            decision, reason = "keep", "observed and variable in the training window"
        rows.append({
            "feature": feature,
            "datatype": str(series.dtype),
            "missing_percentage": round(missing, 2),
            "zero_percentage": round(zero_percentage, 2),
            "unique_value_count": unique,
            "constant": bool(constant),
            "availability": round(availability, 2),
            "availability_percentage": round(availability, 2),
            "decision": decision,
            "reason": reason,
        })
    kept = [row["feature"] for row in rows if row["decision"] == "keep"]
    removed = [row["feature"] for row in rows if row["decision"] == "remove"]
    quality = float(np.mean([row["availability"] for row in rows if row["decision"] == "keep"])) if kept else 0.0
    return {
        "training_rows": int(len(frame)),
        "feature_count_audited": len(rows),
        "features_used": kept,
        "removed_features": removed,
        "removed_invalid_feature_count": len(removed),
        "data_quality_score": round(quality, 2),
        "features": rows,
        "policy": "Missing, placeholder, constant, synthetic, or source-invalid fields are excluded from model fitting.",
    }


def write_feature_quality_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
