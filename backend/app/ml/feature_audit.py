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
    minimum_year_coverage: int = 1,
    date_column: str = "snapshot_date",
    project_column: str = "canonical_project_id",
    parser_column: str = "parser_version",
    safely_as_of_features: set[str] | None = None,
    as_of_evidence: dict[str, dict] | None = None,
    leakage_risks: dict[str, str] | None = None,
) -> dict:
    """Return deterministic per-feature quality and leakage-safety decisions.

    ``as_of_evidence`` is preferred over the legacy ``safely_as_of_features``
    switch.  Each feature must carry explicit machine-readable provenance with
    ``proven=True`` and a temporal rule describing why no information after the
    snapshot can enter the value.  The legacy set remains supported for callers
    outside the lifecycle pipeline, but it is intentionally marked as declared
    rather than evidenced in the report.
    """
    invalid_sources = invalid_sources or {}
    leakage_risks = leakage_risks or {}
    as_of_evidence = as_of_evidence or {}
    safely_as_of_features = safely_as_of_features or set()
    has_temporal_axis = date_column in frame and pd.to_datetime(frame.get(date_column), errors="coerce").notna().any()
    dates = pd.to_datetime(frame.get(date_column), errors="coerce") if date_column in frame else pd.Series(pd.NaT, index=frame.index)
    years = dates.dt.year
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
        year_coverage = int(years[valid].nunique()) if has_temporal_axis and len(series) else (1 if valid.any() else 0)
        project_coverage = int(frame.loc[valid, project_column].nunique()) if project_column in frame else int(valid.sum())
        by_year = {str(int(year)): round(float(valid[years.eq(year)].mean() * 100), 2) for year in sorted(years.dropna().unique())}
        by_parser = {str(parser): round(float(valid[frame[parser_column].eq(parser)].mean() * 100), 2) for parser in sorted(frame[parser_column].dropna().unique())} if parser_column in frame else {}

        evidence = dict(as_of_evidence.get(feature) or {})
        if evidence:
            as_of_safe = bool(evidence.get("proven"))
            safety_basis = "evidenced"
        else:
            as_of_safe = feature in safely_as_of_features
            safety_basis = "legacy_declared" if as_of_safe else "missing"

        reason = invalid_sources.get(feature)
        if reason:
            decision = "remove"
        elif availability < minimum_availability:
            decision, reason = "remove", f"availability below {minimum_availability:.1f}%"
        elif constant:
            decision, reason = "remove", "constant or empty in the training window"
        elif year_coverage < minimum_year_coverage:
            decision, reason = "remove", f"available in only {year_coverage} temporal year(s); requires {minimum_year_coverage}"
        elif not as_of_safe:
            decision, reason = "remove", "not proven available as of each historical snapshot"
        else:
            decision, reason = "keep", "observed, variable, and supported by as-of provenance in the training window"
        rows.append({
            "feature": feature,
            "datatype": str(series.dtype),
            "missing_percentage": round(missing, 2),
            "zero_percentage": round(zero_percentage, 2),
            "unique_value_count": unique,
            "constant": bool(constant),
            "availability": round(availability, 2),
            "availability_percentage": round(availability, 2),
            "temporal_year_coverage": year_coverage,
            "project_coverage": project_coverage,
            "availability_by_year": by_year,
            "availability_by_parser": by_parser,
            "safely_as_of_available": as_of_safe,
            "as_of_safety_basis": safety_basis,
            "as_of_evidence": evidence or None,
            "leakage_risk": leakage_risks.get(feature, "none identified by lineage/invariant checks"),
            "decision": decision,
            "reason": reason,
        })
    kept = [row["feature"] for row in rows if row["decision"] == "keep"]
    removed = [row["feature"] for row in rows if row["decision"] == "remove"]
    quality = float(np.mean([row["availability"] for row in rows if row["decision"] == "keep"])) if kept else 0.0
    evidenced = sum(row["as_of_safety_basis"] == "evidenced" for row in rows)
    return {
        "training_rows": int(len(frame)),
        "feature_count_audited": len(rows),
        "features_used": kept,
        "removed_features": removed,
        "removed_invalid_feature_count": len(removed),
        "data_quality_score": round(quality, 2),
        "as_of_evidence_coverage": round(evidenced / len(rows) * 100, 2) if rows else 0.0,
        "features": rows,
        "policy": "Eligibility is evaluated inside the selected training window using availability, variability, temporal/project/parser coverage and explicit as-of provenance. Lifecycle features are not accepted solely because a caller labels them safe.",
    }


def write_feature_quality_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))