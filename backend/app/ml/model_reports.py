"""Reliability reports derived only from official temporal holdouts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def cost_target_analysis(frame: pd.DataFrame) -> dict:
    values = frame.actual_cost_overrun_percentage.astype(float)
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "projects": int(len(values)),
        "minimum": round(float(values.min()), 4),
        "maximum": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
        "standard_deviation": round(float(values.std()), 4),
        "skew": round(float(values.skew()), 4),
        "percentiles": {str(int(q * 100)): round(float(values.quantile(q)), 4) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "iqr_outlier_projects": int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()),
        "policy": "Valid extreme outcomes are retained. Robust MAE/Huber/quantile and shifted-log candidates are compared by later-year temporal MAE.",
    }


def sector_validation(rows: pd.DataFrame, model_version: str) -> dict:
    sectors = {}
    for sector, group in rows.groupby("sector", dropna=False):
        sectors[str(sector)] = {
            "projects": int(len(group)),
            "cost_mae": round(float(mean_absolute_error(group.actual_cost_overrun, group.predicted_cost_overrun)), 3),
            "delay_mae": round(float(mean_absolute_error(group.actual_delay_days, group.predicted_delay_days)), 3),
            "validation_period_only": True,
        }
    return {
        "model_version": model_version,
        "sectors": dict(sorted(sectors.items(), key=lambda item: (-item[1]["projects"], item[0]))),
        "policy": "Computed only from future held-out completed projects; training rows are excluded.",
    }


def observed_confidence_calibration(rows: pd.DataFrame, model_version: str, training_calibration: dict) -> dict:
    available = {"cost", "delay"}.issubset(training_calibration)
    cost_covered = rows.actual_cost_overrun.between(rows.predicted_cost_p10, rows.predicted_cost_p90) if "predicted_cost_p10" in rows else pd.Series(False, index=rows.index)
    delay_covered = rows.actual_delay_days.between(rows.predicted_delay_p10, rows.predicted_delay_p90) if "predicted_delay_p10" in rows else pd.Series(False, index=rows.index)
    cost_coverage = float(cost_covered.mean() * 100)
    delay_coverage = float(delay_covered.mean() * 100)
    nominal = float(training_calibration.get("nominal_coverage_percentage", 80.0))
    deviation = max(abs(cost_coverage - nominal), abs(delay_coverage - nominal))
    return {
        **training_calibration,
        "model_version": model_version,
        "holdout_observed": {
            "projects": int(len(rows)),
            "cost_interval_coverage_percentage": round(cost_coverage, 2),
            "delay_interval_coverage_percentage": round(delay_coverage, 2),
            "joint_interval_coverage_percentage": round(float((cost_covered & delay_covered).mean() * 100), 2),
        },
        "status": "well_calibrated" if available and deviation <= 10 else "calibration_warning",
        "display_policy": "Confidence is the earlier-year calibrated interval coverage, not an unsupported per-project probability.",
    }


def shap_validation(feature_importance: dict, model_version: str) -> dict:
    expected = {
        "cost": {"cost_escalation_percentage", "expenditure_ratio", "agency_average_cost_overrun", "duration_ratio", "sector_average_cost_overrun", "approved_cost_cr"},
        "delay": {"schedule_slippage_days", "duration_ratio", "agency_average_delay", "sector_average_delay", "approved_cost_cr"},
        "risk": {"agency_failure_rate", "sector_average_delay", "sector_average_cost_overrun", "approved_cost_cr"},
    }
    targets = {}
    for target, rows in feature_importance.items():
        top = [row["feature"] for row in rows[:10]]
        meaningful = [feature for feature in top if feature in expected.get(target, set())]
        targets[target] = {
            "top_features": top,
            "meaningful_expected_factors": meaningful,
            "status": "validated" if meaningful else "warning_context_only",
        }
    return {"model_version": model_version, "targets": targets, "validated": all(item["status"] == "validated" for item in targets.values())}


def final_comparison(before: dict, after_metrics: dict, feature_report: dict) -> dict:
    return {
        "before": {
            "model": "old 4-feature PAIMANA model",
            "cost_mae": before["current_metrics"]["cost_MAE_percentage_points"],
            "delay_mae": before["current_metrics"]["delay_MAE_days"],
            "risk_f1": before["current_metrics"]["risk_f1"],
            "feature_count": len(before["current_features"]),
            "data_quality_score": None,
        },
        "after": {
            "model": "feature-audited official PAIMANA model",
            "cost_mae": after_metrics["cost_model"]["MAE"],
            "delay_mae": after_metrics["delay_model"]["MAE_days"],
            "risk_f1": after_metrics["risk_model"]["f1"],
            "feature_count": len(feature_report["features_used"]),
            "data_quality_score": feature_report["data_quality_score"],
        },
        "honesty_note": "Metrics are from different task definitions where stated; four-level risk F1 is not directly comparable with the old binary risk F1. Worse metrics are retained.",
    }
