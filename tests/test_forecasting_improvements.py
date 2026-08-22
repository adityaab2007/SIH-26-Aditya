import json

import pandas as pd

from backend.app.ml.real_time_windows import (
    FEATURES,
    TARGET_COLUMNS,
    label_quality_report,
    labelled,
    model_dir,
    outcome_data,
)
from backend.app.services.prediction_service import project_forecast


def test_final_features_contain_no_target_or_future_outcome_columns():
    assert TARGET_COLUMNS.isdisjoint(FEATURES)
    assert "reported_completion_expenditure_cr" not in FEATURES
    assert "completion_date" not in FEATURES


def test_invalid_cost_and_delay_labels_are_removed_without_zero_fills():
    clean = labelled(outcome_data())
    quality = label_quality_report()
    assert quality["invalid_cost_labels_removed"] > 0
    assert quality["invalid_delay_labels_removed"] > 0
    assert quality["missing_targets_filled_with_zero"] is False
    assert clean.actual_cost_overrun_percentage.between(-90, 1000).all()
    assert clean.actual_delay_days.ge(0).all()
    assert clean[["actual_cost_overrun_percentage", "actual_delay_days"]].notna().all().all()


def test_feature_count_and_requested_feature_families_increased():
    assert len(FEATURES) > 4
    assert {
        "cost_escalation_percentage", "budget_stress_index", "expenditure_ratio",
        "progress_deviation", "progress_velocity", "progress_acceleration",
        "planned_duration_days", "duration_ratio", "schedule_slippage_score",
        "complexity_score", "project_size_category", "ministry", "state",
    }.issubset(FEATURES)


def test_changing_training_window_changes_real_validation_metrics():
    first = json.loads((model_dir("2001_2015") / "evaluation_results.json").read_text())
    second = json.loads((model_dir("2001_2017") / "evaluation_results.json").read_text())
    assert first["metadata"]["training_end"] != second["metadata"]["training_end"]
    assert (first["cost_model"]["MAE"], first["delay_model"]["MAE_days"]) != (second["cost_model"]["MAE"], second["delay_model"]["MAE_days"])


def test_final_model_shap_and_uncertainty_use_registered_features():
    result = project_forecast("701263")
    assert result["shap_explanation"]
    assert all(item["feature"] in FEATURES for item in result["shap_explanation"])
    assert result["expected_range"]
    assert result["expected_range"]["delay_days"]["p10"] <= result["expected_range"]["delay_days"]["p90"]
    assert result["features_used"] == FEATURES
