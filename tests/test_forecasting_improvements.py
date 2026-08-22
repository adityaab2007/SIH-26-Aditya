import json

import pandas as pd

from backend.app.ml.real_time_windows import (
    FEATURES,
    TARGET_COLUMNS,
    features,
    historical_prior_maps,
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
        "cost_acceleration", "progress_trend_6m", "progress_trend_12m",
        "agency_historical_delay_rate", "agency_historical_cost_overrun_rate", "sector_risk_score",
    }.issubset(FEATURES)


def test_historical_priors_are_fit_only_on_training_outcomes():
    clean = labelled(outcome_data())
    training = clean[clean.completion_year <= 2015].copy()
    future = clean[clean.completion_year > 2015].copy()
    before = historical_prior_maps(training)
    future["actual_delay_days"] = 9_999_999
    future["actual_cost_overrun_percentage"] = 9_999_999
    after = historical_prior_maps(training)
    assert before == after
    assert before["training_end"] <= 2015
    assert all(value < 9_999_999 for value in before["agency_delay"].values())


def test_future_agency_does_not_enter_training_agency_statistics():
    clean = labelled(outcome_data())
    training = clean[clean.completion_year <= 2015].copy()
    future = clean[clean.completion_year > 2015].copy()
    future.loc[:, "implementing_agency"] = "Future-only agency"
    priors = historical_prior_maps(training)
    assert "Future-only agency" not in priors["agency_delay"]
    assert "Future-only agency" not in priors["agency_cost"]
    assert priors["training_end"] == int(training.completion_year.max())


def test_missing_lifecycle_history_stays_missing_instead_of_zero():
    row = pd.DataFrame([{
        "project_id": "one-snapshot", "approved_cost_cr": 100,
        "planned_commissioning_date": "2028-01-01", "snapshot_date": "2026-01-01",
        "physical_progress": 20, "sector": "Roads", "ministry": "Roads",
        "implementing_agency": "Agency", "state": "State",
    }])
    engineered = features(row).iloc[0]
    assert pd.isna(engineered.progress_trend_6m)
    assert pd.isna(engineered.progress_trend_12m)
    assert pd.isna(engineered.progress_acceleration)
    assert pd.isna(engineered.cost_acceleration)


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
    for target_name in ("cost", "delay", "risk"):
        payload = json.loads((model_dir("2001_2015") / "shap" / f"{target_name}_shap_importance.json").read_text())
        assert payload["features"]
        assert all(item["feature"] in FEATURES for item in payload["features"])
