import json

import joblib
import pandas as pd
import numpy as np

from backend.app.ml.feature_audit import audit_features
from backend.app.ml.real_time_windows import (
    CANDIDATE_FEATURES,
    FEATURES,
    INVALID_LIFECYCLE_SOURCES,
    TARGET_COLUMNS,
    _predict_regressor,
    apply_historical_priors,
    add_leave_one_out_training_priors,
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


def test_feature_audit_removes_unavailable_lifecycle_fields():
    clean = labelled(outcome_data())
    training = add_leave_one_out_training_priors(clean[clean.completion_year.between(2001, 2015)].copy())
    report = audit_features(training, CANDIDATE_FEATURES + list(INVALID_LIFECYCLE_SOURCES), invalid_sources=INVALID_LIFECYCLE_SOURCES)
    assert report["features_used"] == FEATURES
    assert set(INVALID_LIFECYCLE_SOURCES).issubset(report["removed_features"])
    assert all(next(item for item in report["features"] if item["feature"] == name)["decision"] == "remove" for name in INVALID_LIFECYCLE_SOURCES)


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


def test_training_agency_history_uses_strictly_past_completions():
    frame = pd.DataFrame({
        "planned_commissioning_date": pd.to_datetime(["2008-01-01", "2011-01-01", "2015-01-01"]),
        "completion_date": pd.to_datetime(["2009-01-01", "2012-01-01", "2016-01-01"]),
        "completion_year": [2009, 2012, 2016], "implementing_agency": ["A", "A", "A"], "sector": ["Power", "Power", "Power"],
        "actual_delay_days": [365.0, 730.0, 9_999_999.0], "actual_cost_overrun_percentage": [10.0, 30.0, 9_999_999.0], "actual_risk": [2, 3, 3],
    })
    enriched = add_leave_one_out_training_priors(frame)
    assert pd.isna(enriched.iloc[0].agency_average_delay)
    assert enriched.iloc[1].agency_average_delay == 365.0
    assert enriched.iloc[2].agency_average_delay < 1_000


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


def test_reliability_reports_are_validation_safe_and_calibrated():
    target = model_dir("2001_2015")
    feature_report = json.loads((target / "feature_quality_report.json").read_text())
    shap_report = json.loads((target / "shap_validation.json").read_text())
    evaluation = json.loads((target / "evaluation_results.json").read_text())
    assert feature_report["features_used"] == FEATURES
    assert shap_report["targets"]["cost"]["meaningful_expected_factors"]
    assert evaluation["sector_validation"]["policy"].startswith("Computed only from future held-out")
    calibration = evaluation["confidence_calibration"]
    assert calibration["status"] in {"well_calibrated", "calibration_warning"}
    expected_confidence = (calibration["cost"]["coverage_percentage"] + calibration["delay"]["coverage_percentage"]) / 2
    assert calibration["confidence_percentage"] == expected_confidence
    if calibration["status"] == "calibration_warning":
        assert calibration["cost"]["scale"] > 1 or calibration["delay"]["scale"] > 1
    assert 0 <= calibration["holdout_observed"]["cost_interval_coverage_percentage"] <= 100
    assert 0 <= calibration["holdout_observed"]["delay_interval_coverage_percentage"] <= 100


def test_project_inputs_change_predictions_and_same_agency_is_not_deterministic():
    target = model_dir("2001_2015")
    priors = json.loads((target / "historical_priors.json").read_text())
    raw = pd.DataFrame([
        {"approved_cost_cr": 100.0, "sector": "Power", "implementing_agency": "Same agency", "planned_commissioning_date": "2028-01-01"},
        {"approved_cost_cr": 100_000.0, "sector": "Power", "implementing_agency": "Same agency", "planned_commissioning_date": "2028-01-01"},
    ])
    model_inputs = apply_historical_priors(features(raw), priors)[FEATURES]
    model = joblib.load(target / "cost_model.pkl")
    predictions = _predict_regressor(model, model_inputs)
    assert predictions[0] != predictions[1]
    assert np.isfinite(predictions).all()
