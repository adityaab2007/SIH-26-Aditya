import json

import pandas as pd

import backend.app.ml.experiments.exp121_actual_rainfall_exposure as exp121
from backend.app.ml.experiments.scientific_challenger_utils import BASE_25_FEATURES


def _rain(end="2018-06-30"):
    dates = pd.date_range("2017-01-31", end, freq="ME")
    return pd.DataFrame({
        "period": dates,
        "geography_type": ["state"] * len(dates),
        "geography_name": ["Delhi"] * len(dates),
        "geography_key": ["delhi"] * len(dates),
        "rainfall_mm": [20.0 + i for i in range(len(dates))],
        "normal_mm": [25.0] * len(dates),
    })


def _row(date="2018-03-15"):
    return pd.DataFrame({
        "canonical_project_id": ["P1"],
        "snapshot_date": pd.to_datetime([date]),
        "state": ["Delhi"],
        "project_name": ["Some Road in Mumbai"],
        "schedule_slippage_days": [10.0],
        "progress_deviation": [-5.0],
        "expenditure_ratio": [0.2],
    })


def test_exp121_freezes_exact_25_feature_base():
    assert len(BASE_25_FEATURES) == 25


def test_future_rainfall_does_not_change_earlier_features():
    before = exp121.build_rainfall_features(_row(), _rain("2018-03-31"))
    future = pd.concat([_rain("2018-03-31"), pd.DataFrame({
        "period": pd.to_datetime(["2025-07-31"]), "geography_type": ["state"],
        "geography_name": ["Delhi"], "geography_key": ["delhi"],
        "rainfall_mm": [9999.0], "normal_mm": [25.0],
    })], ignore_index=True)
    after = exp121.build_rainfall_features(_row(), future)
    cols = [*exp121.FEATURES, "external_data_available", "external_feature_timestamp"]
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_stale_2017_weather_is_not_carried_into_2020_holdout():
    out = exp121.build_rainfall_features(_row("2020-01-15"), _rain("2017-12-31"))
    assert not bool(out.loc[0, "external_data_available"])
    assert pd.isna(out.loc[0, "external_feature_timestamp"])


def test_geography_is_never_guessed_from_project_name():
    row = pd.Series({"project_name": "Highway project in Delhi"})
    assert exp121.resolve_project_geography(row) == ("", "", "unavailable_no_reliable_geography")


def test_explicit_state_beats_project_name_text():
    row = pd.Series({"state": "Karnataka", "project_name": "Project in Delhi"})
    assert exp121.resolve_project_geography(row) == ("state", "karnataka", "exact_normalized_state")


def test_missing_verified_bundle_is_insufficient(tmp_path, monkeypatch):
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({"source_institution": "India Meteorological Department (IMD), Ministry of Earth Sciences, Government of India"}))
    monkeypatch.setattr(exp121, "MANIFEST", manifest)
    monkeypatch.setattr(exp121, "RAINFALL_FILE", tmp_path / "missing.csv")
    data, status = exp121.load_verified_rainfall()
    assert data.empty
    assert status["status"] == "INSUFFICIENT VERIFIED EXTERNAL DATA"
