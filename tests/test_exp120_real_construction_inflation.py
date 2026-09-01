import json

import numpy as np
import pandas as pd

import backend.app.ml.experiments.exp120_real_construction_inflation as exp120
from backend.app.ml.experiments.scientific_challenger_utils import BASE_25_FEATURES, assert_same_keys


def _wpi():
    dates = pd.date_range("2017-01-31", periods=18, freq="ME")
    return pd.DataFrame({
        "date": dates,
        "all_commodities": np.linspace(100, 118, len(dates)),
        "manufactured_products": np.linspace(100, 112, len(dates)),
        "fuel_power": np.linspace(100, 125, len(dates)),
        "steel": np.linspace(100, 130, len(dates)),
        "cement": np.linspace(100, 110, len(dates)),
    })


def test_exp120_freezes_exact_25_feature_base():
    assert len(BASE_25_FEATURES) == 25
    assert len(set(BASE_25_FEATURES)) == 25


def test_wpi_after_snapshot_cannot_enter_feature_vector():
    rows = pd.DataFrame({
        "canonical_project_id": ["P1"],
        "snapshot_date": pd.to_datetime(["2018-03-15"]),
        "approved_cost_cr": [1000.0],
    })
    before = exp120.build_wpi_features(rows, _wpi())
    future = pd.concat([_wpi(), pd.DataFrame({
        "date": pd.to_datetime(["2025-01-31"]),
        "all_commodities": [9999.0],
        "manufactured_products": [9999.0],
        "fuel_power": [9999.0],
        "steel": [9999.0],
        "cement": [9999.0],
    })], ignore_index=True)
    after = exp120.build_wpi_features(rows, future)
    pd.testing.assert_frame_equal(
        before[[*exp120.FEATURES, "external_feature_timestamp", "external_data_available"]].reset_index(drop=True),
        after[[*exp120.FEATURES, "external_feature_timestamp", "external_data_available"]].reset_index(drop=True),
    )
    assert after.loc[0, "external_feature_timestamp"] <= rows.loc[0, "snapshot_date"]


def test_missing_verified_bundle_is_insufficient_not_zero_filled(tmp_path, monkeypatch):
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({
        "source_institution": "Office of the Economic Adviser, DPIIT, Government of India"
    }))
    monkeypatch.setattr(exp120, "MANIFEST", manifest)
    monkeypatch.setattr(exp120, "WPI_FILE", tmp_path / "does_not_exist.csv")
    data, status = exp120.load_verified_wpi()
    assert data.empty
    assert status["status"] == "INSUFFICIENT VERIFIED EXTERNAL DATA"


def test_external_join_key_contract_is_exact():
    base = pd.DataFrame({
        "canonical_project_id": ["P1", "P2"],
        "snapshot_date": pd.to_datetime(["2018-03-15", "2018-04-15"]),
    })
    challenger = base.copy()
    assert_same_keys(base, challenger)
