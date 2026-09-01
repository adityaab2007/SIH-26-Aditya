import numpy as np
import pandas as pd

from backend.app.ml.experiments.exp122_ministry_budget_pressure import (
    FEATURES,
    build_budget_features,
    normalize_ministry,
)
from backend.app.ml.experiments.scientific_challenger_utils import BASE_25_FEATURES, fit_bounded_residual_correction


def _rows():
    return pd.DataFrame({
        "canonical_project_id": ["P1", "P2"],
        "snapshot_date": pd.to_datetime(["2024-05-31", "2024-05-31"]),
        "ministry": ["Ministry of Road Transport & Highways", "Unknown Ministry"],
        "approved_cost_cr": [1000.0, 500.0],
        "revised_cost_cr": [1100.0, 550.0],
    })


def _mapping():
    return pd.DataFrame({
        "raw_ministry": ["Ministry of Road Transport & Highways"],
        "normalized_ministry": [normalize_ministry("Ministry of Road Transport & Highways")],
        "official_budget_ministry": ["Ministry of Road Transport and Highways"],
        "match_method": ["reviewed_exact_alias"],
        "match_confidence": [1.0],
        "effective_from": pd.to_datetime(["2000-01-01"]),
        "effective_to": pd.to_datetime(["2100-12-31"]),
    })


def _budget(published="2024-02-01"):
    return pd.DataFrame({
        "fiscal_year_start": pd.Series([2024], dtype="Int64"),
        "external_feature_timestamp": pd.to_datetime([published]),
        "official_budget_ministry": ["Ministry of Road Transport and Highways"],
        "capital_be_cr": [10000.0],
        "total_be_cr": [12000.0],
        "capital_budget_yoy": [10.0],
        "total_budget_yoy": [5.0],
        "capital_share": [10000.0/12000.0],
        "capital_budget_volatility_3y": [2.0],
    }).rename(columns={"external_feature_timestamp": "published_date"})


def test_base_contract_has_exactly_25_features():
    assert len(BASE_25_FEATURES) == 25
    assert len(set(BASE_25_FEATURES)) == 25


def test_normalization_is_deterministic_not_fuzzy():
    assert normalize_ministry("Ministry of Road Transport & Highways") == "ministry of road transport highways"
    assert normalize_ministry("Road Ministry") != normalize_ministry("Ministry of Road Transport & Highways")


def test_future_budget_publication_cannot_enter_earlier_snapshot():
    features = build_budget_features(_rows(), _budget("2024-08-01"), _mapping())
    assert not features["external_data_available"].any()
    assert features[FEATURES].isna().all().all()


def test_asof_budget_maps_only_reviewed_ministry_and_does_not_duplicate_rows():
    rows = _rows()
    features = build_budget_features(rows, _budget("2024-02-01"), _mapping())
    assert len(features) == len(rows)
    assert features.loc[0, "external_data_available"]
    assert not features.loc[1, "external_data_available"]
    assert features.loc[0, "external_match_method"] == "reviewed_exact_alias"
    assert features.loc[0, "exp122_approved_to_capital_budget"] == 0.1


def test_missing_external_support_falls_back_to_exact_production_prediction():
    n = 10
    oof = pd.DataFrame({
        "canonical_project_id": [f"P{i}" for i in range(n)],
        "snapshot_date": pd.date_range("2018-01-31", periods=n, freq="ME"),
        "sample_weight": np.ones(n),
        "actual_delay_days": np.arange(n, dtype=float),
        "predicted_delay_days": np.arange(n, dtype=float) + 1.0,
        "oof_year": [2019] * n,
        "external_data_available": [False] * n,
    })
    score = oof.drop(columns=["oof_year"]).copy()
    pred, diag, model = fit_bounded_residual_correction(
        oof, score, features=FEATURES, actual="actual_delay_days",
        production_col="predicted_delay_days", available_col="external_data_available", seed=122,
    )
    assert model is None
    assert diag["selected_scale"] == 0.0
    assert np.array_equal(pred, score["predicted_delay_days"].to_numpy(float))
