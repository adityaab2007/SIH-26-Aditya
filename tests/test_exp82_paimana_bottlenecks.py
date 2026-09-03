import pandas as pd

from backend.app.ml.experiments.exp82_paimana_bottlenecks import (
    CATEGORIES,
    add_bottleneck_features,
    category_active,
    discover_text_columns,
    normalize_text,
)


def test_discovery_normalization_and_detection():
    assert discover_text_columns(["project_id", "latest_remarks", "reason_for_delay"]) == [
        "latest_remarks",
        "reason_for_delay",
    ]
    assert normalize_text(" Land-Acquisition!! ") == "land acquisition"
    assert category_active("land acquisition pending", CATEGORIES["land_acquisition"]) == 1
    assert category_active("no land acquisition issue", CATEGORIES["land_acquisition"]) == 0


def test_asof_persistence_and_first_seen():
    frame = pd.DataFrame(
        {
            "canonical_project_id": ["p", "p", "p"],
            "snapshot_date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "remarks": ["", "Land acquisition pending", "Land acquisition still pending"],
        }
    )
    out, columns = add_bottleneck_features(frame, text_columns=["remarks"])

    assert columns == ["remarks"]
    assert out.issue_land_acquisition_active.tolist() == [0, 1, 1]
    assert out.issue_land_acquisition_seen_before.tolist() == [0, 1, 1]
    assert out.issue_land_acquisition_months_active.tolist() == [0, 1, 2]
    assert out.repeated_bottleneck_flag.tolist() == [0, 0, 1]
    assert pd.isna(out.months_since_first_bottleneck.iloc[0])
    assert out.months_since_first_bottleneck.iloc[1] == 0
    assert out.months_since_first_bottleneck.iloc[2] > 0


def test_future_text_does_not_change_earlier_features():
    early = pd.DataFrame(
        {
            "canonical_project_id": ["p"],
            "snapshot_date": ["2020-01-01"],
            "remarks": ["no issue"],
        }
    )
    full = pd.concat(
        [
            early,
            pd.DataFrame(
                {
                    "canonical_project_id": ["p"],
                    "snapshot_date": ["2020-02-01"],
                    "remarks": ["court case pending"],
                }
            ),
        ],
        ignore_index=True,
    )

    early_features, _ = add_bottleneck_features(early, text_columns=["remarks"])
    full_features, _ = add_bottleneck_features(full, text_columns=["remarks"])
    feature_columns = [
        column
        for column in early_features.columns
        if column.startswith("issue_") or "bottleneck" in column
    ]

    # pandas' assertion treats corresponding NaNs as equal, unlike raw dict
    # equality where NaN != NaN. This test is specifically about temporal
    # invariance: adding a future report must not alter the earlier features.
    pd.testing.assert_series_equal(
        early_features.iloc[0][feature_columns],
        full_features.iloc[0][feature_columns],
        check_names=False,
    )
