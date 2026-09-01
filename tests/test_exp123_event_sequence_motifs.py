import pandas as pd

import backend.app.ml.experiments.exp123_event_sequence_motifs as exp123
from backend.app.ml import production_exp35_baseline as exp35_production
from backend.app.ml.experiments.exp123_event_sequence_motifs import build_sequence_features, learn_event_thresholds
from backend.app.ml.experiments.scientific_challenger_utils import BASE_25_FEATURES


def _history():
    return pd.DataFrame({
        "canonical_project_id": ["P1", "P1", "P1"],
        "snapshot_date": pd.to_datetime(["2018-01-01", "2018-02-01", "2018-03-01"]),
        "completion_year": [2019, 2019, 2019],
        "cost_escalation_percentage": [0.0, 2.0, 2.0],
        "revised_cost_cr": [100.0, 102.0, 102.0],
        "schedule_slippage_days": [0.0, 15.0, 30.0],
        "physical_progress": [10.0, 10.0, 20.0],
        "expenditure_ratio": [0.10, 0.11, 0.12],
    })


def test_exp123_freezes_exact_25_feature_base():
    assert len(BASE_25_FEATURES) == 25


def test_future_report_cannot_change_earlier_sequence_features():
    base = _history()
    thresholds = learn_event_thresholds(base, 2019)
    before = build_sequence_features(base, thresholds).copy()
    future = pd.concat([base, pd.DataFrame({
        "canonical_project_id": ["P1"], "snapshot_date": pd.to_datetime(["2020-01-01"]),
        "completion_year": [2020], "cost_escalation_percentage": [99.0], "revised_cost_cr": [999.0],
        "schedule_slippage_days": [999.0], "physical_progress": [100.0], "expenditure_ratio": [1.0],
    })], ignore_index=True)
    after = build_sequence_features(future, thresholds).iloc[:len(before)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after)


def test_thresholds_are_training_end_bounded():
    base = _history()
    future = base.copy(); future["completion_year"] = 2025; future["cost_escalation_percentage"] = 10000
    mixed = pd.concat([base, future], ignore_index=True)
    a = learn_event_thresholds(base, 2019)
    b = learn_event_thresholds(mixed, 2019)
    assert a == b


def test_exp123_adapts_only_the_oof_aft_gate_and_restores_selector(monkeypatch, tmp_path):
    calls = []

    def fake_selector(frame, limit=688):
        calls.append(int(limit))
        if int(limit) == 688:
            raise RuntimeError(
                "Only 249 projects have AFT evidence; cannot form the requested "
                "688-project calibration cohort."
            )
        return {f"P{i}" for i in range(int(limit))}

    def fake_rolling(*args, **kwargs):
        selected = exp35_production._select_aft_calibration_projects(pd.DataFrame(), limit=688)
        assert len(selected) == 249
        return pd.DataFrame({
            "canonical_project_id": ["P1"],
            "snapshot_date": pd.to_datetime(["2019-01-01"]),
            "oof_year": [2019],
        })

    monkeypatch.setattr(exp35_production, "_select_aft_calibration_projects", fake_selector)
    monkeypatch.setattr(exp123, "rolling_production_oof", fake_rolling)

    result = exp123._rolling_oof_with_adaptive_aft_gate(
        pd.DataFrame(),
        pd.DataFrame(),
        training_start=2001,
        training_end=2019,
        root=tmp_path,
    )

    assert calls == [688, 249]
    assert result["oof_year"].tolist() == [2019]
    assert exp35_production._select_aft_calibration_projects is fake_selector
