from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.experiments import lifecycle_specialists as specialists


def _cohort() -> pd.DataFrame:
    rows = []
    stages = ["early", "early_mid", "late_mid", "late"]
    for stage_index, stage in enumerate(stages):
        for index in range(10):
            rows.append({
                "canonical_project_id": f"TRAIN-{stage}-{index}",
                "project_name": f"Train {stage} {index}",
                "completion_year": 2015,
                "snapshot_date": pd.Timestamp("2014-06-30"),
                "lifecycle_stage": stage,
                "approved_cost_cr": 100.0 + index,
                "actual_cost_overrun_percentage": float(index),
                "actual_delay_days": float(30 + index),
                "actual_risk": "LOW",
                "sample_weight": 0.1,
            })
        for index in range(2):
            rows.append({
                "canonical_project_id": f"TEST-{stage}-{index}",
                "project_name": f"Test {stage} {index}",
                "completion_year": 2019,
                "snapshot_date": pd.Timestamp("2018-06-30"),
                "lifecycle_stage": stage,
                "approved_cost_cr": 200.0 + index,
                "actual_cost_overrun_percentage": float(10 + index),
                "actual_delay_days": float(100 + index),
                "actual_risk": "MEDIUM",
                "sample_weight": 0.5,
            })
    return pd.DataFrame(rows)


def _fake_train_variant(train, test, features, seed):
    rows = test[[
        "canonical_project_id", "project_name", "snapshot_date", "completion_year", "lifecycle_stage",
        "actual_cost_overrun_percentage", "actual_delay_days", "actual_risk", "sample_weight",
    ]].copy()
    rows["predicted_cost_overrun"] = rows.actual_cost_overrun_percentage.to_numpy(dtype=float)
    rows["predicted_delay_days"] = rows.actual_delay_days.to_numpy(dtype=float)
    rows["predicted_risk"] = rows.actual_risk.to_numpy(dtype=object)
    rows["cost_error"] = 0.0
    rows["delay_error"] = 0.0
    metrics = {
        "cost": {"MAE": 0.0, "RMSE": 0.0, "R2": 1.0, "MAPE": 0.0, "rows": len(rows), "unique_projects": rows.canonical_project_id.nunique()},
        "delay": {"MAE": 0.0, "RMSE": 0.0, "R2": 1.0, "MAPE": 0.0, "rows": len(rows), "unique_projects": rows.canonical_project_id.nunique()},
        "risk": {"accuracy": 1.0, "macro_precision": 0.25, "macro_recall": 0.25, "macro_f1": 0.25, "confusion_matrix": [], "labels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
    }
    return {
        "models": {"cost": object(), "delay": object(), "risk": object()},
        "selected_algorithms": {"cost": "stage_cost", "delay": "stage_delay"},
        "internal_comparisons": {},
    }, metrics, rows


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def test_experiment_four_fits_and_saves_independent_stage_specialists(tmp_path, monkeypatch):
    data = _cohort()
    monkeypatch.setattr(
        specialists,
        "audit_features",
        lambda *args, **kwargs: {"features_used": ["approved_cost_cr"]},
    )
    monkeypatch.setattr(specialists, "_train_variant", _fake_train_variant)
    monkeypatch.setattr(specialists, "_strict_selection", lambda train, features, target, seed: ("test", _ConstantModel(1.0), [{"algorithm": "test", "MAE": 1.0, "RMSE": 1.0}]))
    monkeypatch.setattr(specialists, "_importance", lambda *args, **kwargs: {"method": "tree_feature_importance", "features": []})
    monkeypatch.setattr(specialists.joblib, "dump", lambda model, path: Path(path).write_bytes(b"specialist"))
    monkeypatch.setattr(specialists, "git_commit_sha", lambda root: "test-commit")

    result = specialists.train_lifecycle_specialists(
        2001,
        2015,
        2021,
        data=data,
        identity=pd.DataFrame(),
        artifact_root=tmp_path,
    )

    assert result["implementation"] == "independent_stage_models"
    assert set(result["specialists"]) == {"early", "early_mid", "late_mid", "late"}
    for stage, item in result["specialists"].items():
        assert item["available"] is True
        assert item["training_projects"] == 10
        assert item["testing_projects"] == 2
        assert (tmp_path / "2001_2015" / stage / "cost_model.pkl").exists()
        assert (tmp_path / "2001_2015" / stage / "delay_model.pkl").exists()
    assert (tmp_path / "2001_2015" / "experiment_4_results.json").exists()
