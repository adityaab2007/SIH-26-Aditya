import hashlib
import json

import numpy as np

from backend.app.ml.experiments.log_cost_target import (
    BASELINE_KEY,
    EXPERIMENT_DIR,
    evaluate_on_original_scale,
    inverse_log_cost_transform,
    log_cost_transform,
)
from backend.app.ml.real_time_windows import FEATURES, MODELS


def test_log_transformation_applied_correctly():
    values = np.array([0.0, 10.0, 500.0])
    assert np.allclose(log_cost_transform(values), np.log1p(values))


def test_log_prediction_converts_back_to_percentage():
    logged = np.array([3.5])
    assert np.allclose(inverse_log_cost_transform(logged), np.expm1(logged))


def test_baseline_model_remains_unchanged():
    baseline = MODELS / BASELINE_KEY / "cost_model.pkl"
    metadata = json.loads((EXPERIMENT_DIR / "metadata.json").read_text())
    digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert EXPERIMENT_DIR.resolve() != baseline.parent.resolve()
    assert metadata["baseline_unchanged"] is True
    assert metadata["baseline_cost_model_sha256"] == digest


def test_same_features_are_used_in_both_experiments():
    baseline = json.loads((MODELS / BASELINE_KEY / "metadata.json").read_text())
    experiment = json.loads((EXPERIMENT_DIR / "metadata.json").read_text())
    assert experiment["features_used"] == baseline["features_used"] == FEATURES


def test_evaluation_occurs_on_original_percentage_scale():
    actual = np.array([10.0, 500.0])
    logged_predictions = log_cost_transform(np.array([12.0, 450.0]))
    metrics = evaluate_on_original_scale(actual, logged_predictions)
    assert metrics["mae"] == 26.0
    assert np.allclose(metrics["predictions"], [12.0, 450.0])


def test_signed_extension_preserves_negative_overruns():
    values = np.array([-80.0, -10.0, 0.0, 10.0])
    assert np.allclose(inverse_log_cost_transform(log_cost_transform(values)), values)
