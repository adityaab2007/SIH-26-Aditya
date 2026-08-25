"""Retrain & Compare adapter for Experiment 12."""
from __future__ import annotations

from backend.app.ml.experiments.trajectory_exp12 import (
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERIMENT_SCOPE,
    filter_comparable_rows,
    fit_experiment,
    predict_project,
)

EXPERIMENT_SEQUENCE = 12


def fit_against_production(**kwargs):
    return fit_experiment(**kwargs)
