"""Experiment 75: reliability-weighted revised-cost correction."""
from backend.app.ml.experiments.cost_residual_challenger_common import ChallengerConfig, fit_challenger, filter_rows, predict_row

EXPERIMENT_ID = "exp_75"
EXPERIMENT_SEQUENCE = 75
EXPERIMENT_NAME = "Reliability-Weighted Revised-Cost Correction"
EXPERIMENT_SCOPE = "cost"
CONFIG = ChallengerConfig(EXPERIMENT_ID, EXPERIMENT_SEQUENCE, EXPERIMENT_NAME, "revised_cost_reliability")


def fit_against_production(**kwargs):
    return fit_challenger(CONFIG, **kwargs)


def filter_comparable_rows(held, runtime_state):
    return filter_rows(held, runtime_state)


def predict_project(row, runtime_state):
    return predict_row(row, runtime_state)
