"""Experiment 76: medium-project residual specialist."""
from backend.app.ml.experiments.cost_residual_challenger_common import ChallengerConfig, fit_challenger, filter_rows, predict_row

EXPERIMENT_ID = "exp_76"
EXPERIMENT_SEQUENCE = 76
EXPERIMENT_NAME = "Medium-Project Residual Specialist"
EXPERIMENT_SCOPE = "cost"
CONFIG = ChallengerConfig(EXPERIMENT_ID, EXPERIMENT_SEQUENCE, EXPERIMENT_NAME, "medium_project_specialist")

def fit_against_production(**kwargs): return fit_challenger(CONFIG, **kwargs)
def filter_comparable_rows(held, runtime_state): return filter_rows(held, runtime_state)
def predict_project(row, runtime_state): return predict_row(row, runtime_state)
