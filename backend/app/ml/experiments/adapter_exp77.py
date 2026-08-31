"""Experiment 77: early-stage financial-surrogate booster."""
from backend.app.ml.experiments.cost_residual_challenger_common import ChallengerConfig, fit_challenger, filter_rows, predict_row
EXPERIMENT_ID="exp_77"; EXPERIMENT_SEQUENCE=77; EXPERIMENT_NAME="Early-Stage Financial-Surrogate Booster"; EXPERIMENT_SCOPE="cost"
CONFIG=ChallengerConfig(EXPERIMENT_ID,EXPERIMENT_SEQUENCE,EXPERIMENT_NAME,"early_financial_surrogate")
def fit_against_production(**kwargs): return fit_challenger(CONFIG,**kwargs)
def filter_comparable_rows(held,runtime_state): return filter_rows(held,runtime_state)
def predict_project(row,runtime_state): return predict_row(row,runtime_state)
