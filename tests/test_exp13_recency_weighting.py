import numpy as np
import pandas as pd

from backend.app.ml.experiments import adapter_exp13
from backend.app.ml.experiments.adapter_exp13 import (
    CANDIDATE_HALF_LIVES,
    _window_verdict,
    apply_project_recency_weights,
    compute_project_recency_weights,
)


def _frame():
    return pd.DataFrame({
        "canonical_project_id": ["cutoff", "old", "old", "older"],
        "completion_year": [2021, 2016, 2016, 2011],
    })


def test_recency_formula_and_project_balance():
    weights = compute_project_recency_weights(_frame(), 2021, 5)
    assert np.isclose(weights["cutoff"], 3 / (1 + .5 + .25))
    assert np.isclose(weights["old"] / weights["cutoff"], .5)
    assert np.isclose(weights["older"] / weights["cutoff"], .25)

    weighted, _ = apply_project_recency_weights(_frame(), 2021, 5)
    totals = weighted.groupby("canonical_project_id").sample_weight.sum()
    assert np.isclose(totals["old"], weights["old"])
    assert np.isclose(totals["older"], weights["older"])
    assert np.isfinite(weighted.sample_weight).all()
    assert (weighted.sample_weight > 0).all()


def test_control_and_candidate_batch_contract_is_fixed():
    control = compute_project_recency_weights(_frame(), 2021, None)
    assert np.allclose(control.to_numpy(), 1.0)
    assert CANDIDATE_HALF_LIVES == (None, 15, 10, 7, 5, 3)
    assert adapter_exp13.EXPERIMENT_ID == "exp_13"
    assert adapter_exp13.EXPERIMENT_NAME == "Recency-Weighted Project Training"
    assert adapter_exp13.EXPERIMENT_SCOPE == "cost_delay"
    assert callable(adapter_exp13.fit_against_production)
    assert callable(adapter_exp13.filter_comparable_rows)
    assert callable(adapter_exp13.predict_project)


def test_verdict_contract_is_explicit():
    assert _window_verdict(1.0, 1.0) == "PROMOTION CANDIDATE"
    assert _window_verdict(-1.0, -1.0) == "REGRESSION / DO NOT PROMOTE"
    assert _window_verdict(1.0, -1.0) == "MIXED / NEEDS REVIEW"
    assert _window_verdict(None, 1.0) == "INVALID / INCOMPLETE"
