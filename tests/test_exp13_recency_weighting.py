import numpy as np
import pandas as pd

from backend.app.ml.experiments.adapter_exp13 import (
    CANDIDATE_HALF_LIVES,
    apply_project_recency_weights,
    compute_project_recency_weights,
)
from backend.app.ml.experiments.adapters import get_experiment_adapter


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
    assert np.isfinite(weighted.sample_weight).all() and (weighted.sample_weight > 0).all()


def test_control_and_candidate_contract_are_fixed_and_discoverable():
    control = compute_project_recency_weights(_frame(), 2021, None)
    assert np.allclose(control.to_numpy(), 1.0)
    assert CANDIDATE_HALF_LIVES == (None, 15, 10, 7, 5, 3)
    adapter = get_experiment_adapter("exp_13")
    assert adapter.name == "Recency-Weighted Project Training"
    assert adapter.scope == "cost_delay"
