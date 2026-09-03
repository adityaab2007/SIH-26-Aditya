import numpy as np
from backend.app.ml.experiments.exp135_dynamic_local_competence import local_competence_weights


def test_weights_are_nonnegative_and_sum_to_one():
    hx = np.arange(60.0).reshape(20, 3); years = np.arange(2000, 2020)
    err = np.column_stack([np.ones(20), np.ones(20) * 2, np.ones(20) * 4])
    w, n = local_competence_weights(np.array([60., 61., 62.]), 2025, hx, years, err, min_neighbors=5, k=10)
    assert n == 10
    assert w is not None and np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1] > w[2]


def test_only_prior_oof_rows_are_eligible():
    hx = np.zeros((8, 2)); years = np.array([2010, 2011, 2012, 2013, 2020, 2021, 2022, 2023])
    err = np.ones((8, 3))
    w, n = local_competence_weights(np.zeros(2), 2014, hx, years, err, min_neighbors=4, k=8)
    assert n == 4
    assert w is not None


def test_sparse_region_falls_back_to_production_signal():
    hx = np.zeros((3, 2)); years = np.array([2010, 2011, 2012]); err = np.ones((3, 3))
    w, n = local_competence_weights(np.zeros(2), 2014, hx, years, err, min_neighbors=4)
    assert w is None and n == 3
