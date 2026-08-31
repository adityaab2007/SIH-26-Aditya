import numpy as np
import pandas as pd
from backend.app.ml.experiments.adapter_exp75 import CONFIG, EXPERIMENT_ID
from backend.app.ml.experiments.cost_residual_challenger_common import _qedges, _bins

def test_exp75_identity_and_scope():
    assert EXPERIMENT_ID == "exp_75"
    assert CONFIG.strategy == "revised_cost_reliability"

def test_quantile_binning_is_deterministic_and_missing_safe():
    values=pd.Series([0.0,0.1,0.4,0.8,1.0,np.nan])
    edges=_qedges(values,4)
    first=_bins(values,edges)
    second=_bins(values,edges)
    assert first.equals(second)
    assert int(first.iloc[-1]) == -1
