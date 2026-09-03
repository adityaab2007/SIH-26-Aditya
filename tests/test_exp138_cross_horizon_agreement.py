import numpy as np,pandas as pd
from backend.app.ml.experiments.exp138_cross_horizon_agreement import agreement_features,bounded_gate,apply_gated_correction

def test_agreement_and_insufficient_history():
    f=pd.DataFrame({"cost_growth_velocity_3m":[1.,1.,np.nan],"cost_growth_velocity_6m":[2.,-2.,np.nan],"cost_acceleration":[-1.,3.,np.nan],"progress_velocity_3m":[3.,1.,np.nan],"progress_velocity_6m":[2.,-1.,np.nan],"progress_acceleration":[1.,2.,np.nan]})
    x=agreement_features(f)
    assert x.loc[0,"exp138_cost_direction_agreement"]==1
    assert x.loc[1,"exp138_cost_direction_agreement"]==0
    assert not bool(x.loc[2,"exp138_history_sufficient"])
    assert bounded_gate(x)[2]==0

def test_gate_is_bounded_and_zero_preserves_production():
    prod=np.array([10.,20.]);corr=np.array([100.,-100.]);gate=np.array([0.,2.])
    out=apply_gated_correction(prod,corr,gate)
    assert out[0]==prod[0] and out[1]==-80.

def test_gate_values_are_between_zero_and_one():
    x=pd.DataFrame({"exp138_agreement_score":[-1,.4,2.]})
    g=bounded_gate(x,1.5);assert np.all((g>=0)&(g<=1))
