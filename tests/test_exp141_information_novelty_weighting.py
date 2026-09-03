import numpy as np,pandas as pd
from backend.app.ml.experiments.exp141_information_novelty_weighting import fit_novelty_state,novelty_scores,redistribute_project_weights

def _frame():
    return pd.DataFrame({"canonical_project_id":["a","a","a","b","b"],"snapshot_date":pd.date_range("2020-01-01",periods=5),"sample_weight":[1/3,1/3,1/3,.5,.5],"x":[1.,1.,10.,2.,2.],"cat":["s","s","t","q","q"],"actual_cost_overrun_percentage":[1,9,99,2,8]})
def test_project_weight_mass_is_conserved():
    f=_frame();out,_,diag=redistribute_project_weights(f,["x","cat"]);before=f.groupby("canonical_project_id").sample_weight.sum();after=out.groupby("canonical_project_id").sample_weight.sum();assert np.allclose(before,after);assert diag["project_weight_conservation_error"]<1e-12

def test_targets_do_not_affect_novelty_weights():
    f=_frame();a,_,_=redistribute_project_weights(f,["x","cat"]);g=f.copy();g["actual_cost_overrun_percentage"]*=10000;b,_,_=redistribute_project_weights(g,["x","cat"]);assert np.allclose(a.sample_weight,b.sample_weight)

def test_duplicate_state_is_not_more_novel_than_material_transition():
    f=_frame().iloc[:3].copy();state=fit_novelty_state(f,["x","cat"]);n=novelty_scores(f,state);assert n.iloc[1]<=n.iloc[2]

def test_first_snapshot_has_positive_neutral_mass():
    f=_frame();state=fit_novelty_state(f,["x","cat"]);n=novelty_scores(f,state);assert n.iloc[0]>0 and n.iloc[3]>0
