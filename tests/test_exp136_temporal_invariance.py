import pandas as pd
from backend.app.ml.experiments.exp136_temporal_invariance import stable_feature_scores

def _frame():
    rows=[]
    for i in range(40):
        rows.append({"canonical_project_id":f"p{i//2}","completion_year":2000+i//4,"sample_weight":1.,"a":float(i),"b":float(i%3),"actual_cost_overrun_percentage":float(i%11)})
    return pd.DataFrame(rows)

def test_selector_is_deterministic_and_training_only():
    f=_frame();s1,d1=stable_feature_scores(f,["a","b"],"actual_cost_overrun_percentage",13603);s2,d2=stable_feature_scores(f.copy(),["a","b"],"actual_cost_overrun_percentage",13603)
    assert s1==s2 and d1["feature_scores"]==d2["feature_scores"]
    holdout=f.copy();holdout["completion_year"]+=100
    assert stable_feature_scores(f,["a","b"],"actual_cost_overrun_percentage",13603)[0]==s1

def test_scores_include_temporal_penalty():
    _,d=stable_feature_scores(_frame(),["a","b"],"actual_cost_overrun_percentage",13603)
    assert all(x["temporal_instability"]>=0 and x["stability_score"]>=0 for x in d["feature_scores"])
