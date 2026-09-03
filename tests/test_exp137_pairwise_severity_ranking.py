import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from backend.app.ml.experiments.exp137_pairwise_severity_ranking import sample_pairs,monotonic_calibration_check

def test_pair_labels_match_severity_and_cross_projects():
    f=pd.DataFrame({"canonical_project_id":["a","b","c","d"],"snapshot_date":pd.date_range("2020-01-01",periods=4),"y":[1.,4.,2.,9.]})
    x,pairs=sample_pairs(f,"y",max_pairs=20,seed=2)
    assert pairs
    for i,j,label in pairs:
        assert x.iloc[i].canonical_project_id!=x.iloc[j].canonical_project_id
        assert label==int(float(x.iloc[i].y)>float(x.iloc[j].y))

def test_monotonic_calibrator_remains_monotonic():
    cal=IsotonicRegression(out_of_bounds="clip").fit([-2,-1,0,1,2],[0,1,1,3,5])
    assert monotonic_calibration_check(cal,[2,-2,.5,1.5])

def test_pair_sampling_is_reproducible_and_bounded():
    f=pd.DataFrame({"canonical_project_id":[f"p{i}" for i in range(20)],"snapshot_date":pd.date_range("2020-01-01",periods=20),"y":np.arange(20.)})
    _,a=sample_pairs(f,"y",max_pairs=50,seed=7);_,b=sample_pairs(f,"y",max_pairs=50,seed=7)
    assert a==b and len(a)<=50
