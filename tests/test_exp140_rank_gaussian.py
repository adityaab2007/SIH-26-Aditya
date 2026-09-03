import numpy as np,pandas as pd
from backend.app.ml.experiments.exp140_rank_gaussian import RankGaussian

def test_transform_is_deterministic_and_frozen_to_training():
    train=pd.DataFrame({"x":[1.,2.,3.,4.,5.,6.]});hold=pd.DataFrame({"x":[0.,3.,100.]});t=RankGaussian().fit(train,["x"]);a,d=t.transform(hold);b,_=t.transform(hold.copy());assert np.allclose(a["x__rg"],b["x__rg"]);assert d["x"]["below_training_min"]>0 and d["x"]["above_training_max"]>0

def test_unseen_extremes_are_finite_after_clipping():
    t=RankGaussian().fit(pd.DataFrame({"x":np.arange(10.)}),["x"]);x,_=t.transform(pd.DataFrame({"x":[-1e9,1e9]}));assert np.isfinite(x["x__rg"]).all()

def test_low_cardinality_feature_is_skipped():
    t=RankGaussian().fit(pd.DataFrame({"x":[0,0,1,1]}),["x"]);assert t.added==[]
