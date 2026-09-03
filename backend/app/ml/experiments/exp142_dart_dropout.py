"""Exp142 — clean DART architecture challenge for production-selected boosted families."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.base import clone
from backend.app.ml.experiments.nextgen_common import _compare,_gain,_metric,_prepare
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production
EXPERIMENT_ID="Exp142";SEED=14203;GRID=((.05,.2),(.05,.5),(.10,.2),(.10,.5))
def model_family(pipe):
    n=pipe.named_steps["model"].__class__.__name__.lower();return "lightgbm" if "lgbm" in n else "xgboost" if "xgb" in n else "extra_trees"
def dart_estimator(family,seed,drop_rate=.05,skip_drop=.2):
    if family not in {"lightgbm","xgboost"}:raise ValueError("DART is only applicable to LightGBM/XGBoost")
    m=clone(_regressors(seed)[family])
    if family=="lightgbm":m.set_params(boosting_type="dart",drop_rate=float(drop_rate),skip_drop=float(skip_drop),random_state=seed)
    else:m.set_params(booster="dart",rate_drop=float(drop_rate),skip_drop=float(skip_drop),sample_type="uniform",normalize_type="tree",random_state=seed,n_jobs=2)
    return m
def is_dart_estimator(model):
    p=model.get_params();return p.get("boosting_type")=="dart" or p.get("booster")=="dart"
def _select(train,features,target,family,seed):
    year=int(pd.to_numeric(train.completion_year,errors="coerce").max());fit=train[pd.to_numeric(train.completion_year,errors="coerce").lt(year)].copy();val=train[pd.to_numeric(train.completion_year,errors="coerce").eq(year)].copy()
    if fit.canonical_project_id.nunique()<10 or val.canonical_project_id.nunique()<2:return GRID[0],{}
    standard=_fit_pipeline(_regressors(seed)[family],fit,features,target);base=_metric(val,target,np.asarray(standard.predict(val[features]),float));scores={}
    for rate,skip in GRID:
        m=_fit_pipeline(dart_estimator(family,seed,rate,skip),fit,features,target);scores[f"rate={rate},skip={skip}"]=_metric(val,target,np.asarray(m.predict(val[features]),float))
    key=min(scores,key=scores.get);parts=dict(p.split("=") for p in key.split(","));best=(float(parts["rate"]),float(parts["skip"]));return best,{"standard_boosting_validation_mae":base,"dart_validation_mae":scores,"selected":key}
def _target(train,c,features,target,prod,pipe,seed):
    family=model_family(pipe)
    if family=="extra_trees":return np.asarray(prod,float).copy(),{"applicable":False,"reason":"current production selected ExtraTrees; clean DART substitution intentionally skipped","production_family":family}
    params,diag=_select(train,features,target,family,seed);start=time.perf_counter();model=_fit_pipeline(dart_estimator(family,seed,*params),train,features,target);runtime=time.perf_counter()-start;pred=np.asarray(model.predict(c[features]),float);diag.update({"applicable":True,"production_family":family,"selected_dart_parameters":{"drop_rate":params[0],"skip_drop":params[1]},"dart_confirmed":is_dart_estimator(model.named_steps["model"]),"fit_runtime_seconds":runtime,"feature_count":len(features),"seed":seed});return pred,diag
def fit_against_production(*,data,training_start,training_end,test_end,production_bundle,production_receipt=None):
    f=_prepare(data);train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);meta=production_bundle["metadata"];cf=list(meta["cost_features_used"]);df=list(meta["delay_features_used"]);cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c[cf]),float);pdly=np.maximum(0,np.asarray(dm.predict(c[df]),float));ec,cd=_target(train,c,cf,"actual_cost_overrun_percentage",pc,cm,SEED);ed,dd=_target(train,c,df,"actual_delay_days",pdly,dm,SEED+100);ed=np.maximum(0,ed);pcm,ecm=_metric(c,"actual_cost_overrun_percentage",pc),_metric(c,"actual_cost_overrun_percentage",ec);pdm,edm=_metric(c,"actual_delay_days",pdly),_metric(c,"actual_delay_days",ed);m={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":_gain(pcm,ecm),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":_gain(pdm,edm),"comparison_test_projects":int(c.canonical_project_id.nunique()),"comparison_test_snapshots":len(c)};return {"experiment":{"experiment_id":EXPERIMENT_ID,"seed":SEED,"diagnostics":{"cost":cd,"delay":dd}},"overall_comparison":m}
def main():
 p=argparse.ArgumentParser();p.add_argument("--start",type=int,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--test-end",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args();data,_=build_training_dataset();data=data.copy();data["completion_year"]=pd.to_numeric(data.completion_year,errors="coerce");b,r=fast_current_production(data,a.start,a.end,a.test_end);z=fit_against_production(data=data,training_start=a.start,training_end=a.end,test_end=a.test_end,production_bundle=b,production_receipt=r);m=z["overall_comparison"];v={k:("IMPROVED" if m[f"{k}_improvement_percentage"]>0 else "REGRESSED" if m[f"{k}_improvement_percentage"]<0 else "UNCHANGED") for k in ("cost","delay")};payload={"experiment":EXPERIMENT_ID,"window":f"{a.start}-{a.end}","production":{"cost_mae":m["production_cost_mae"],"delay_mae":m["production_delay_mae"]},"experiment_metrics":{"cost_mae":m["experiment_cost_mae"],"delay_mae":m["experiment_delay_mae"]},"improvement":{"cost_percent":m["cost_improvement_percentage"],"delay_percent":m["delay_improvement_percentage"]},"cohort":{"projects":m["comparison_test_projects"],"snapshots":m["comparison_test_snapshots"]},"diagnostics":z["experiment"]["diagnostics"],"verdict":v};o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");
 for k,val in [("WINDOW",f"{a.start}_{a.end}"),("PRODUCTION_COST_MAE",m["production_cost_mae"]),("EXPERIMENT_COST_MAE",m["experiment_cost_mae"]),("COST_IMPROVEMENT_PERCENT",m["cost_improvement_percentage"]),("PRODUCTION_DELAY_MAE",m["production_delay_mae"]),("EXPERIMENT_DELAY_MAE",m["experiment_delay_mae"]),("DELAY_IMPROVEMENT_PERCENT",m["delay_improvement_percentage"]),("PROJECT_COUNT",m["comparison_test_projects"]),("SNAPSHOT_COUNT",m["comparison_test_snapshots"]),("VERDICT_COST",v["cost"]),("VERDICT_DELAY",v["delay"])]:print(f"{k}={val}")
if __name__=="__main__":main()
