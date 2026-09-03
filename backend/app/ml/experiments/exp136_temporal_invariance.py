"""Exp136 — training-only adversarial temporal-invariance feature selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from backend.app.ml.experiments.nextgen_common import _compare,_gain,_metric,_prepare
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline,_preprocessor,_regressors,temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production
EXPERIMENT_ID="Exp136";SEED=13603;KEEP_FRACTION=.70

def _aggregate(pipe,features):
    names=pipe.named_steps["preprocess"].get_feature_names_out();imp=np.asarray(pipe.named_steps["model"].feature_importances_,float);out={f:0. for f in features}
    for n,v in zip(names,imp):
        clean=n.split("__",1)[-1]; f=next((x for x in features if clean==x or clean.startswith(x+"_")),None)
        if f is not None:out[f]+=float(v)
    total=sum(out.values()) or 1.;return {k:v/total for k,v in out.items()}

def stable_feature_scores(train,features,target,seed=SEED):
    years=pd.to_numeric(train.completion_year,errors="coerce");cut=float(years.median());era=years.gt(cut).astype(int)
    prep=_preprocessor(train,features); reg=Pipeline([("preprocess",prep),("model",ExtraTreesRegressor(n_estimators=220,min_samples_leaf=3,max_features=.8,random_state=seed,n_jobs=2))]);reg.fit(train[features],train[target],model__sample_weight=train.sample_weight.to_numpy(float));target_imp=_aggregate(reg,features)
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed);ti,vi=next(split.split(train,groups=train.canonical_project_id));clf=Pipeline([("preprocess",_preprocessor(train.iloc[ti],features)),("model",RandomForestClassifier(n_estimators=220,min_samples_leaf=3,class_weight="balanced",random_state=seed,n_jobs=2))]);clf.fit(train.iloc[ti][features],era.iloc[ti],model__sample_weight=train.iloc[ti].sample_weight.to_numpy(float));acc=float(accuracy_score(era.iloc[vi],clf.predict(train.iloc[vi][features])))
    full=Pipeline([("preprocess",_preprocessor(train,features)),("model",RandomForestClassifier(n_estimators=220,min_samples_leaf=3,class_weight="balanced",random_state=seed,n_jobs=2))]);full.fit(train[features],era,model__sample_weight=train.sample_weight.to_numpy(float));temporal=_aggregate(full,features)
    rows=[]
    for f in features:
        combined=target_imp.get(f,0.)/(1.+2.*temporal.get(f,0.));rows.append({"feature":f,"target_importance":target_imp.get(f,0.),"temporal_instability":temporal.get(f,0.),"stability_score":combined})
    rows=sorted(rows,key=lambda x:(-x["stability_score"],x["feature"]));keep=max(8,int(np.ceil(len(features)*KEEP_FRACTION)));selected=[r["feature"] for r in rows[:keep]]
    return selected,{"era_cutoff":cut,"adversarial_validation_accuracy":acc,"selected_features":selected,"rejected_features":[f for f in features if f not in selected],"feature_scores":rows}

def _family(pipe):
    n=pipe.named_steps["model"].__class__.__name__.lower();return "lightgbm" if "lgbm" in n else "xgboost" if "xgb" in n else "extra_trees"

def fit_against_production(*,data,training_start,training_end,test_end,production_bundle,production_receipt=None):
    f=_prepare(data);train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);meta=production_bundle["metadata"];cf=list(meta["cost_features_used"]);df=list(meta["delay_features_used"]);cm,dm=production_bundle["cost"],production_bundle["delay"]
    pc=np.asarray(cm.predict(c[cf]),float);pdly=np.maximum(0,np.asarray(dm.predict(c[df]),float));cs,cd=stable_feature_scores(train,cf,"actual_cost_overrun_percentage",SEED);ds,dd=stable_feature_scores(train,df,"actual_delay_days",SEED+1)
    ec=np.asarray(_fit_pipeline(_regressors(SEED)[_family(cm)],train,cs,"actual_cost_overrun_percentage").predict(c[cs]),float);ed=np.maximum(0,np.asarray(_fit_pipeline(_regressors(SEED+1)[_family(dm)],train,ds,"actual_delay_days").predict(c[ds]),float))
    pcm,ecm=_metric(c,"actual_cost_overrun_percentage",pc),_metric(c,"actual_cost_overrun_percentage",ec);pdm,edm=_metric(c,"actual_delay_days",pdly),_metric(c,"actual_delay_days",ed);m={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":_gain(pcm,ecm),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":_gain(pdm,edm),"comparison_test_projects":int(c.canonical_project_id.nunique()),"comparison_test_snapshots":len(c)};return {"experiment":{"experiment_id":EXPERIMENT_ID,"seed":SEED,"diagnostics":{"cost":cd,"delay":dd}},"overall_comparison":m}
def main():
 p=argparse.ArgumentParser();p.add_argument("--start",type=int,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--test-end",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args();data,_=build_training_dataset();data=data.copy();data["completion_year"]=pd.to_numeric(data.completion_year,errors="coerce");b,r=fast_current_production(data,a.start,a.end,a.test_end);z=fit_against_production(data=data,training_start=a.start,training_end=a.end,test_end=a.test_end,production_bundle=b,production_receipt=r);m=z["overall_comparison"];v={k:("IMPROVED" if m[f"{k}_improvement_percentage"]>0 else "REGRESSED" if m[f"{k}_improvement_percentage"]<0 else "UNCHANGED") for k in ("cost","delay")};payload={"experiment":EXPERIMENT_ID,"window":f"{a.start}-{a.end}","production":{"cost_mae":m["production_cost_mae"],"delay_mae":m["production_delay_mae"]},"experiment_metrics":{"cost_mae":m["experiment_cost_mae"],"delay_mae":m["experiment_delay_mae"]},"improvement":{"cost_percent":m["cost_improvement_percentage"],"delay_percent":m["delay_improvement_percentage"]},"cohort":{"projects":m["comparison_test_projects"],"snapshots":m["comparison_test_snapshots"]},"diagnostics":z["experiment"]["diagnostics"],"verdict":v};o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");
 for k,val in [("WINDOW",f"{a.start}_{a.end}"),("PRODUCTION_COST_MAE",m["production_cost_mae"]),("EXPERIMENT_COST_MAE",m["experiment_cost_mae"]),("COST_IMPROVEMENT_PERCENT",m["cost_improvement_percentage"]),("PRODUCTION_DELAY_MAE",m["production_delay_mae"]),("EXPERIMENT_DELAY_MAE",m["experiment_delay_mae"]),("DELAY_IMPROVEMENT_PERCENT",m["delay_improvement_percentage"]),("PROJECT_COUNT",m["comparison_test_projects"]),("SNAPSHOT_COUNT",m["comparison_test_snapshots"]),("VERDICT_COST",v["cost"]),("VERDICT_DELAY",v["delay"])]:print(f"{k}={val}")
if __name__=="__main__":main()
