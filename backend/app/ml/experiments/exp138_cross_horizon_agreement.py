"""Exp138 — gate residual corrections by causal 3m/6m trajectory agreement."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from backend.app.ml.experiments.nextgen_common import _compare,_gain,_metric,_prepare
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production
EXPERIMENT_ID="Exp138";SEED=13803
GATE_FEATURES=["exp138_cost_direction_agreement","exp138_progress_direction_agreement","exp138_cost_magnitude_consistency","exp138_progress_magnitude_consistency","exp138_cost_acceleration_consistency","exp138_progress_acceleration_consistency","exp138_agreement_score"]

def agreement_features(frame):
    x=frame.copy()
    def num(c):return pd.to_numeric(x[c],errors="coerce") if c in x else pd.Series(np.nan,index=x.index)
    c3,c6,ca=num("cost_growth_velocity_3m"),num("cost_growth_velocity_6m"),num("cost_acceleration");p3,p6,pa=num("progress_velocity_3m"),num("progress_velocity_6m"),num("progress_acceleration")
    def direction(a,b):return pd.Series(np.where(a.notna()&b.notna(),(np.sign(a)==np.sign(b)).astype(float),np.nan),index=x.index)
    def magnitude(a,b):
        ok=a.notna()&b.notna();v=1-(a-b).abs()/(a.abs()+b.abs()+1e-6);return pd.Series(np.where(ok,np.clip(v,0,1),np.nan),index=x.index)
    x["exp138_cost_direction_agreement"]=direction(c3,c6);x["exp138_progress_direction_agreement"]=direction(p3,p6);x["exp138_cost_magnitude_consistency"]=magnitude(c3,c6);x["exp138_progress_magnitude_consistency"]=magnitude(p3,p6)
    x["exp138_cost_acceleration_consistency"]=pd.Series(np.where(ca.notna()&c3.notna()&c6.notna(),(np.sign(ca)==np.sign(c3-c6)).astype(float),np.nan),index=x.index);x["exp138_progress_acceleration_consistency"]=pd.Series(np.where(pa.notna()&p3.notna()&p6.notna(),(np.sign(pa)==np.sign(p3-p6)).astype(float),np.nan),index=x.index)
    vals=x[GATE_FEATURES[:-1]];x["exp138_agreement_score"]=vals.mean(axis=1,skipna=True);x["exp138_history_sufficient"]=vals.notna().sum(axis=1).ge(2);x.loc[~x.exp138_history_sufficient,"exp138_agreement_score"]=0.;return x

def bounded_gate(frame,scale=1.0):return np.clip(pd.to_numeric(frame["exp138_agreement_score"],errors="coerce").fillna(0).to_numpy(float)*float(scale),0,1)
def apply_gated_correction(production,correction,gate):return np.asarray(production,float)+np.clip(np.asarray(gate,float),0,1)*np.asarray(correction,float)
def _family(pipe):
    n=pipe.named_steps["model"].__class__.__name__.lower();return "lightgbm" if "lgbm" in n else "xgboost" if "xgb" in n else "extra_trees"
def _fit_residual_gate(train,features,target,family,seed):
    chunks=[]
    for j,(fit,val,year) in enumerate(_rolling_folds(train)):
        base=_fit_pipeline(_regressors(seed+j)[family],fit,features,target);p=np.asarray(base.predict(val[features]),float);part=agreement_features(val);part=part[[target,"sample_weight",*GATE_FEATURES,"exp138_history_sufficient"]].copy();part["base_prediction"]=p;part["residual"]=pd.to_numeric(val[target],errors="coerce").to_numpy(float)-p;chunks.append(part)
    if not chunks:raise ValueError("no forward OOF folds for Exp138")
    oof=pd.concat(chunks,ignore_index=True);med=oof[GATE_FEATURES].median();X=oof[GATE_FEATURES].fillna(med).fillna(0);model=Ridge(alpha=20.).fit(X,oof.residual,sample_weight=oof.sample_weight);corr=np.asarray(model.predict(X),float);best=(0.,float(np.average(np.abs(oof.residual),weights=oof.sample_weight)))
    for scale in (.25,.5,.75,1.):
        pred=apply_gated_correction(oof.base_prediction,corr,bounded_gate(oof,scale));mae=float(np.average(np.abs(pd.to_numeric(oof[target],errors="coerce")-pred),weights=oof.sample_weight))
        if mae<best[1]:best=(scale,mae)
    return model,med,float(best[0]),{"oof_rows":len(oof),"selected_gate_scale":float(best[0]),"oof_corrected_mae":best[1],"insufficient_history_percentage":float(100*(~oof.exp138_history_sufficient).mean())}
def _target(train,c,features,target,prod,pipe,seed):
    model,med,scale,diag=_fit_residual_gate(train,features,target,_family(pipe),seed);X=c[GATE_FEATURES].fillna(med).fillna(0);corr=np.asarray(model.predict(X),float);gate=bounded_gate(c,scale);pred=apply_gated_correction(prod,corr,gate);diag.update({"average_gate":float(np.mean(gate)),"average_abs_correction":float(np.mean(np.abs(gate*corr)))})
    groups={"high_agreement":gate>=.5,"low_agreement":(gate>0)&(gate<.5),"insufficient_history":~c.exp138_history_sufficient.to_numpy(bool)};diag["segments"]={}
    for name,mask in groups.items():
        if np.any(mask):diag["segments"][name]={"rows":int(np.sum(mask)),"production_mae":_metric(c.loc[mask],target,np.asarray(prod)[mask]),"experiment_mae":_metric(c.loc[mask],target,np.asarray(pred)[mask])}
    return pred,diag
def fit_against_production(*,data,training_start,training_end,test_end,production_bundle,production_receipt=None):
    f=agreement_features(_prepare(data));train,test=temporal_project_split(f,training_start,training_end,test_end);c=agreement_features(_compare(test));meta=production_bundle["metadata"];cf=list(meta["cost_features_used"]);df=list(meta["delay_features_used"]);cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c[cf]),float);pdly=np.maximum(0,np.asarray(dm.predict(c[df]),float));ec,cd=_target(train,c,cf,"actual_cost_overrun_percentage",pc,cm,SEED);ed,dd=_target(train,c,df,"actual_delay_days",pdly,dm,SEED+100);ed=np.maximum(0,ed);pcm,ecm=_metric(c,"actual_cost_overrun_percentage",pc),_metric(c,"actual_cost_overrun_percentage",ec);pdm,edm=_metric(c,"actual_delay_days",pdly),_metric(c,"actual_delay_days",ed);m={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":_gain(pcm,ecm),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":_gain(pdm,edm),"comparison_test_projects":int(c.canonical_project_id.nunique()),"comparison_test_snapshots":len(c)};return {"experiment":{"experiment_id":EXPERIMENT_ID,"seed":SEED,"diagnostics":{"cost":cd,"delay":dd}},"overall_comparison":m}
def main():
 p=argparse.ArgumentParser();p.add_argument("--start",type=int,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--test-end",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args();data,_=build_training_dataset();data=data.copy();data["completion_year"]=pd.to_numeric(data.completion_year,errors="coerce");b,r=fast_current_production(data,a.start,a.end,a.test_end);z=fit_against_production(data=data,training_start=a.start,training_end=a.end,test_end=a.test_end,production_bundle=b,production_receipt=r);m=z["overall_comparison"];v={k:("IMPROVED" if m[f"{k}_improvement_percentage"]>0 else "REGRESSED" if m[f"{k}_improvement_percentage"]<0 else "UNCHANGED") for k in ("cost","delay")};payload={"experiment":EXPERIMENT_ID,"window":f"{a.start}-{a.end}","production":{"cost_mae":m["production_cost_mae"],"delay_mae":m["production_delay_mae"]},"experiment_metrics":{"cost_mae":m["experiment_cost_mae"],"delay_mae":m["experiment_delay_mae"]},"improvement":{"cost_percent":m["cost_improvement_percentage"],"delay_percent":m["delay_improvement_percentage"]},"cohort":{"projects":m["comparison_test_projects"],"snapshots":m["comparison_test_snapshots"]},"diagnostics":z["experiment"]["diagnostics"],"verdict":v};o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");
 for k,val in [("WINDOW",f"{a.start}_{a.end}"),("PRODUCTION_COST_MAE",m["production_cost_mae"]),("EXPERIMENT_COST_MAE",m["experiment_cost_mae"]),("COST_IMPROVEMENT_PERCENT",m["cost_improvement_percentage"]),("PRODUCTION_DELAY_MAE",m["production_delay_mae"]),("EXPERIMENT_DELAY_MAE",m["experiment_delay_mae"]),("DELAY_IMPROVEMENT_PERCENT",m["delay_improvement_percentage"]),("PROJECT_COUNT",m["comparison_test_projects"]),("SNAPSHOT_COUNT",m["comparison_test_snapshots"]),("VERDICT_COST",v["cost"]),("VERDICT_DELAY",v["delay"])]:print(f"{k}={val}")
if __name__=="__main__":main()
