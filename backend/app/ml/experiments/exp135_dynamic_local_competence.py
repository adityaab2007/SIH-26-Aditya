"""Exp135 — per-snapshot ET/LGBM/XGB weights from strict temporal OOF competence."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from backend.app.ml.experiments.nextgen_common import _compare, _gain, _metric, _prepare
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline, _regressors, temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production

EXPERIMENT_ID="Exp135"; SEED=13503; FAMILIES=("extra_trees","lightgbm","xgboost"); MIN_NEIGHBORS=12; K_NEIGHBORS=48; LOCAL_SHARE=.75

def _representation(train,features):
    cols=[f for f in features if f in train and pd.api.types.is_numeric_dtype(train[f])]
    if not cols: raise ValueError("Exp135 requires numeric production features")
    med=train[cols].apply(pd.to_numeric,errors="coerce").median(); sc=RobustScaler().fit(train[cols].apply(pd.to_numeric,errors="coerce").fillna(med)); return cols,med,sc

def _x(frame,state):
    cols,med,sc=state; return np.asarray(sc.transform(frame[cols].apply(pd.to_numeric,errors="coerce").fillna(med)),float)

def _oof(train,features,target,seed):
    chunks=[]
    for j,(fit,val,year) in enumerate(_rolling_folds(train)):
        part=val[["canonical_project_id","snapshot_date","completion_year",target,*features]].copy()
        y=pd.to_numeric(val[target],errors="coerce").to_numpy(float)
        for family in FAMILIES:
            model=_fit_pipeline(_regressors(seed+j)[family],fit,features,target); p=np.asarray(model.predict(val[features]),float)
            part[f"pred_{family}"]=p; part[f"err_{family}"]=np.abs(y-p)
        part["oof_fold_year"]=int(year); chunks.append(part)
    if not chunks: raise ValueError("No strict forward OOF folds available")
    return pd.concat(chunks,ignore_index=True)

def local_competence_weights(query,query_year,history_x,history_year,errors,min_neighbors=MIN_NEIGHBORS,k=K_NEIGHBORS):
    eligible=np.flatnonzero(np.asarray(history_year,int)<int(query_year))
    if len(eligible)<min_neighbors:return None,int(len(eligible))
    d=np.linalg.norm(history_x[eligible]-np.asarray(query,float),axis=1); ok=np.isfinite(d); eligible,d=eligible[ok],d[ok]
    if len(eligible)<min_neighbors:return None,int(len(eligible))
    order=np.argsort(d)[:min(k,len(d))]; idx,d=eligible[order],d[order]; local=errors[idx]; valid=np.isfinite(local).all(axis=1)
    if int(valid.sum())<min_neighbors:return None,int(valid.sum())
    mae=np.average(local[valid],axis=0,weights=1/(d[valid]+.25)); inv=1/np.maximum(mae,1e-6); w=inv/inv.sum()
    return (w,int(valid.sum())) if np.isfinite(w).all() and np.all(w>=0) else (None,int(valid.sum()))

def _target(train,cohort,features,target,prod,seed):
    oof=_oof(train,features,target,seed); state=_representation(train,features); hx,qx=_x(oof,state),_x(cohort,state)
    hy=pd.to_numeric(oof.completion_year,errors="coerce").fillna(-1).to_numpy(int); qy=pd.to_numeric(cohort.completion_year,errors="coerce").fillna(10**9).to_numpy(int)
    errors=oof[[f"err_{f}" for f in FAMILIES]].to_numpy(float); models={f:_fit_pipeline(_regressors(seed)[f],train,features,target) for f in FAMILIES}
    fp=np.column_stack([np.asarray(models[f].predict(cohort[features]),float) for f in FAMILIES]); out=np.asarray(prod,float).copy(); ws=[]; ns=[]; fallback=0
    for i in range(len(cohort)):
        w,n=local_competence_weights(qx[i],qy[i],hx,hy,errors); ns.append(n)
        if w is None or not np.isfinite(fp[i]).all(): fallback+=1; ws.append([np.nan]*3); continue
        out[i]=LOCAL_SHARE*float(w@fp[i])+(1-LOCAL_SHARE)*out[i]; ws.append(w.tolist())
    wa=np.asarray(ws,float)
    diag={"oof_rows":len(oof),"fallback_percentage":100*fallback/max(len(cohort),1),"neighbor_count_mean":float(np.mean(ns)) if ns else 0,
          "average_weights":{f:(float(np.nanmean(wa[:,j])) if np.isfinite(wa[:,j]).any() else None) for j,f in enumerate(FAMILIES)},
          "weight_std":{f:(float(np.nanstd(wa[:,j])) if np.isfinite(wa[:,j]).any() else None) for j,f in enumerate(FAMILIES)},
          "prior_fold_rule":"source completion_year < query completion_year","production_fallback_share":1-LOCAL_SHARE}
    return out,diag

def fit_against_production(*,data,training_start,training_end,test_end,production_bundle,production_receipt=None):
    frame=_prepare(data); train,test=temporal_project_split(frame,training_start,training_end,test_end); c=_compare(test); meta=production_bundle["metadata"]
    cf=list(meta["cost_features_used"]); df=list(meta["delay_features_used"]); cm,dm=production_bundle["cost"],production_bundle["delay"]
    pc=np.asarray(cm.predict(c[cf]),float); pdly=np.maximum(0,np.asarray(dm.predict(c[df]),float)); ec,cd=_target(train,c,cf,"actual_cost_overrun_percentage",pc,SEED); ed,dd=_target(train,c,df,"actual_delay_days",pdly,SEED+100); ed=np.maximum(0,ed)
    pcm,ecm=_metric(c,"actual_cost_overrun_percentage",pc),_metric(c,"actual_cost_overrun_percentage",ec); pdm,edm=_metric(c,"actual_delay_days",pdly),_metric(c,"actual_delay_days",ed)
    m={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":_gain(pcm,ecm),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":_gain(pdm,edm),"comparison_test_projects":int(c.canonical_project_id.nunique()),"comparison_test_snapshots":len(c)}
    return {"experiment":{"experiment_id":EXPERIMENT_ID,"seed":SEED,"diagnostics":{"cost":cd,"delay":dd}},"overall_comparison":m}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--start",type=int,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--test-end",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args(); data,_=build_training_dataset();data=data.copy();data["completion_year"]=pd.to_numeric(data.completion_year,errors="coerce");bundle,receipt=fast_current_production(data,a.start,a.end,a.test_end);r=fit_against_production(data=data,training_start=a.start,training_end=a.end,test_end=a.test_end,production_bundle=bundle,production_receipt=receipt);m=r["overall_comparison"]
    v={k:("IMPROVED" if m[f"{k}_improvement_percentage"]>0 else "REGRESSED" if m[f"{k}_improvement_percentage"]<0 else "UNCHANGED") for k in ("cost","delay")}; payload={"experiment":EXPERIMENT_ID,"window":f"{a.start}-{a.end}","production":{"cost_mae":m["production_cost_mae"],"delay_mae":m["production_delay_mae"]},"experiment_metrics":{"cost_mae":m["experiment_cost_mae"],"delay_mae":m["experiment_delay_mae"]},"improvement":{"cost_percent":m["cost_improvement_percentage"],"delay_percent":m["delay_improvement_percentage"]},"cohort":{"projects":m["comparison_test_projects"],"snapshots":m["comparison_test_snapshots"]},"diagnostics":r["experiment"]["diagnostics"],"verdict":v};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    for k,val in [("WINDOW",f"{a.start}_{a.end}"),("PRODUCTION_COST_MAE",m["production_cost_mae"]),("EXPERIMENT_COST_MAE",m["experiment_cost_mae"]),("COST_IMPROVEMENT_PERCENT",m["cost_improvement_percentage"]),("PRODUCTION_DELAY_MAE",m["production_delay_mae"]),("EXPERIMENT_DELAY_MAE",m["experiment_delay_mae"]),("DELAY_IMPROVEMENT_PERCENT",m["delay_improvement_percentage"]),("PROJECT_COUNT",m["comparison_test_projects"]),("SNAPSHOT_COUNT",m["comparison_test_snapshots"]),("VERDICT_COST",v["cost"]),("VERDICT_DELAY",v["delay"])]: print(f"{k}={val}")
if __name__=="__main__":main()
