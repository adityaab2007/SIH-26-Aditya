"""Exp141 — redistribute each project's fixed training mass by target-free snapshot novelty."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from backend.app.ml.experiments.nextgen_common import _compare,_gain,_metric,_prepare
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production
EXPERIMENT_ID="Exp141";SEED=14103;EPS=.15

def fit_novelty_state(train,features):
    x=train.sort_values(["canonical_project_id","snapshot_date"]);numeric=[f for f in features if f in x and pd.api.types.is_numeric_dtype(x[f])];categorical=[f for f in features if f in x and f not in numeric];scales={}
    for f in numeric:
        v=pd.to_numeric(x[f],errors="coerce");d=v.groupby(x.canonical_project_id).diff().abs().dropna();s=float(d.median()) if len(d) else 1.;scales[f]=s if np.isfinite(s) and s>1e-9 else 1.
    return {"numeric":numeric,"categorical":categorical,"scales":scales}
def novelty_scores(frame,state):
    x=frame.copy();x["_order"]=np.arange(len(x));x=x.sort_values(["canonical_project_id","snapshot_date"]);parts=[]
    for f in state["numeric"]:
        v=pd.to_numeric(x[f],errors="coerce");d=v.groupby(x.canonical_project_id).diff().abs()/state["scales"][f];parts.append(d.clip(upper=5).rename(f))
    for f in state["categorical"]:
        cur=x[f].astype("string");prev=cur.groupby(x.canonical_project_id).shift(1);parts.append((cur.ne(prev)&prev.notna()).astype(float).rename(f))
    comp=pd.concat(parts,axis=1) if parts else pd.DataFrame(index=x.index);nov=comp.mean(axis=1,skipna=True).fillna(0.);first=x.groupby("canonical_project_id").cumcount().eq(0);neutral=float(nov[~first].median()) if (~first).any() else 1.;neutral=neutral if np.isfinite(neutral) and neutral>0 else 1.;nov.loc[first]=neutral;x["novelty_score"]=nov;x=x.sort_values("_order");return x["novelty_score"].set_axis(frame.index)
def redistribute_project_weights(train,features):
    state=fit_novelty_state(train,features);out=train.copy();nov=novelty_scores(out,state);raw=EPS+nov.clip(lower=0);orig=out.groupby("canonical_project_id").sample_weight.transform("sum");den=raw.groupby(out.canonical_project_id).transform("sum");out["sample_weight"]=orig*raw/den
    before=train.groupby("canonical_project_id").sample_weight.sum();after=out.groupby("canonical_project_id").sample_weight.sum();err=float((before-after).abs().max());eff=out.groupby("canonical_project_id").sample_weight.apply(lambda w:float(w.sum()**2/max(float((w*w).sum()),1e-12)))
    diag={"project_weight_conservation_error":err,"novelty_min":float(nov.min()),"novelty_median":float(nov.median()),"novelty_max":float(nov.max()),"effective_snapshot_count_mean":float(eff.mean()),"minimum_raw_floor":EPS,"uses_target_columns":False};return out,state,diag
def _family(pipe):
    n=pipe.named_steps["model"].__class__.__name__.lower();return "lightgbm" if "lgbm" in n else "xgboost" if "xgb" in n else "extra_trees"
def _segment_diag(c,target,prod,exp,nov,threshold):
    out={}
    for name,mask in {"high_change":nov>threshold,"low_change":nov<=threshold}.items():
        mask=np.asarray(mask,bool)
        if mask.any():out[name]={"rows":int(mask.sum()),"production_mae":_metric(c.loc[mask],target,np.asarray(prod)[mask]),"experiment_mae":_metric(c.loc[mask],target,np.asarray(exp)[mask])}
    return out
def fit_against_production(*,data,training_start,training_end,test_end,production_bundle,production_receipt=None):
    f=_prepare(data);train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);meta=production_bundle["metadata"];cf=list(meta["cost_features_used"]);df=list(meta["delay_features_used"]);allf=list(dict.fromkeys(cf+df));weighted,state,diag=redistribute_project_weights(train,allf);cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c[cf]),float);pdly=np.maximum(0,np.asarray(dm.predict(c[df]),float));ec=np.asarray(_fit_pipeline(_regressors(SEED)[_family(cm)],weighted,cf,"actual_cost_overrun_percentage").predict(c[cf]),float);ed=np.maximum(0,np.asarray(_fit_pipeline(_regressors(SEED+1)[_family(dm)],weighted,df,"actual_delay_days").predict(c[df]),float));nov=novelty_scores(c,state);threshold=diag["novelty_median"];diag["cost_segments"]=_segment_diag(c,"actual_cost_overrun_percentage",pc,ec,nov,threshold);diag["delay_segments"]=_segment_diag(c,"actual_delay_days",pdly,ed,nov,threshold)
    pcm,ecm=_metric(c,"actual_cost_overrun_percentage",pc),_metric(c,"actual_cost_overrun_percentage",ec);pdm,edm=_metric(c,"actual_delay_days",pdly),_metric(c,"actual_delay_days",ed);m={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":_gain(pcm,ecm),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":_gain(pdm,edm),"comparison_test_projects":int(c.canonical_project_id.nunique()),"comparison_test_snapshots":len(c)};return {"experiment":{"experiment_id":EXPERIMENT_ID,"seed":SEED,"diagnostics":diag},"overall_comparison":m}
def main():
 p=argparse.ArgumentParser();p.add_argument("--start",type=int,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--test-end",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args();data,_=build_training_dataset();data=data.copy();data["completion_year"]=pd.to_numeric(data.completion_year,errors="coerce");b,r=fast_current_production(data,a.start,a.end,a.test_end);z=fit_against_production(data=data,training_start=a.start,training_end=a.end,test_end=a.test_end,production_bundle=b,production_receipt=r);m=z["overall_comparison"];v={k:("IMPROVED" if m[f"{k}_improvement_percentage"]>0 else "REGRESSED" if m[f"{k}_improvement_percentage"]<0 else "UNCHANGED") for k in ("cost","delay")};payload={"experiment":EXPERIMENT_ID,"window":f"{a.start}-{a.end}","production":{"cost_mae":m["production_cost_mae"],"delay_mae":m["production_delay_mae"]},"experiment_metrics":{"cost_mae":m["experiment_cost_mae"],"delay_mae":m["experiment_delay_mae"]},"improvement":{"cost_percent":m["cost_improvement_percentage"],"delay_percent":m["delay_improvement_percentage"]},"cohort":{"projects":m["comparison_test_projects"],"snapshots":m["comparison_test_snapshots"]},"diagnostics":z["experiment"]["diagnostics"],"verdict":v};o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");
 for k,val in [("WINDOW",f"{a.start}_{a.end}"),("PRODUCTION_COST_MAE",m["production_cost_mae"]),("EXPERIMENT_COST_MAE",m["experiment_cost_mae"]),("COST_IMPROVEMENT_PERCENT",m["cost_improvement_percentage"]),("PRODUCTION_DELAY_MAE",m["production_delay_mae"]),("EXPERIMENT_DELAY_MAE",m["experiment_delay_mae"]),("DELAY_IMPROVEMENT_PERCENT",m["delay_improvement_percentage"]),("PROJECT_COUNT",m["comparison_test_projects"]),("SNAPSHOT_COUNT",m["comparison_test_snapshots"]),("VERDICT_COST",v["cost"]),("VERDICT_DELAY",v["delay"])]:print(f"{k}={val}")
if __name__=="__main__":main()
