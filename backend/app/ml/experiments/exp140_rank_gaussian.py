"""Exp140 — training-only empirical percentile / Gaussianized numeric representation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from statistics import NormalDist
import numpy as np,pandas as pd
from backend.app.ml.experiments.nextgen_common import _compare,_gain,_metric,_prepare
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _fit_pipeline,_regressors,temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production
EXPERIMENT_ID="Exp140";SEED=14003;CLIP=1e-4
class RankGaussian:
    def __init__(self,clip=CLIP):self.clip=float(clip);self.sorted={}
    def fit(self,frame,features):
        self.sorted={}
        for f in features:
            v=pd.to_numeric(frame[f],errors="coerce").dropna().to_numpy(float);u=np.unique(v)
            if len(u)>=5:self.sorted[f]=np.sort(v)
        return self
    def transform(self,frame):
        out=frame.copy();diag={}
        nd=NormalDist()
        for f,ref in self.sorted.items():
            v=pd.to_numeric(out[f],errors="coerce").to_numpy(float);mask=np.isfinite(v);pct=np.full(len(v),np.nan,float);pct[mask]=(np.searchsorted(ref,v[mask],side="right")-.5)/len(ref);pct[mask]=np.clip(pct[mask],self.clip,1-self.clip);out[f+"__rg"]=[nd.inv_cdf(float(p)) if np.isfinite(p) else np.nan for p in pct];diag[f]={"below_training_min":float(np.mean(v[mask]<ref[0])) if mask.any() else 0.,"above_training_max":float(np.mean(v[mask]>ref[-1])) if mask.any() else 0.}
        return out,diag
    @property
    def added(self):return [f+"__rg" for f in self.sorted]
def _family(pipe):
    n=pipe.named_steps["model"].__class__.__name__.lower();return "lightgbm" if "lgbm" in n else "xgboost" if "xgb" in n else "extra_trees"
def _numeric(frame,features):return [f for f in features if f in frame and pd.api.types.is_numeric_dtype(frame[f])]
def _variant_features(base,transformer,mode):
    numeric=set(transformer.sorted);categorical=[f for f in base if f not in numeric];return list(dict.fromkeys((base+transformer.added) if mode=="raw_plus_rank" else (categorical+transformer.added)))
def _select_mode(train,base,target,family,seed):
    year=int(pd.to_numeric(train.completion_year,errors="coerce").max());fit=train[pd.to_numeric(train.completion_year,errors="coerce").lt(year)].copy();val=train[pd.to_numeric(train.completion_year,errors="coerce").eq(year)].copy()
    if fit.canonical_project_id.nunique()<10 or val.canonical_project_id.nunique()<2:return "raw_plus_rank",{}
    tr=RankGaussian().fit(fit,_numeric(fit,base));ft,_=tr.transform(fit);vv,_=tr.transform(val);scores={}
    for mode in ("raw_plus_rank","rank_only"):
        feats=_variant_features(base,tr,mode);m=_fit_pipeline(_regressors(seed)[family],ft,feats,target);p=np.asarray(m.predict(vv[feats]),float);scores[mode]=_metric(vv,target,p)
    return min(scores,key=scores.get),scores
def _target(train,c,base,target,pipe,seed):
    family=_family(pipe);mode,selection=_select_mode(train,base,target,family,seed);tr=RankGaussian().fit(train,_numeric(train,base));tt,_=tr.transform(train);cc,clipdiag=tr.transform(c);feats=_variant_features(base,tr,mode);model=_fit_pipeline(_regressors(seed)[family],tt,feats,target);pred=np.asarray(model.predict(cc[feats]),float);return pred,{"transformed_features":tr.added,"selected_representation":mode,"internal_validation_mae":selection,"clipping_rates":clipdiag,"feature_count":len(feats)}
def fit_against_production(*,data,training_start,training_end,test_end,production_bundle,production_receipt=None):
    f=_prepare(data);train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);meta=production_bundle["metadata"];cf=list(meta["cost_features_used"]);df=list(meta["delay_features_used"]);cm,dm=production_bundle["cost"],production_bundle["delay"];pc=np.asarray(cm.predict(c[cf]),float);pdly=np.maximum(0,np.asarray(dm.predict(c[df]),float));ec,cd=_target(train,c,cf,"actual_cost_overrun_percentage",cm,SEED);ed,dd=_target(train,c,df,"actual_delay_days",dm,SEED+100);ed=np.maximum(0,ed);pcm,ecm=_metric(c,"actual_cost_overrun_percentage",pc),_metric(c,"actual_cost_overrun_percentage",ec);pdm,edm=_metric(c,"actual_delay_days",pdly),_metric(c,"actual_delay_days",ed);m={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":_gain(pcm,ecm),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":_gain(pdm,edm),"comparison_test_projects":int(c.canonical_project_id.nunique()),"comparison_test_snapshots":len(c)};return {"experiment":{"experiment_id":EXPERIMENT_ID,"seed":SEED,"diagnostics":{"cost":cd,"delay":dd}},"overall_comparison":m}
def main():
 p=argparse.ArgumentParser();p.add_argument("--start",type=int,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--test-end",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args();data,_=build_training_dataset();data=data.copy();data["completion_year"]=pd.to_numeric(data.completion_year,errors="coerce");b,r=fast_current_production(data,a.start,a.end,a.test_end);z=fit_against_production(data=data,training_start=a.start,training_end=a.end,test_end=a.test_end,production_bundle=b,production_receipt=r);m=z["overall_comparison"];v={k:("IMPROVED" if m[f"{k}_improvement_percentage"]>0 else "REGRESSED" if m[f"{k}_improvement_percentage"]<0 else "UNCHANGED") for k in ("cost","delay")};payload={"experiment":EXPERIMENT_ID,"window":f"{a.start}-{a.end}","production":{"cost_mae":m["production_cost_mae"],"delay_mae":m["production_delay_mae"]},"experiment_metrics":{"cost_mae":m["experiment_cost_mae"],"delay_mae":m["experiment_delay_mae"]},"improvement":{"cost_percent":m["cost_improvement_percentage"],"delay_percent":m["delay_improvement_percentage"]},"cohort":{"projects":m["comparison_test_projects"],"snapshots":m["comparison_test_snapshots"]},"diagnostics":z["experiment"]["diagnostics"],"verdict":v};o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");
 for k,val in [("WINDOW",f"{a.start}_{a.end}"),("PRODUCTION_COST_MAE",m["production_cost_mae"]),("EXPERIMENT_COST_MAE",m["experiment_cost_mae"]),("COST_IMPROVEMENT_PERCENT",m["cost_improvement_percentage"]),("PRODUCTION_DELAY_MAE",m["production_delay_mae"]),("EXPERIMENT_DELAY_MAE",m["experiment_delay_mae"]),("DELAY_IMPROVEMENT_PERCENT",m["delay_improvement_percentage"]),("PROJECT_COUNT",m["comparison_test_projects"]),("SNAPSHOT_COUNT",m["comparison_test_snapshots"]),("VERDICT_COST",v["cost"]),("VERDICT_DELAY",v["delay"])]:print(f"{k}={val}")
if __name__=="__main__":main()
