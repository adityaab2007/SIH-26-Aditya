"""Exp137 — pairwise severity ranking with strict forward-OOF isotonic calibration."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from backend.app.ml.experiments.nextgen_common import _compare,_gain,_metric,_prepare
from backend.app.ml.experiments.path_oof_delay_exp34 import _rolling_folds
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import _preprocessor,temporal_project_split
from scripts.run_fast_current_experiment import fast_current_production
EXPERIMENT_ID="Exp137";SEED=13703;MAX_PAIRS=12000

def _balanced_rows(frame,target):
    x=frame.dropna(subset=[target]).sort_values(["canonical_project_id","snapshot_date"])
    pieces=[]
    for _,g in x.groupby("canonical_project_id",sort=False): pieces.append(g.iloc[np.unique([0,len(g)-1])])
    return pd.concat(pieces,ignore_index=True) if pieces else x.iloc[:0].copy()

def sample_pairs(frame,target,max_pairs=MAX_PAIRS,seed=SEED):
    x=_balanced_rows(frame,target);rng=np.random.default_rng(seed);pairs=[];n=len(x);attempts=0
    if n<2:return pairs
    while len(pairs)<max_pairs and attempts<max_pairs*20:
        i,j=rng.integers(0,n,size=2);attempts+=1
        if i==j or str(x.iloc[i].canonical_project_id)==str(x.iloc[j].canonical_project_id):continue
        yi,yj=float(x.iloc[i][target]),float(x.iloc[j][target])
        if not np.isfinite(yi) or not np.isfinite(yj) or abs(yi-yj)<1e-9:continue
        if rng.random()<.5:i,j=j,i;yi,yj=yj,yi
        pairs.append((int(i),int(j),int(yi>yj)))
    return x,pairs

def _dense(v):return v.toarray() if hasattr(v,"toarray") else np.asarray(v)

def fit_ranker(frame,features,target,seed=SEED,max_pairs=MAX_PAIRS):
    x,pairs=sample_pairs(frame,target,max_pairs,seed)
    if len(pairs)<100:raise ValueError("insufficient valid cross-project ranking pairs")
    prep=_preprocessor(x,features);z=_dense(prep.fit_transform(x[features]));scale=StandardScaler().fit(z);zs=scale.transform(z)
    d=np.vstack([zs[i]-zs[j] for i,j,_ in pairs]);y=np.asarray([lab for _,_,lab in pairs],int);model=LogisticRegression(max_iter=800,random_state=seed,C=0.3).fit(d,y)
    return {"preprocess":prep,"scale":scale,"model":model,"pair_count":len(pairs)}
def rank_scores(state,frame,features):
    z=_dense(state["preprocess"].transform(frame[features]));zs=state["scale"].transform(z);return np.asarray(zs@state["model"].coef_.reshape(-1),float)
def fit_oof_calibrator(train,features,target,seed=SEED):
    chunks=[];pairs=0
    for j,(fit,val,year) in enumerate(_rolling_folds(train)):
        state=fit_ranker(fit,features,target,seed+j);score=rank_scores(state,val,features);part=val[[target,"sample_weight","completion_year"]].copy();part["score"]=score;part["fold_year"]=year;chunks.append(part);pairs+=state["pair_count"]
    if not chunks:raise ValueError("no forward ranking OOF folds")
    oof=pd.concat(chunks,ignore_index=True).dropna(subset=[target,"score"]);cal=IsotonicRegression(out_of_bounds="clip").fit(oof.score.to_numpy(float),oof[target].to_numpy(float),sample_weight=oof.sample_weight.to_numpy(float));corr=float(pd.Series(oof.score).corr(pd.Series(oof[target]),method="spearman"));return cal,{"oof_rows":len(oof),"sampled_pairs":pairs,"spearman_rank_correlation":corr,"calibration_uses_oof_only":True}
def monotonic_calibration_check(cal,scores):
    p=np.asarray(cal.predict(np.sort(np.asarray(scores,float))),float);return bool(np.all(np.diff(p)>=-1e-10))
def _target(train,c,features,target,seed):
    cal,diag=fit_oof_calibrator(train,features,target,seed);state=fit_ranker(train,features,target,seed+50);score=rank_scores(state,c,features);pred=np.asarray(cal.predict(score),float);diag.update({"final_sampled_pairs":state["pair_count"],"monotonic_calibration":monotonic_calibration_check(cal,score)});return pred,diag
def fit_against_production(*,data,training_start,training_end,test_end,production_bundle,production_receipt=None):
    f=_prepare(data);train,test=temporal_project_split(f,training_start,training_end,test_end);c=_compare(test);meta=production_bundle["metadata"];cf=list(meta["cost_features_used"]);df=list(meta["delay_features_used"]);pc=np.asarray(production_bundle["cost"].predict(c[cf]),float);pdly=np.maximum(0,np.asarray(production_bundle["delay"].predict(c[df]),float));ec,cd=_target(train,c,cf,"actual_cost_overrun_percentage",SEED);ed,dd=_target(train,c,df,"actual_delay_days",SEED+100);ed=np.maximum(0,ed);pcm,ecm=_metric(c,"actual_cost_overrun_percentage",pc),_metric(c,"actual_cost_overrun_percentage",ec);pdm,edm=_metric(c,"actual_delay_days",pdly),_metric(c,"actual_delay_days",ed);m={"production_cost_mae":pcm,"experiment_cost_mae":ecm,"cost_improvement_percentage":_gain(pcm,ecm),"production_delay_mae":pdm,"experiment_delay_mae":edm,"delay_improvement_percentage":_gain(pdm,edm),"comparison_test_projects":int(c.canonical_project_id.nunique()),"comparison_test_snapshots":len(c)};return {"experiment":{"experiment_id":EXPERIMENT_ID,"seed":SEED,"diagnostics":{"cost":cd,"delay":dd}},"overall_comparison":m}
def main():
 p=argparse.ArgumentParser();p.add_argument("--start",type=int,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--test-end",type=int,required=True);p.add_argument("--output",required=True);a=p.parse_args();data,_=build_training_dataset();data=data.copy();data["completion_year"]=pd.to_numeric(data.completion_year,errors="coerce");b,r=fast_current_production(data,a.start,a.end,a.test_end);z=fit_against_production(data=data,training_start=a.start,training_end=a.end,test_end=a.test_end,production_bundle=b,production_receipt=r);m=z["overall_comparison"];v={k:("IMPROVED" if m[f"{k}_improvement_percentage"]>0 else "REGRESSED" if m[f"{k}_improvement_percentage"]<0 else "UNCHANGED") for k in ("cost","delay")};payload={"experiment":EXPERIMENT_ID,"window":f"{a.start}-{a.end}","production":{"cost_mae":m["production_cost_mae"],"delay_mae":m["production_delay_mae"]},"experiment_metrics":{"cost_mae":m["experiment_cost_mae"],"delay_mae":m["experiment_delay_mae"]},"improvement":{"cost_percent":m["cost_improvement_percentage"],"delay_percent":m["delay_improvement_percentage"]},"cohort":{"projects":m["comparison_test_projects"],"snapshots":m["comparison_test_snapshots"]},"diagnostics":z["experiment"]["diagnostics"],"verdict":v};o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");
 for k,val in [("WINDOW",f"{a.start}_{a.end}"),("PRODUCTION_COST_MAE",m["production_cost_mae"]),("EXPERIMENT_COST_MAE",m["experiment_cost_mae"]),("COST_IMPROVEMENT_PERCENT",m["cost_improvement_percentage"]),("PRODUCTION_DELAY_MAE",m["production_delay_mae"]),("EXPERIMENT_DELAY_MAE",m["experiment_delay_mae"]),("DELAY_IMPROVEMENT_PERCENT",m["delay_improvement_percentage"]),("PROJECT_COUNT",m["comparison_test_projects"]),("SNAPSHOT_COUNT",m["comparison_test_snapshots"]),("VERDICT_COST",v["cost"]),("VERDICT_DELAY",v["delay"])]:print(f"{k}={val}")
if __name__=="__main__":main()
