"""Compatibility layer for Cost-only Exp75-79 on current compound production models."""
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
from backend.app.ml.experiments import cost_residual_challenger_common as legacy
ChallengerConfig = legacy.ChallengerConfig

def _X(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=list(features))

def fit_challenger(config: ChallengerConfig, *, data: pd.DataFrame, training_start: int, training_end: int, test_end: int, production_bundle: dict, production_receipt: dict) -> dict:
    frame=legacy.enrich_supervised_for_production(data.copy()); frame["completion_year"]=pd.to_numeric(frame.completion_year,errors="coerce")
    train,test=legacy.temporal_project_split(frame,int(training_start),int(training_end),int(test_end)); test=legacy._project_balanced(test)
    contract=legacy.target_feature_contract(production_bundle.get("metadata") or {}); cost_features=list(contract.get("cost") or production_receipt.get("features_used") or []); delay_features=list(contract.get("delay") or production_receipt.get("features_used") or [])
    if not cost_features or not delay_features: raise ValueError("Production target feature contract unavailable.")
    algorithm=legacy._algorithm(production_bundle,production_receipt); production_cost=np.asarray(production_bundle["cost"].predict(_X(test,cost_features)),dtype=float); production_delay=np.maximum(0.0,np.asarray(production_bundle["delay"].predict(_X(test,delay_features)),dtype=float))
    if config.strategy=="uncertainty_shrinkage":
        policy=legacy._fit_exp79(train,cost_features,algorithm); disagreement=np.vstack([m.predict(_X(test,cost_features)) for m in policy["models"]]).std(axis=0); experiment_cost=production_cost.copy(); high=disagreement>=policy["threshold"]; experiment_cost[high]=(1.0-policy["alpha"])*experiment_cost[high]+policy["alpha"]*policy["target_median"]; serializable_policy={k:v for k,v in policy.items() if k!="models"}
    else:
        oof=legacy._rolling_oof(train,cost_features,algorithm,max_folds=3)
        if config.strategy=="revised_cost_reliability": policy=legacy._fit_exp75(oof); correction=legacy._apply_exp75(test,policy)
        elif config.strategy=="medium_project_specialist": policy=legacy._fit_exp76(oof); correction=legacy._apply_exp76(test,policy)
        elif config.strategy=="early_financial_surrogate": policy=legacy._fit_exp77(oof); correction=legacy._apply_exp77(test,policy)
        elif config.strategy=="cross_window_consensus": policy=legacy._fit_exp78(oof); correction=legacy._apply_exp78(test,policy,production_cost)
        else: raise ValueError(f"Unknown challenger strategy: {config.strategy}")
        experiment_cost=production_cost+correction; serializable_policy=policy
    experiment_delay=production_delay.copy(); test["production_cost"]=production_cost; test["experiment_cost"]=experiment_cost; test["production_delay"]=production_delay; test["experiment_delay"]=experiment_delay
    pcm=legacy._metric(test,legacy.TARGET,"production_cost"); ecm=legacy._metric(test,legacy.TARGET,"experiment_cost"); pdm=legacy._metric(test,legacy.DELAY_TARGET,"production_delay"); edm=legacy._metric(test,legacy.DELAY_TARGET,"experiment_delay"); ci=legacy._improvement(pcm["MAE"],ecm["MAE"]); di=legacy._improvement(pdm["MAE"],edm["MAE"])
    pc=legacy.paired_project_mae_comparison(test,actual=legacy.TARGET,baseline_prediction="production_cost",candidate_prediction="experiment_cost"); pdly=legacy.paired_project_mae_comparison(test,actual=legacy.DELAY_TARGET,baseline_prediction="production_delay",candidate_prediction="experiment_delay",seed=26104)
    state={"strategy":config.strategy,"policy":policy,"serializable_policy":serializable_policy,"cost_features":cost_features,"delay_features":delay_features,"production_cost_model":production_bundle["cost"],"production_delay_model":production_bundle["delay"]}; rid=f"{config.experiment_id}-{training_start}-{training_end}-{uuid.uuid4().hex[:10]}"
    overall={"production_cost_mae":pcm["MAE"],"experiment_cost_mae":ecm["MAE"],"cost_improvement_percentage":round(ci,6) if ci is not None else None,"production_delay_mae":pdm["MAE"],"experiment_delay_mae":edm["MAE"],"delay_improvement_percentage":round(di,6) if di is not None else None,"comparison_test_projects":int(test.canonical_project_id.nunique()),"comparison_test_snapshots":int(len(test)),"paired_project_cost_comparison":pc,"paired_project_delay_comparison":pdly,"cost_only_delay_predictions_identical":bool(np.array_equal(production_delay,experiment_delay)),"production_wrapper_feature_policy":"reindex_to_persisted_contract","verdict":legacy._verdict(ci,di)}
    experiment={"experiment_id":config.experiment_id,"experiment_name":config.name,"scope":"cost","run_id":rid,"model_role":"experiment","promotion_allowed":False,"strategy":config.strategy,"training_only_policy":serializable_policy,"future_holdout_used_for_selection":False,"delay_mode":"production_control"}
    return {"experiment":experiment,"overall_comparison":overall,"runtime_state":state}

def filter_rows(held: pd.DataFrame,runtime_state:dict)->pd.DataFrame: return held.copy()

def predict_row(row:pd.DataFrame,runtime_state:dict)->dict:
    if not isinstance(row,pd.DataFrame): row=pd.DataFrame([row])
    cf=runtime_state["cost_features"]; df=runtime_state["delay_features"]; pc=np.asarray(runtime_state["production_cost_model"].predict(_X(row,cf)),dtype=float); pdly=np.maximum(0.0,np.asarray(runtime_state["production_delay_model"].predict(_X(row,df)),dtype=float)); s=runtime_state["strategy"]; p=runtime_state["policy"]
    if s=="revised_cost_reliability": cc=pc+legacy._apply_exp75(row,p)
    elif s=="medium_project_specialist": cc=pc+legacy._apply_exp76(row,p)
    elif s=="early_financial_surrogate": cc=pc+legacy._apply_exp77(row,p)
    elif s=="cross_window_consensus": cc=pc+legacy._apply_exp78(row,p,pc)
    elif s=="uncertainty_shrinkage":
        d=np.vstack([m.predict(_X(row,cf)) for m in p["models"]]).std(axis=0); cc=pc.copy(); h=d>=p["threshold"]; cc[h]=(1.0-p["alpha"])*cc[h]+p["alpha"]*p["target_median"]
    else: raise ValueError(s)
    return {"predicted_cost_overrun":float(cc[0]),"predicted_delay_days":float(pdly[0])}
