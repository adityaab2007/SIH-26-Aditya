from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from backend.app.ml.features import COST_FEATURES, SCHEDULE_FEATURES, load_and_engineer, save_processed
else:
    from .features import COST_FEATURES, SCHEDULE_FEATURES, load_and_engineer, save_processed

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"
METRICS_PATH = MODELS / "metrics.json"
REGISTRY_PATH = MODELS / "registry.json"
IMPORTANCE_PATH = MODELS / "global_feature_importance.json"


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    categorical = [c for c in feature_cols if c in {"sector", "ministry"}]
    numeric = [c for c in feature_cols if c not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop", verbose_feature_names_out=True)


def classifiers(seed: int = 42):
    return {
        "logistic_regression": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=seed),
        "xgboost": XGBClassifier(n_estimators=180, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=seed, n_jobs=2),
    }


def regressors(seed: int = 42):
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=seed, n_jobs=2),
        "xgboost": XGBRegressor(n_estimators=180, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=seed, n_jobs=2),
    }


def _clean_xy(df, features, target):
    subset = df[df[target].notna()].copy()
    return subset[features], subset[target], subset


def evaluate_classifier(model, X, y):
    pipe = Pipeline([("preprocess", make_preprocessor(list(X.columns))), ("model", model)])
    n_splits = max(2, min(5, int(pd.Series(y).value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = cross_val_predict(pipe, X, y.astype(int), cv=cv, method="predict", n_jobs=1)
    proba = cross_val_predict(pipe, X, y.astype(int), cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    m = {"rows": len(y), "folds": n_splits, "accuracy": round(float(accuracy_score(y,pred)),4), "f1": round(float(f1_score(y,pred,zero_division=0)),4), "roc_auc": round(float(roc_auc_score(y,proba)),4), "positive_rate": round(float(np.mean(y)),4)}
    pipe.fit(X, y.astype(int))
    return m, pipe


def evaluate_regressor(model, X, y):
    pipe = Pipeline([("preprocess", make_preprocessor(list(X.columns))), ("model", model)])
    n_splits = min(5, max(2, len(y)//12))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = cross_val_predict(pipe, X, y.astype(float), cv=cv, method="predict", n_jobs=1)
    m={"rows":len(y),"folds":n_splits,"mae":round(float(mean_absolute_error(y,pred)),4),"rmse":round(float(math.sqrt(mean_squared_error(y,pred))),4),"r2":round(float(r2_score(y,pred)),4)}
    pipe.fit(X, y.astype(float))
    return m, pipe


def evaluate_catboost_classifier(X,y):
    n_splits=max(2,min(5,int(pd.Series(y).value_counts().min())))
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
    pred=np.zeros(len(y),dtype=int); proba=np.zeros(len(y))
    for tr,te in cv.split(X,y.astype(int)):
        pre=make_preprocessor(list(X.columns)); xtr=pre.fit_transform(X.iloc[tr]); xte=pre.transform(X.iloc[te])
        model=CatBoostClassifier(iterations=220,depth=4,learning_rate=.04,verbose=False,random_seed=42,loss_function="Logloss",allow_writing_files=False)
        model.fit(xtr,y.iloc[tr].astype(int)); pred[te]=model.predict(xte).astype(int).reshape(-1); proba[te]=model.predict_proba(xte)[:,1]
    m={"rows":len(y),"folds":n_splits,"accuracy":round(float(accuracy_score(y,pred)),4),"f1":round(float(f1_score(y,pred,zero_division=0)),4),"roc_auc":round(float(roc_auc_score(y,proba)),4),"positive_rate":round(float(np.mean(y)),4)}
    pre=make_preprocessor(list(X.columns)); xt=pre.fit_transform(X); model=CatBoostClassifier(iterations=220,depth=4,learning_rate=.04,verbose=False,random_seed=42,loss_function="Logloss",allow_writing_files=False); model.fit(xt,y.astype(int))
    return m,{"preprocess":pre,"model":model}


def evaluate_catboost_regressor(X,y):
    n_splits=min(5,max(2,len(y)//12)); cv=KFold(n_splits=n_splits,shuffle=True,random_state=42); pred=np.zeros(len(y))
    for tr,te in cv.split(X):
        pre=make_preprocessor(list(X.columns)); xtr=pre.fit_transform(X.iloc[tr]); xte=pre.transform(X.iloc[te]); model=CatBoostRegressor(iterations=220,depth=4,learning_rate=.04,verbose=False,random_seed=42,loss_function="RMSE",allow_writing_files=False); model.fit(xtr,y.iloc[tr].astype(float)); pred[te]=model.predict(xte)
    m={"rows":len(y),"folds":n_splits,"mae":round(float(mean_absolute_error(y,pred)),4),"rmse":round(float(math.sqrt(mean_squared_error(y,pred))),4),"r2":round(float(r2_score(y,pred)),4)}
    pre=make_preprocessor(list(X.columns)); xt=pre.fit_transform(X); model=CatBoostRegressor(iterations=220,depth=4,learning_rate=.04,verbose=False,random_seed=42,loss_function="RMSE",allow_writing_files=False); model.fit(xt,y.astype(float)); return m,{"preprocess":pre,"model":model}


def importance(bundle):
    if isinstance(bundle,dict): pre=bundle["preprocess"]; model=bundle["model"]
    else: pre=bundle.named_steps["preprocess"]; model=bundle.named_steps["model"]
    names=list(pre.get_feature_names_out()); vals=np.asarray(model.feature_importances_ if hasattr(model,"feature_importances_") else np.abs(model.coef_).reshape(-1),dtype=float); pairs=sorted(zip(names,vals),key=lambda x:abs(x[1]),reverse=True)[:18]; total=sum(abs(v) for _,v in pairs) or 1
    return [{"feature":n.replace("num__","").replace("cat__",""),"importance":round(abs(float(v))/total,4)} for n,v in pairs]


def main():
    MODELS.mkdir(parents=True,exist_ok=True); df=load_and_engineer(); save_processed(df)
    metrics={"metadata":{"dataset_rows":len(df),"dataset_kind":"curated official PAIMANA May 2026 public-project subset","forecasting_note":"Real-data observed-overrun baselines; forward-horizon labels need the expanded monthly archive."},"schedule_classifier":{},"cost_classifier":{},"schedule_regressor":{},"cost_regressor":{}}
    registry={}; importances={}
    tasks=[("schedule_classifier",SCHEDULE_FEATURES,"schedule_overrun_90d",classifiers(),evaluate_classifier,evaluate_catboost_classifier), ("cost_classifier",COST_FEATURES,"cost_overrun_5pct",classifiers(),evaluate_classifier,evaluate_catboost_classifier), ("schedule_regressor",SCHEDULE_FEATURES,"schedule_extension_days",regressors(),evaluate_regressor,evaluate_catboost_regressor), ("cost_regressor",COST_FEATURES,"cost_escalation_pct",regressors(),evaluate_regressor,evaluate_catboost_regressor)]
    for task,features,target,models,evaluator,cat_eval in tasks:
        X,y,_=_clean_xy(df,features,target)
        for name,model in models.items():
            m,b=evaluator(model,X,y); path=MODELS/f"{task}_{name}.joblib"; joblib.dump(b,path); metrics[task][name]=m; registry[f"{task}:{name}"]={"path":path.name,"features":features}; importances[f"{task}:{name}"]=importance(b)
        m,b=cat_eval(X,y); path=MODELS/f"{task}_catboost.joblib"; joblib.dump(b,path); metrics[task]["catboost"]=m; registry[f"{task}:catboost"]={"path":path.name,"features":features,"artifact_type":"bundle"}; importances[f"{task}:catboost"]=importance(b)
    for task in ["schedule_classifier","cost_classifier"]:
        best=max([k for k in metrics[task] if k!="best_model"],key=lambda n:metrics[task][n].get("roc_auc") or -1); registry[f"{task}:best"]={"model":best,**registry[f"{task}:{best}"]}; metrics[task]["best_model"]=best
    for task in ["schedule_regressor","cost_regressor"]:
        best=min([k for k in metrics[task] if k!="best_model"],key=lambda n:metrics[task][n].get("mae",float("inf"))); registry[f"{task}:best"]={"model":best,**registry[f"{task}:{best}"]}; metrics[task]["best_model"]=best
    METRICS_PATH.write_text(json.dumps(metrics,indent=2)); REGISTRY_PATH.write_text(json.dumps(registry,indent=2)); IMPORTANCE_PATH.write_text(json.dumps(importances,indent=2)); print(json.dumps(metrics,indent=2)); print(f"Saved {len(list(MODELS.glob('*.joblib')))} trained model artifacts to {MODELS}")

if __name__=="__main__": main()
