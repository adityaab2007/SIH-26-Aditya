"""Train cost and schedule forecasts with a project-level, time-based split."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from backend.app.ml.features import CATEGORICAL_COLUMNS, TEMPORAL_FEATURES, engineer_temporal_features, load_project_history
    from backend.app.ml.forward_labels import build_forward_labels
else:
    from .features import CATEGORICAL_COLUMNS, TEMPORAL_FEATURES, engineer_temporal_features, load_project_history
    from .forward_labels import build_forward_labels

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"
METRICS_PATH = MODELS / "model_metrics.json"


def preprocessor() -> ColumnTransformer:
    numeric = [x for x in TEMPORAL_FEATURES if x not in CATEGORICAL_COLUMNS]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", Pipeline([( "imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_COLUMNS),
    ])


def models(seed: int = 42) -> dict:
    return {
        "xgboost": XGBRegressor(n_estimators=180, max_depth=3, learning_rate=.05, subsample=.9, colsample_bytree=.9, objective="reg:squarederror", random_state=seed, n_jobs=2),
        "random_forest": RandomForestRegressor(n_estimators=240, min_samples_leaf=3, random_state=seed, n_jobs=2),
        "catboost": CatBoostRegressor(iterations=220, depth=4, learning_rate=.05, loss_function="RMSE", verbose=False, random_seed=seed, allow_writing_files=False),
    }


def score(y, prediction) -> dict:
    return {"MAE": round(float(mean_absolute_error(y, prediction)), 3), "RMSE": round(float(math.sqrt(mean_squared_error(y, prediction))), 3), "R2": round(float(r2_score(y, prediction)), 4)}


def project_time_split(frame):
    """A project belongs to one temporal cohort, preventing project history leakage."""
    starts = frame.groupby("project_id")["planned_start_date"].min().dt.year
    train_ids = starts[starts <= 2023].index
    validation_ids = starts[(starts >= 2024) & (starts <= 2025)].index
    test_ids = starts[starts >= 2026].index
    splits = {"train": frame[frame.project_id.isin(train_ids)], "validation": frame[frame.project_id.isin(validation_ids)], "test": frame[frame.project_id.isin(test_ids)]}
    if any(part.empty for part in splits.values()):
        raise ValueError("Longitudinal data must contain projects starting before 2024, in 2024-25, and from 2026 onward for the configured time split.")
    return splits


def fit_one(model, train, validation, test, target):
    pre = preprocessor()
    x_train = pre.fit_transform(train[TEMPORAL_FEATURES])
    x_val = pre.transform(validation[TEMPORAL_FEATURES])
    model.fit(x_train, train[target])
    validation_metrics = score(validation[target], model.predict(x_val))
    # Refit after selection only on historical train+validation projects; test stays untouched.
    combined = __import__("pandas").concat([train, validation], ignore_index=True)
    final_pre = preprocessor(); x_combined = final_pre.fit_transform(combined[TEMPORAL_FEATURES])
    final_model = model.__class__(**model.get_params())
    final_model.fit(x_combined, combined[target])
    test_metrics = score(test[target], final_model.predict(final_pre.transform(test[TEMPORAL_FEATURES])))
    return validation_metrics, test_metrics, {"preprocess": final_pre, "model": final_model, "features": TEMPORAL_FEATURES}


def local_importance(bundle):
    names = bundle["preprocess"].get_feature_names_out().tolist()
    values = np.asarray(bundle["model"].feature_importances_, dtype=float)
    pairs = sorted(zip(names, values), key=lambda x: abs(x[1]), reverse=True)[:12]
    total = sum(abs(v) for _, v in pairs) or 1
    return [{"feature": name.replace("num__", "").replace("cat__", ""), "importance": round(float(abs(value) / total), 4)} for name, value in pairs]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    history = load_project_history()
    labelled = build_forward_labels(engineer_temporal_features(history)).dropna(subset=["future_cost_escalation_percentage", "future_schedule_extension_days"])
    splits = project_time_split(labelled)
    report = {"metadata": {"dataset_rows": len(labelled), "dataset_kind": "deterministic synthetic longitudinal demonstration data", "split_strategy": "time-based, project-level planned-start-year split: train <=2023, validation 2024-2025, test >=2026", "leakage_policy": "features use snapshot/past values only; sector and agency outcomes use projects completed before snapshot month", "prediction_vs_actual": []}, "cost_model": {"candidates": {}}, "delay_model": {"candidates": {}}}
    registry, importances = {}, {}
    for name, target, output_name in [("cost_model", "future_cost_escalation_percentage", "cost_model.pkl"), ("delay_model", "future_schedule_extension_days", "delay_model.pkl")]:
        choices = []
        for model_name, model in models().items():
            validation, test, bundle = fit_one(model, splits["train"], splits["validation"], splits["test"], target)
            report[name]["candidates"][model_name] = {"validation": validation, "test": test}
            choices.append((validation["MAE"], model_name, bundle))
        _, best_name, best_bundle = min(choices, key=lambda item: item[0])
        joblib.dump(best_bundle, MODELS / output_name)
        report[name].update({"best_model": best_name, **report[name]["candidates"][best_name]["test"], "target": target})
        registry[f"{name}:best"] = {"model": best_name, "path": output_name, "features": TEMPORAL_FEATURES}
        importances[name] = local_importance(best_bundle)
    # A real held-out example for the validation screen, never a fabricated hard-coded claim.
    held = splits["test"].iloc[0]
    for name, target in [("cost_model", "future_cost_escalation_percentage"), ("delay_model", "future_schedule_extension_days")]:
        bundle = joblib.load(MODELS / ("cost_model.pkl" if name == "cost_model" else "delay_model.pkl"))
        pred = float(bundle["model"].predict(bundle["preprocess"].transform(held[TEMPORAL_FEATURES].to_frame().T))[0])
        report["metadata"]["prediction_vs_actual"].append({"project_id": str(held.project_id), "month": held.month.strftime("%Y-%m-%d"), "target": name, "predicted": round(pred, 2), "actual": round(float(held[target]), 2), "absolute_error": round(abs(pred - float(held[target])), 2)})
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    (MODELS / "registry.json").write_text(json.dumps(registry, indent=2))
    (MODELS / "global_feature_importance.json").write_text(json.dumps(importances, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved selected temporal models to {MODELS / 'cost_model.pkl'} and {MODELS / 'delay_model.pkl'}")


if __name__ == "__main__":
    main()
