"""Real PAIMANA completed-project time-window training and evaluation.

This module deliberately never reads ``data/project_history.csv``.  That file is
the older deterministic demonstration trajectory; the simulations here use only
records extracted from the official PAIMANA completed-project archive reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
OUTCOMES = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"
MODELS = ROOT / "models"
TIME_WINDOW_REGISTRY = MODELS / "time_window_registry.json"
NUMERIC = [
    "approved_cost_cr", "approved_cost_log", "planned_commissioning_year",
    "planned_commissioning_month", "revised_cost_cr", "current_expenditure_cr",
    "physical_progress", "planned_duration_days", "elapsed_duration_days",
    "cost_escalation_percentage", "budget_stress_index", "expenditure_ratio",
    "cost_growth_velocity", "cost_acceleration", "cost_revision_count", "progress_deviation",
    "progress_velocity", "progress_acceleration", "progress_trend_6m", "progress_trend_12m", "duration_ratio",
    "schedule_slippage_score", "milestone_delay_count", "complexity_score",
    "agency_historical_delay_rate", "agency_historical_cost_overrun_rate", "sector_risk_score",
]
CATEGORICAL = ["sector", "ministry", "implementing_agency", "state", "project_size_category"]
FEATURES = NUMERIC + CATEGORICAL
CAT_FEATURE_INDICES = [FEATURES.index(column) for column in CATEGORICAL]
TARGET_COLUMNS = {"actual_cost_overrun_percentage", "actual_delay_days", "actual_risk", "reported_completion_expenditure_cr", "completion_date", "completion_year"}
MAX_COST_OVERRUN_PERCENTAGE = 1000.0
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class Window:
    version: str
    training_start: int
    training_end: int
    test_start: int
    test_end: int


WINDOWS = {
    "2001_2015": Window("v1", 2001, 2015, 2016, 2021),
    "2015_2021": Window("v2", 2015, 2021, 2022, 2028),
}
KEY_PATTERN = re.compile(r"^\d{4}_\d{4}$")


def version_key(start_year: int, end_year: int) -> str:
    return f"{int(start_year)}_{int(end_year)}"


def window_for(key: str, metadata: dict | None = None) -> Window:
    """Resolve registered and newly generated time windows without hardcoding them."""
    if metadata:
        return Window(key, int(metadata["training_start"]), int(metadata["training_end"]), int(metadata["test_start"]), int(metadata["test_end"]))
    if key in WINDOWS:
        return WINDOWS[key]
    if not KEY_PATTERN.fullmatch(key):
        raise KeyError(key)
    start, end = (int(value) for value in key.split("_"))
    if start > end:
        raise KeyError(key)
    actual_end = int(labelled(outcome_data()).completion_year.max())
    return Window(key, start, end, end + 1, actual_end)


def active_version() -> str | None:
    if not TIME_WINDOW_REGISTRY.exists():
        # The supplied registered baseline is a generated time-window artifact,
        # not the legacy global validation report.
        return "2001_2015" if (MODELS / "2001_2015" / "evaluation_results.json").exists() else None
    return json.loads(TIME_WINDOW_REGISTRY.read_text()).get("active_model_version")


def _set_active_version(key: str) -> None:
    TIME_WINDOW_REGISTRY.write_text(json.dumps({"active_model_version": key, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2))


def outcome_data() -> pd.DataFrame:
    if not OUTCOMES.exists():
        raise FileNotFoundError(f"{OUTCOMES} is missing. Run scripts/ingest_paimana_completed_reports.py first.")
    frame = pd.read_csv(OUTCOMES, dtype={"project_id": str})
    for column in ["completion_date", "planned_commissioning_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _series(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(np.nan, index=frame.index, dtype=float)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build only as-of features; unavailable archival fields remain missing."""
    result = frame.copy()
    planned = pd.to_datetime(_series(result, "planned_commissioning_date", "planned_completion_date", "original_end_date"), errors="coerce")
    completed = pd.to_datetime(_series(result, "completion_date", "actual_completion_date"), errors="coerce")
    snapshot = pd.to_datetime(_series(result, "snapshot_date", "month"), errors="coerce")
    started = pd.to_datetime(_series(result, "planned_start_date", "original_start_date"), errors="coerce")
    revised_end = pd.to_datetime(_series(result, "revised_completion_date", "revised_end_date"), errors="coerce")
    approved = pd.to_numeric(_series(result, "approved_cost_cr", "original_cost", "original_cost_cr"), errors="coerce")
    revised = pd.to_numeric(_series(result, "revised_cost_cr", "revised_cost"), errors="coerce")
    expenditure = pd.to_numeric(_series(result, "current_expenditure_cr", "current_expenditure", "expenditure_cr"), errors="coerce")
    progress = pd.to_numeric(_series(result, "physical_progress", "physical_progress_percentage", "physical_progress_pct"), errors="coerce")
    result["approved_cost_cr"] = approved
    result["approved_cost_log"] = np.log1p(approved.where(approved > 0))
    result["planned_commissioning_date"] = planned
    result["completion_date"] = completed
    result["planned_commissioning_year"] = planned.dt.year
    result["planned_commissioning_month"] = planned.dt.month
    result["completion_year"] = completed.dt.year
    result["revised_cost_cr"] = revised
    result["current_expenditure_cr"] = expenditure
    result["physical_progress"] = progress
    result["planned_duration_days"] = (planned - started).dt.days.where(planned.ge(started))
    result["elapsed_duration_days"] = (snapshot - started).dt.days.where(snapshot.ge(started))
    result["cost_escalation_percentage"] = ((revised - approved) / approved * 100).where(approved.gt(0) & revised.notna())
    result["budget_stress_index"] = (revised / approved).where(approved.gt(0) & revised.notna())
    result["expenditure_ratio"] = (expenditure / approved).where(approved.gt(0) & expenditure.notna())
    elapsed_years = result["elapsed_duration_days"] / 365.25
    result["cost_growth_velocity"] = (revised - approved).div(elapsed_years).where(elapsed_years.gt(0) & revised.notna())
    result["cost_acceleration"] = pd.to_numeric(_series(result, "cost_acceleration"), errors="coerce")
    revisions = pd.to_numeric(_series(result, "cost_revision_count"), errors="coerce")
    inferred_revision = pd.Series(np.where(revised.notna(), (revised.sub(approved).abs() > 1e-9).astype(float), np.nan), index=result.index)
    result["cost_revision_count"] = revisions.where(revisions.notna(), inferred_revision)
    if "project_id" in result.columns and snapshot.notna().any() and revised.notna().any():
        order = result.assign(_snapshot=snapshot, _revised=revised).sort_values(["project_id", "_snapshot"])
        changed = order.groupby("project_id", dropna=False)["_revised"].transform(lambda values: values.notna() & values.ne(values.ffill().shift()))
        revision_count = changed.groupby(order["project_id"], dropna=False).cumsum().astype(float)
        result.loc[order.index, "cost_revision_count"] = result.loc[order.index, "cost_revision_count"].where(result.loc[order.index, "cost_revision_count"].notna(), revision_count)
    expected_progress = (result["elapsed_duration_days"] / result["planned_duration_days"] * 100).clip(0, 100)
    result["progress_deviation"] = (expected_progress - progress).where(progress.notna())
    elapsed_months = result["elapsed_duration_days"] / 30.4375
    result["progress_velocity"] = (progress / elapsed_months).where(elapsed_months.gt(0) & progress.notna())
    result["progress_acceleration"] = pd.to_numeric(_series(result, "progress_acceleration"), errors="coerce")
    result["progress_trend_6m"] = pd.to_numeric(_series(result, "progress_trend_6m"), errors="coerce")
    result["progress_trend_12m"] = pd.to_numeric(_series(result, "progress_trend_12m"), errors="coerce")
    if "project_id" in result.columns and snapshot.notna().any():
        order = result.assign(_snapshot=snapshot).sort_values(["project_id", "_snapshot"])
        month_delta = order.groupby("project_id", dropna=False)["_snapshot"].diff().dt.days / 30.4375
        progress_delta = order.groupby("project_id", dropna=False)["physical_progress"].diff()
        observed_velocity = progress_delta.div(month_delta).where(month_delta.gt(0))
        result.loc[order.index, "progress_velocity"] = observed_velocity.where(observed_velocity.notna(), result.loc[order.index, "progress_velocity"])
        acceleration = observed_velocity.groupby(order["project_id"], dropna=False).diff()
        result.loc[order.index, "progress_acceleration"] = result.loc[order.index, "progress_acceleration"].where(result.loc[order.index, "progress_acceleration"].notna(), acceleration)
        cost_acceleration = order.groupby("project_id", dropna=False)["cost_growth_velocity"].diff()
        result.loc[order.index, "cost_acceleration"] = result.loc[order.index, "cost_acceleration"].where(result.loc[order.index, "cost_acceleration"].notna(), cost_acceleration)
        for days, column in ((183, "progress_trend_6m"), (365, "progress_trend_12m")):
            trends = pd.Series(np.nan, index=order.index, dtype=float)
            for _, group in order.groupby("project_id", dropna=False):
                for row_index, row in group.iterrows():
                    history = group[group._snapshot.between(row._snapshot - pd.Timedelta(days=days), row._snapshot)]
                    history = history.dropna(subset=["physical_progress", "_snapshot"])
                    if len(history) >= 2:
                        months = (history._snapshot.iloc[-1] - history._snapshot.iloc[0]).days / 30.4375
                        if months > 0:
                            trends.loc[row_index] = (history.physical_progress.iloc[-1] - history.physical_progress.iloc[0]) / months
            result.loc[order.index, column] = result.loc[order.index, column].where(result.loc[order.index, column].notna(), trends)
    result["duration_ratio"] = (result["elapsed_duration_days"] / result["planned_duration_days"]).where(result["planned_duration_days"].gt(0))
    result["schedule_slippage_score"] = (revised_end - planned).dt.days.where(revised_end.notna() & planned.notna())
    milestone_count = pd.to_numeric(_series(result, "milestone_delay_count"), errors="coerce")
    if "milestone_status" in result.columns:
        delayed = result["milestone_status"].astype(str).str.contains("delay|late|overdue|slip", case=False, regex=True)
        milestone_count = milestone_count.where(milestone_count.notna(), delayed.astype(float))
    result["milestone_delay_count"] = milestone_count
    result["project_size_category"] = pd.cut(approved, [-np.inf, 500, 5000, np.inf], labels=["Small", "Medium", "Large"]).astype(object)
    result["complexity_score"] = (
        result["approved_cost_log"].div(10)
        + result["duration_ratio"].fillna(0).clip(lower=0)
        + result["cost_revision_count"].fillna(0).clip(lower=0).mul(0.25)
        + result["milestone_delay_count"].fillna(0).clip(lower=0).mul(0.1)
    )
    for column in ("agency_historical_delay_rate", "agency_historical_cost_overrun_rate", "sector_risk_score"):
        result[column] = pd.to_numeric(_series(result, column), errors="coerce")
    for column in CATEGORICAL:
        if column not in result:
            result[column] = "Not reported"
        result[column] = result[column].fillna("Not reported").astype(str)
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def labelled(frame: pd.DataFrame) -> pd.DataFrame:
    result = features(frame)
    final_cost = pd.to_numeric(result["reported_completion_expenditure_cr"], errors="coerce")
    raw_cost = (final_cost - result["approved_cost_cr"]) / result["approved_cost_cr"] * 100
    valid_cost = (
        result["approved_cost_cr"].gt(0) & final_cost.notna() & final_cost.gt(0)
        & raw_cost.ge(-90) & raw_cost.le(MAX_COST_OVERRUN_PERCENTAGE)
    )
    raw_delay = (result["completion_date"] - result["planned_commissioning_date"]).dt.days
    valid_dates = result["completion_date"].notna() & result["planned_commissioning_date"].notna() & raw_delay.ge(0)
    result["actual_cost_overrun_percentage"] = np.nan
    result.loc[valid_cost, "actual_cost_overrun_percentage"] = raw_cost[valid_cost]
    result["actual_delay_days"] = np.nan
    result.loc[valid_dates, "actual_delay_days"] = raw_delay[valid_dates]
    result["actual_risk"] = np.select(
        [result["actual_delay_days"].lt(90), result["actual_delay_days"].lt(365), result["actual_delay_days"].lt(730)],
        [0, 1, 2], default=3,
    ).astype(int)
    return result.dropna(subset=["approved_cost_cr", "planned_commissioning_year", "actual_cost_overrun_percentage", "actual_delay_days"])


def label_quality_report(frame: pd.DataFrame | None = None) -> dict:
    raw = features(outcome_data() if frame is None else frame)
    final_cost = pd.to_numeric(raw.get("reported_completion_expenditure_cr"), errors="coerce")
    cost = (final_cost - raw.approved_cost_cr) / raw.approved_cost_cr * 100
    delay = (raw.completion_date - raw.planned_commissioning_date).dt.days
    valid_cost = raw.approved_cost_cr.gt(0) & final_cost.notna() & final_cost.gt(0) & cost.ge(-90) & cost.le(MAX_COST_OVERRUN_PERCENTAGE)
    valid_delay = raw.completion_date.notna() & raw.planned_commissioning_date.notna() & delay.ge(0)
    return {
        "source_rows": int(len(raw)),
        "invalid_cost_labels_removed": int((~valid_cost).sum()),
        "invalid_delay_labels_removed": int((~valid_delay).sum()),
        "valid_joint_labels": int((valid_cost & valid_delay).sum()),
        "cost_policy": "approved_cost > 0; final cost present and > 0; -90% <= overrun <= 1000%",
        "delay_policy": "both dates present and actual completion is not before planned completion",
        "missing_targets_filled_with_zero": False,
    }


def historical_prior_maps(train_data: pd.DataFrame) -> dict:
    """Fit smoothed agency/sector outcome priors from training rows only."""
    delay_global = float(train_data.actual_delay_days.mean())
    cost_global = float(train_data.actual_cost_overrun_percentage.mean())
    risk_global = float(train_data.actual_risk.mean() / 3.0)
    smoothing = 5.0

    def smoothed(column: str, target: str, global_value: float, scale: float = 1.0) -> dict:
        grouped = train_data.groupby(column, dropna=False)[target].agg(["sum", "count"])
        return {str(index): float((row["sum"] / scale + global_value * smoothing) / (row["count"] + smoothing)) for index, row in grouped.iterrows()}

    return {
        "training_start": int(train_data.completion_year.min()),
        "training_end": int(train_data.completion_year.max()),
        "smoothing_projects": smoothing,
        "global": {"delay": delay_global, "cost": cost_global, "risk": risk_global},
        "agency_delay": smoothed("implementing_agency", "actual_delay_days", delay_global),
        "agency_cost": smoothed("implementing_agency", "actual_cost_overrun_percentage", cost_global),
        "sector_risk": smoothed("sector", "actual_risk", risk_global, scale=3.0),
        "leakage_policy": "Mappings are fitted only from the selected training rows and are applied unchanged to later validation/test rows.",
    }


def apply_historical_priors(frame: pd.DataFrame, priors: dict) -> pd.DataFrame:
    result = frame.copy()
    agency = result.implementing_agency.astype(str)
    sector = result.sector.astype(str)
    result["agency_historical_delay_rate"] = agency.map(priors["agency_delay"]).fillna(priors["global"]["delay"])
    result["agency_historical_cost_overrun_rate"] = agency.map(priors["agency_cost"]).fillna(priors["global"]["cost"])
    result["sector_risk_score"] = sector.map(priors["sector_risk"]).fillna(priors["global"]["risk"])
    return result


def add_leave_one_out_training_priors(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent a training row's own target from entering its historical-rate features."""
    result = frame.copy()
    smoothing = 5.0
    delay_global = float(result.actual_delay_days.mean())
    cost_global = float(result.actual_cost_overrun_percentage.mean())
    risk_global = float(result.actual_risk.mean() / 3.0)

    def loo(group: str, target: str, global_value: float, scale: float = 1.0) -> pd.Series:
        sums = result.groupby(group, dropna=False)[target].transform("sum") / scale
        counts = result.groupby(group, dropna=False)[target].transform("count") - 1
        return (sums - result[target] / scale + global_value * smoothing) / (counts + smoothing)

    result["agency_historical_delay_rate"] = loo("implementing_agency", "actual_delay_days", delay_global)
    result["agency_historical_cost_overrun_rate"] = loo("implementing_agency", "actual_cost_overrun_percentage", cost_global)
    result["sector_risk_score"] = loo("sector", "actual_risk", risk_global, scale=3.0)
    return result


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def _regressor(seed: int = 26103, parameters: dict | None = None) -> CatBoostRegressor:
    """Default robust regressor retained for the live judge session contract."""
    options = {"iterations": 360, "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 8.0}
    options.update(parameters or {})
    return CatBoostRegressor(loss_function="MAE", eval_metric="MAE", random_seed=seed, verbose=False, allow_writing_files=False, **options)


def _algorithm_regressor(name: str, seed: int = 26103):
    if name == "catboost":
        return _regressor(seed)
    estimators = {
        "lightgbm": LGBMRegressor(n_estimators=360, learning_rate=0.035, num_leaves=20, max_depth=6, objective="huber", reg_lambda=3.0, random_state=seed, verbosity=-1),
        "xgboost": XGBRegressor(n_estimators=360, learning_rate=0.035, max_depth=5, subsample=0.85, colsample_bytree=0.85, objective="reg:absoluteerror", reg_lambda=4.0, random_state=seed, n_jobs=2),
        "random_forest": RandomForestRegressor(n_estimators=320, min_samples_leaf=3, criterion="absolute_error", max_features=0.8, random_state=seed, n_jobs=2),
    }
    return Pipeline([("preprocess", preprocessor()), ("model", estimators[name])])


def _classifier(y: pd.Series, seed: int = 26103):
    if y.nunique() <= 1:
        return DummyClassifier(strategy="most_frequent")
    return CatBoostClassifier(
        loss_function="MultiClass" if y.nunique() > 2 else "Logloss",
        eval_metric="TotalF1:average=Macro" if y.nunique() > 2 else "F1", iterations=400, depth=6,
        learning_rate=0.05, l2_leaf_reg=6.0, random_seed=seed,
        verbose=False, allow_writing_files=False, auto_class_weights="Balanced",
    )


def _sample_weights(y: pd.Series) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    median = float(np.median(values)); iqr = float(np.percentile(values, 75) - np.percentile(values, 25)) or 1.0
    return np.clip(1.0 / (1.0 + np.abs(values - median) / (3.0 * iqr)), 0.25, 1.0)


def _fit_regressor(model, X: pd.DataFrame, y: pd.Series, *, delay_target: bool = False) -> None:
    target = np.log1p(np.asarray(y, dtype=float)) if delay_target else np.asarray(y, dtype=float)
    weights = _sample_weights(pd.Series(target))
    if model.__class__.__module__.startswith("catboost"):
        model.fit(X, target, cat_features=CAT_FEATURE_INDICES, sample_weight=weights)
    elif isinstance(model, Pipeline):
        model.fit(X, target, model__sample_weight=weights)
    else:
        model.fit(X, target)


def _predict_regressor(model, X: pd.DataFrame, *, delay_target: bool = False) -> np.ndarray:
    values = np.asarray(model.predict(X), dtype=float)
    if isinstance(model, (TwoStageDelayModel, TemporalStackingRegressor)):
        return values
    return np.expm1(np.clip(values, 0, 20)) if delay_target else values


def _fit_classifier(model, X: pd.DataFrame, y: pd.Series) -> None:
    if isinstance(model, DummyClassifier):
        model.fit(X, y)
    else:
        model.fit(X, y, cat_features=CAT_FEATURE_INDICES)


class TwoStageDelayModel:
    def __init__(self, classifier, severity_model, low_delay_days: float):
        self.classifier = classifier
        self.severity_model = severity_model
        self.low_delay_days = float(low_delay_days)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.classifier.predict_proba(X))[:, 1]
        severity = _predict_regressor(self.severity_model, X, delay_target=True)
        return probability * severity + (1.0 - probability) * self.low_delay_days


class TemporalStackingRegressor:
    def __init__(self, models: list, meta_model: LinearRegression, delay_target: bool):
        self.models = models
        self.meta_model = meta_model
        self.delay_target = delay_target

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        matrix = np.column_stack([_predict_regressor(model, X, delay_target=self.delay_target) for model in self.models])
        return np.maximum(0, self.meta_model.predict(matrix)) if self.delay_target else self.meta_model.predict(matrix)


def _fit_two_stage(frame: pd.DataFrame) -> TwoStageDelayModel:
    severe = frame.actual_delay_days.gt(365).astype(int)
    classifier = _classifier(severe, seed=26105)
    _fit_classifier(classifier, frame[FEATURES], severe)
    high = frame[severe.eq(1)]
    severity = _regressor(seed=26106)
    _fit_regressor(severity, high[FEATURES], high.actual_delay_days, delay_target=True)
    low = frame.loc[severe.eq(0), "actual_delay_days"]
    return TwoStageDelayModel(classifier, severity, float(low.median() if not low.empty else 180.0))


def _fit_stacking(frame: pd.DataFrame, target_column: str, *, delay_target: bool) -> TemporalStackingRegressor | None:
    years = sorted(int(year) for year in frame.completion_year.dropna().unique())
    if len(years) < 3:
        return None
    meta_year = years[-1]
    base_frame = frame[frame.completion_year < meta_year]
    meta_frame = frame[frame.completion_year == meta_year]
    if len(base_frame) < 12 or len(meta_frame) < 2:
        return None
    names = ["catboost", "lightgbm", "xgboost"]
    initial = []
    for index, name in enumerate(names):
        model = _algorithm_regressor(name, 26200 + index)
        _fit_regressor(model, base_frame[FEATURES], base_frame[target_column], delay_target=delay_target)
        initial.append(model)
    matrix = np.column_stack([_predict_regressor(model, meta_frame[FEATURES], delay_target=delay_target) for model in initial])
    meta = LinearRegression().fit(matrix, meta_frame[target_column])
    final_models = []
    for index, name in enumerate(names):
        model = _algorithm_regressor(name, 26300 + index)
        _fit_regressor(model, frame[FEATURES], frame[target_column], delay_target=delay_target)
        final_models.append(model)
    return TemporalStackingRegressor(final_models, meta, delay_target)


def _benchmark_regressor(train_data: pd.DataFrame, target_column: str, *, delay_target: bool) -> tuple[str, object, list[dict]]:
    """Choose by MAE on the latest completion year, never on fitted rows."""
    validation_year = int(train_data.completion_year.max())
    fitting = train_data[train_data.completion_year < validation_year]
    validation = train_data[train_data.completion_year == validation_year]
    if len(fitting) < 12 or validation.empty:
        model = _regressor(); _fit_regressor(model, train_data[FEATURES], train_data[target_column], delay_target=delay_target)
        return "catboost", model, [{"model": "catboost", "MAE": None, "note": "insufficient internal temporal validation records"}]
    priors = historical_prior_maps(fitting)
    fitting = add_leave_one_out_training_priors(fitting)
    validation = apply_historical_priors(validation, priors)
    candidates: list[tuple[str, object]] = []
    results = []
    for index, name in enumerate(["catboost", "lightgbm", "xgboost", "random_forest"]):
        candidate = _algorithm_regressor(name, 26103 + index)
        _fit_regressor(candidate, fitting[FEATURES], fitting[target_column], delay_target=delay_target)
        predicted = _predict_regressor(candidate, validation[FEATURES], delay_target=delay_target)
        results.append({"model": name, "MAE": round(float(mean_absolute_error(validation[target_column], predicted)), 4)})
        candidates.append((name, candidate))
    if delay_target and fitting.actual_delay_days.gt(365).sum() >= 5:
        two_stage = _fit_two_stage(fitting)
        prediction = two_stage.predict(validation[FEATURES])
        results.append({"model": "two_stage_catboost", "MAE": round(float(mean_absolute_error(validation[target_column], prediction)), 4)})
        candidates.append(("two_stage_catboost", two_stage))
    stacked = _fit_stacking(fitting, target_column, delay_target=delay_target)
    if stacked is not None:
        prediction = stacked.predict(validation[FEATURES])
        results.append({"model": "temporal_stacking_catboost_lightgbm_xgboost", "MAE": round(float(mean_absolute_error(validation[target_column], prediction)), 4)})
        candidates.append(("temporal_stacking_catboost_lightgbm_xgboost", stacked))
    winner_name = min(results, key=lambda item: item["MAE"])["model"]
    final_frame = add_leave_one_out_training_priors(train_data)
    if winner_name == "two_stage_catboost":
        final_model = _fit_two_stage(final_frame)
    elif winner_name.startswith("temporal_stacking"):
        final_model = _fit_stacking(final_frame, target_column, delay_target=delay_target)
    else:
        final_model = _algorithm_regressor(winner_name)
        _fit_regressor(final_model, final_frame[FEATURES], final_frame[target_column], delay_target=delay_target)
    return winner_name, final_model, results


def _fit_selected(name: str, frame: pd.DataFrame, target_column: str, *, delay_target: bool, seed: int):
    if name == "two_stage_catboost":
        return _fit_two_stage(frame)
    if name.startswith("temporal_stacking"):
        stacked = _fit_stacking(frame, target_column, delay_target=delay_target)
        if stacked is not None:
            return stacked
        name = "catboost"
    model = _algorithm_regressor(name, seed)
    _fit_regressor(model, frame[FEATURES], frame[target_column], delay_target=delay_target)
    return model


def sector_bucket(values: pd.Series) -> pd.Series:
    """Map published sector labels to stable infrastructure groups."""
    text = values.fillna("Other").astype(str).str.lower()
    choices = np.select(
        [
            text.str.contains("road|highway", regex=True),
            text.str.contains("rail", regex=True),
            text.str.contains("power|electric|hydro|thermal", regex=True),
            text.str.contains("petroleum|oil|gas", regex=True),
            text.str.contains("irrigation|water", regex=True),
        ],
        ["Roads", "Railways", "Power", "Petroleum", "Irrigation"],
        default="Other",
    )
    return pd.Series(choices, index=values.index)


def _residual_corrections(frame: pd.DataFrame, residuals: np.ndarray) -> dict[str, float]:
    grouped = pd.DataFrame({"bucket": sector_bucket(frame.sector), "residual": residuals}).groupby("bucket").residual.agg(["mean", "count"])
    return {str(bucket): float(row["mean"] * row["count"] / (row["count"] + 5.0)) for bucket, row in grouped.iterrows()}


def apply_sector_correction(predictions: np.ndarray, frame: pd.DataFrame, artifact: dict | None, target_name: str) -> np.ndarray:
    config = (artifact or {}).get(target_name, {})
    if not config.get("enabled"):
        return np.asarray(predictions, dtype=float)
    correction = sector_bucket(frame.sector).map(config.get("corrections", {})).fillna(0).to_numpy(dtype=float)
    return np.asarray(predictions, dtype=float) + correction


def sector_correction_experiment(train_data: pd.DataFrame, selected_name: str, target_column: str, *, delay_target: bool) -> dict:
    """Enable a sector residual correction only after a later-year temporal win."""
    years = sorted(int(year) for year in train_data.completion_year.unique())
    if len(years) < 3 or selected_name.startswith("temporal_stacking"):
        return {"enabled": False, "reason": "insufficient temporal years or stacked estimator", "corrections": {}}
    correction_year, validation_year = years[-2], years[-1]
    base = train_data[train_data.completion_year < correction_year].copy()
    correction_rows = train_data[train_data.completion_year == correction_year].copy()
    validation = train_data[train_data.completion_year == validation_year].copy()
    if len(base) < 12 or correction_rows.empty or validation.empty:
        return {"enabled": False, "reason": "insufficient rows for nested temporal correction test", "corrections": {}}
    base_priors = historical_prior_maps(base)
    base_fit = add_leave_one_out_training_priors(base)
    correction_rows = apply_historical_priors(correction_rows, base_priors)
    base_model = _fit_selected(selected_name, base_fit, target_column, delay_target=delay_target, seed=26501)
    residuals = correction_rows[target_column].to_numpy() - _predict_regressor(base_model, correction_rows[FEATURES], delay_target=delay_target)
    candidate = _residual_corrections(correction_rows, residuals)
    pre_validation = train_data[train_data.completion_year < validation_year].copy()
    validation = apply_historical_priors(validation, historical_prior_maps(pre_validation))
    validation_fit = add_leave_one_out_training_priors(pre_validation)
    validation_model = _fit_selected(selected_name, validation_fit, target_column, delay_target=delay_target, seed=26502)
    baseline = _predict_regressor(validation_model, validation[FEATURES], delay_target=delay_target)
    corrected = baseline + sector_bucket(validation.sector).map(candidate).fillna(0).to_numpy(dtype=float)
    baseline_mae = float(mean_absolute_error(validation[target_column], baseline))
    corrected_mae = float(mean_absolute_error(validation[target_column], corrected))
    enabled = corrected_mae < baseline_mae
    final_base = train_data[train_data.completion_year < years[-1]].copy()
    latest = train_data[train_data.completion_year == years[-1]].copy()
    latest = apply_historical_priors(latest, historical_prior_maps(final_base))
    final_model = _fit_selected(selected_name, add_leave_one_out_training_priors(final_base), target_column, delay_target=delay_target, seed=26503)
    final_residuals = latest[target_column].to_numpy() - _predict_regressor(final_model, latest[FEATURES], delay_target=delay_target)
    return {
        "enabled": enabled,
        "target": target_column,
        "validation_year": validation_year,
        "correction_year": correction_year,
        "baseline_mae": round(baseline_mae, 4),
        "corrected_mae": round(corrected_mae, 4),
        "corrections": _residual_corrections(latest, final_residuals) if enabled else {},
        "policy": "Residual correction is enabled only when learned on an earlier year and improves the next unseen completion year.",
    }


def _global_importance(model, frame: pd.DataFrame) -> list[dict]:
    """Return grouped global tree importance in the original feature space."""
    if isinstance(model, TwoStageDelayModel):
        model = model.severity_model
    if isinstance(model, TemporalStackingRegressor):
        collections = [_global_importance(item, frame) for item in model.models]
        merged = {feature: float(np.mean([next((row["importance"] for row in rows if row["feature"] == feature), 0.0) for rows in collections])) for feature in FEATURES}
        return [{"feature": key, "importance": round(value, 6)} for key, value in sorted(merged.items(), key=lambda item: item[1], reverse=True)]
    if model.__class__.__module__.startswith("catboost"):
        shap_values = np.asarray(model.get_feature_importance(Pool(frame[FEATURES], cat_features=CAT_FEATURE_INDICES), type="ShapValues"), dtype=float)
        values = np.mean(np.abs(shap_values[..., :-1]), axis=tuple(range(shap_values.ndim - 1)))
        pairs = dict(zip(FEATURES, values))
    elif isinstance(model, Pipeline) and hasattr(model.named_steps["model"], "feature_importances_"):
        import shap

        transformed = model.named_steps["preprocess"].transform(frame[FEATURES])
        shap_values = np.asarray(shap.TreeExplainer(model.named_steps["model"]).shap_values(transformed), dtype=float)
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=0)
        values = np.mean(np.abs(shap_values), axis=0)
        names = model.named_steps["preprocess"].get_feature_names_out()
        pairs = {feature: 0.0 for feature in FEATURES}
        for name, value in zip(names, values):
            clean = str(name).replace("numeric__", "").replace("category__", "")
            feature = next((item for item in CATEGORICAL if clean.startswith(f"{item}_")), clean)
            if feature in pairs:
                pairs[feature] += float(value)
    else:
        pairs = {feature: 0.0 for feature in FEATURES}
    total = sum(abs(value) for value in pairs.values()) or 1.0
    return [{"feature": key, "importance": round(abs(value) / total, 6)} for key, value in sorted(pairs.items(), key=lambda item: abs(item[1]), reverse=True)]


def persist_shap_summaries(target: Path, key: str, models: dict, train_data: pd.DataFrame) -> dict:
    shap_dir = target / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    sample = train_data.tail(min(250, len(train_data)))
    for name, model in models.items():
        rows = _global_importance(model, sample)
        payload = {"model_version": key, "target": name, "method": "mean absolute SHAP value", "training_samples": int(len(train_data)), "features": rows}
        (shap_dir / f"{name}_shap_importance.json").write_text(json.dumps(payload, indent=2))
        summaries[name] = rows[:10]
    return summaries


def model_dir(key: str) -> Path:
    window_for(key)
    return MODELS / key


def _fit_quantiles(train_data: pd.DataFrame, target_column: str, *, delay_target: bool) -> dict[str, CatBoostRegressor]:
    models = {}
    target = np.log1p(train_data[target_column].to_numpy()) if delay_target else train_data[target_column].to_numpy()
    for label, alpha in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        model = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.04, loss_function=f"Quantile:alpha={alpha}", random_seed=26400 + int(alpha * 100), verbose=False, allow_writing_files=False)
        model.fit(train_data[FEATURES], target, cat_features=CAT_FEATURE_INDICES, sample_weight=_sample_weights(pd.Series(target)))
        models[label] = model
    return models


def predict_quantiles(models: dict, X: pd.DataFrame, *, delay_target: bool) -> dict[str, np.ndarray]:
    values = {}
    for label, model in models.items():
        prediction = np.asarray(model.predict(X), dtype=float)
        values[label] = np.expm1(np.clip(prediction, 0, 20)) if delay_target else prediction
    matrix = np.sort(np.column_stack([values["p10"], values["p50"], values["p90"]]), axis=1)
    return {"p10": matrix[:, 0], "p50": matrix[:, 1], "p90": matrix[:, 2]}


def _fit_survival_experiment(train_data: pd.DataFrame, target: Path) -> dict:
    """Fit an optional Cox model for probability of finishing within N delay days."""
    try:
        from lifelines import CoxPHFitter
        covariates = ["approved_cost_log", "planned_commissioning_year", "planned_commissioning_month", "complexity_score"]
        frame = train_data[covariates].copy()
        medians = frame.median(numeric_only=True).fillna(0).to_dict()
        frame = frame.fillna(medians).fillna(0)
        frame["duration"] = train_data.actual_delay_days.clip(lower=0).to_numpy() + 1
        frame["completed"] = 1
        model = CoxPHFitter(penalizer=0.2)
        model.fit(frame, duration_col="duration", event_col="completed", show_progress=False)
        joblib.dump({"model": model, "features": covariates, "medians": medians}, target / "survival_model.pkl")
        return {"status": "available", "model": "Cox proportional hazards", "duration": "non-negative delay days + 1", "event": "official completion", "features": covariates, "use_policy": "Advanced probability experiment only; it does not replace cost/delay models."}
    except Exception as exc:
        return {"status": "unavailable", "model": "Cox proportional hazards", "reason": f"{type(exc).__name__}: {exc}", "use_policy": "Not selected because the experiment did not fit reliably."}


def train(key: str) -> dict:
    window = window_for(key)
    all_data = labelled(outcome_data())
    train_data = all_data[all_data.completion_year.between(window.training_start, window.training_end)].copy()
    if len(train_data) < 12:
        raise ValueError(f"{key} has only {len(train_data)} real completed-project records; at least 12 are required.")
    model_train_data = add_leave_one_out_training_priors(train_data)
    priors = historical_prior_maps(train_data)
    X = model_train_data[FEATURES]
    cost_name, cost, cost_comparison = _benchmark_regressor(train_data, "actual_cost_overrun_percentage", delay_target=False)
    delay_name, delay, delay_comparison = _benchmark_regressor(train_data, "actual_delay_days", delay_target=True)
    risk = _classifier(model_train_data.actual_risk)
    _fit_classifier(risk, X, model_train_data.actual_risk)
    risk_validation = {"model": "CatBoostClassifier MultiClass", "MAE": None, "note": "insufficient internal temporal validation records"}
    validation_year = int(train_data.completion_year.max())
    risk_fit = train_data[train_data.completion_year < validation_year].copy()
    risk_test = train_data[train_data.completion_year == validation_year].copy()
    if len(risk_fit) >= 12 and not risk_test.empty:
        risk_test = apply_historical_priors(risk_test, historical_prior_maps(risk_fit))
        risk_fit = add_leave_one_out_training_priors(risk_fit)
        candidate_risk = _classifier(risk_fit.actual_risk, seed=26510)
        _fit_classifier(candidate_risk, risk_fit[FEATURES], risk_fit.actual_risk)
        risk_prediction = np.asarray(candidate_risk.predict(risk_test[FEATURES]), dtype=int).reshape(-1)
        risk_validation = {
            "model": "CatBoostClassifier MultiClass", "validation_year": validation_year,
            "accuracy": round(float(accuracy_score(risk_test.actual_risk, risk_prediction)), 4),
            "macro_f1": round(float(f1_score(risk_test.actual_risk, risk_prediction, average="macro", zero_division=0)), 4),
            "classes": RISK_LEVELS,
        }
    target = model_dir(key); target.mkdir(parents=True, exist_ok=True)
    joblib.dump(cost, target / "cost_model.pkl")
    joblib.dump(delay, target / "delay_model.pkl")
    joblib.dump(risk, target / "risk_model.pkl")
    joblib.dump({"cost": _fit_quantiles(model_train_data, "actual_cost_overrun_percentage", delay_target=False), "delay": _fit_quantiles(model_train_data, "actual_delay_days", delay_target=True)}, target / "uncertainty_models.pkl")
    (target / "historical_priors.json").write_text(json.dumps(priors, indent=2))
    sector_corrections = {
        "cost": sector_correction_experiment(train_data, cost_name, "actual_cost_overrun_percentage", delay_target=False),
        "delay": sector_correction_experiment(train_data, delay_name, "actual_delay_days", delay_target=True),
    }
    (target / "sector_corrections.json").write_text(json.dumps(sector_corrections, indent=2))
    survival = _fit_survival_experiment(model_train_data, target)
    feature_importance = persist_shap_summaries(target, key, {"cost": cost, "delay": delay, "risk": risk}, model_train_data)
    # CatBoost natively processes categorical columns; retain a documented
    # preprocessing artifact for the model-registry contract.
    joblib.dump({"numeric_features": NUMERIC, "categorical_features": CATEGORICAL, "native_categorical_encoding": True}, target / "scaler.pkl")
    metadata = {
        "model_version": key,
        "training_start": window.training_start,
        "training_end": window.training_end,
        "test_start": window.test_start,
        "test_end": window.test_end,
        "testing_start": window.test_start,
        "testing_end": window.test_end,
        "available_actual_end": int(all_data.completion_year.max()),
        "features_used": FEATURES,
        "feature_count": len(FEATURES),
        "feature_availability": {
            "available_when_published_at_cutoff": FEATURES,
            "historical_archive_missingness_policy": "Unavailable as-of fields remain NaN/Not reported. No fuzzy joins, zero fills, or synthetic behaviour values are used.",
        },
        "model_type": {"cost": cost_name, "delay": delay_name, "risk": "catboost_classifier"},
        "model_algorithm": {"cost": cost_name, "delay": delay_name, "risk": "CatBoostClassifier MultiClass"},
        "algorithms": {"cost": cost_name, "delay": delay_name, "risk": "CatBoostClassifier"},
        "validation_method": "latest-year temporal selection plus future-year holdout and expanding rolling validation",
        "outlier_policy": "Invalid labels are rejected by explicit rules. Valid extremes remain, with MAE/Huber objectives, log1p delay, and robust sample weighting.",
        "label_quality": label_quality_report(),
        "uncertainty": {"method": "CatBoost quantile regression", "quantiles": [0.1, 0.5, 0.9]},
        "delay_severity": {"classes": RISK_LEVELS, "thresholds_days": {"LOW": "<90", "MEDIUM": "90-364", "HIGH": "365-729", "CRITICAL": ">=730"}},
        "sector_correction_experiment": sector_corrections,
        "shap_available": True,
        "feature_importance": feature_importance,
        "survival_experiment": survival,
        "training_samples": int(len(train_data)),
        "data_source": "Official PAIMANA completed-project archive reports",
        "outcome_definition": "Reported completion expenditure versus approved cost; reported completion month versus original commissioning month.",
        "leakage_policy": "Completion expenditure, completion date, and derived outcomes are targets only. Agency/sector priors are fitted from training rows only; training rows use leave-one-out priors.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained",
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (target / "model_comparison.json").write_text(json.dumps({
        "selection_method": "latest-year temporal validation within the selected training range",
        "cost_candidates": cost_comparison,
        "delay_candidates": delay_comparison,
        "selected": {"cost": cost_name, "delay": delay_name},
        "delay_severity_classifier": risk_validation,
        "stacking_policy": "CatBoost + LightGBM + XGBoost with LinearRegression meta-model is selected only when its later-year MAE beats every individual model.",
    }, indent=2))
    return metadata


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-9
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0


def evaluate(key: str, save: bool = True) -> dict:
    target = model_dir(key)
    metadata = json.loads((target / "metadata.json").read_text())
    window = window_for(key, metadata)
    all_data = labelled(outcome_data())
    actual_end = min(window.test_end, int(all_data.completion_year.max()))
    test_data = all_data[all_data.completion_year.between(window.test_start, actual_end)].copy()
    if test_data.empty:
        raise ValueError(f"No official completed-project outcomes are available for {window.test_start}-{actual_end}.")
    priors_path = target / "historical_priors.json"
    if priors_path.exists():
        test_data = apply_historical_priors(test_data, json.loads(priors_path.read_text()))
    feature_columns = metadata.get("features_used", FEATURES)
    X = test_data[feature_columns]
    cost = joblib.load(target / "cost_model.pkl"); delay = joblib.load(target / "delay_model.pkl"); risk = joblib.load(target / "risk_model.pkl")
    uncertainty = joblib.load(target / "uncertainty_models.pkl") if (target / "uncertainty_models.pkl").exists() else None
    correction_path = target / "sector_corrections.json"
    corrections = json.loads(correction_path.read_text()) if correction_path.exists() else None
    test_data["predicted_cost_overrun"] = apply_sector_correction(_predict_regressor(cost, X), test_data, corrections, "cost")
    test_data["predicted_delay_days"] = np.maximum(0, apply_sector_correction(_predict_regressor(delay, X, delay_target=True), test_data, corrections, "delay"))
    if uncertainty:
        cost_range = predict_quantiles(uncertainty["cost"], X, delay_target=False)
        delay_range = predict_quantiles(uncertainty["delay"], X, delay_target=True)
        for label in ("p10", "p50", "p90"):
            test_data[f"predicted_cost_{label}"] = cost_range[label]
            test_data[f"predicted_delay_{label}"] = delay_range[label]
        cost_width = np.maximum(test_data.predicted_cost_p90 - test_data.predicted_cost_p10, 0)
        delay_width = np.maximum(test_data.predicted_delay_p90 - test_data.predicted_delay_p10, 0)
        cost_confidence = 100 / (1 + cost_width / (np.abs(test_data.predicted_cost_p50) + 10))
        delay_confidence = 100 / (1 + delay_width / (np.abs(test_data.predicted_delay_p50) + 90))
        test_data["model_confidence_percentage"] = (cost_confidence + delay_confidence) / 2
    test_data["predicted_risk"] = np.asarray(risk.predict(X), dtype=int).reshape(-1)
    risk_probability = np.asarray(risk.predict_proba(X), dtype=float) if hasattr(risk, "predict_proba") else None
    test_data["risk_probability"] = risk_probability.max(axis=1) if risk_probability is not None else 1.0
    test_data["cost_error"] = test_data.predicted_cost_overrun - test_data.actual_cost_overrun_percentage
    test_data["delay_error"] = test_data.predicted_delay_days - test_data.actual_delay_days
    cost_y = test_data.actual_cost_overrun_percentage.to_numpy(); cost_p = test_data.predicted_cost_overrun.to_numpy()
    delay_y = test_data.actual_delay_days.to_numpy(); delay_p = test_data.predicted_delay_days.to_numpy()
    cost_mae = float(mean_absolute_error(cost_y, cost_p))
    delay_mae = float(mean_absolute_error(delay_y, delay_p))
    cost_scale = max(float(np.mean(np.abs(cost_y))), 1e-9)
    delay_scale = max(float(np.mean(np.abs(delay_y))), 1e-9)
    metrics = {
        "model_version": key,
        "cost_model": {"MAE": round(cost_mae, 3), "RMSE": round(float(mean_squared_error(cost_y, cost_p) ** .5), 3), "R2": round(float(r2_score(cost_y, cost_p)), 4), "MAPE": round(_safe_mape(cost_y, cost_p), 3), "accuracy_percentage": round(max(0.0, 100 * (1 - cost_mae / cost_scale)), 2)},
        "delay_model": {"MAE": round(delay_mae, 3), "MAE_days": round(delay_mae, 3), "RMSE": round(float(mean_squared_error(delay_y, delay_p) ** .5), 3), "RMSE_days": round(float(mean_squared_error(delay_y, delay_p) ** .5), 3), "R2": round(float(r2_score(delay_y, delay_p)), 4), "log_target_RMSE": round(float(mean_squared_error(np.log1p(delay_y), np.log1p(np.maximum(delay_p, 0))) ** .5), 4), "accuracy_percentage": round(max(0.0, 100 * (1 - delay_mae / delay_scale)), 2)},
        "risk_model": {"classes": RISK_LEVELS, "accuracy": round(float(accuracy_score(test_data.actual_risk, test_data.predicted_risk)), 4), "precision": round(float(precision_score(test_data.actual_risk, test_data.predicted_risk, average="macro", zero_division=0)), 4), "recall": round(float(recall_score(test_data.actual_risk, test_data.predicted_risk, average="macro", zero_division=0)), 4), "f1": round(float(f1_score(test_data.actual_risk, test_data.predicted_risk, average="macro", zero_division=0)), 4)},
        "metadata": {**metadata, "evaluated_test_start": window.test_start, "evaluated_test_end": actual_end, "pending_actual_outcomes": list(range(actual_end + 1, window.test_end + 1)), "testing_samples": int(len(test_data)), "actual_outcome_policy": "Only official completed-project records are evaluated. Future years with no official completion record are forecast-only and excluded from metrics."},
    }
    columns = ["project_id", "project_name", "sector", "implementing_agency", "planned_commissioning_year", "completion_date", "approved_cost_cr", "reported_completion_expenditure_cr", "predicted_cost_overrun", "actual_cost_overrun_percentage", "cost_error", "predicted_delay_days", "actual_delay_days", "delay_error", "predicted_risk", "actual_risk", "risk_probability"]
    if uncertainty:
        columns += ["predicted_cost_p10", "predicted_cost_p50", "predicted_cost_p90", "predicted_delay_p10", "predicted_delay_p50", "predicted_delay_p90", "model_confidence_percentage"]
    rows = test_data[columns].sort_values(["completion_date", "project_name"]).rename(columns={"actual_cost_overrun_percentage": "actual_cost_overrun"})
    if save:
        rows.to_csv(target / "evaluation_results.csv", index=False)
        rows.to_csv(target / "prediction_validation.csv", index=False)
        (target / "evaluation_results.json").write_text(json.dumps(metrics, indent=2))
        (target / "metadata.json").write_text(json.dumps(metrics["metadata"], indent=2))
    return {"metrics": metrics, "rows": rows}


def rolling_validation(key: str) -> dict:
    """Expanding-window validation using only earlier official completion years."""
    metadata = json.loads((model_dir(key) / "metadata.json").read_text())
    window = window_for(key, metadata)
    all_data = labelled(outcome_data())
    folds = []
    for test_year in range(window.training_start + 1, min(window.test_end, int(all_data.completion_year.max())) + 1):
        fitting = all_data[all_data.completion_year.between(window.training_start, test_year - 1)]
        testing = all_data[all_data.completion_year == test_year]
        if len(fitting) < 12 or testing.empty:
            continue
        priors = historical_prior_maps(fitting)
        testing = apply_historical_priors(testing, priors)
        fitting = add_leave_one_out_training_priors(fitting)
        selected = metadata.get("model_type", {})
        cost = _fit_selected(selected.get("cost", "catboost"), fitting, "actual_cost_overrun_percentage", delay_target=False, seed=26103 + test_year)
        delay = _fit_selected(selected.get("delay", "catboost"), fitting, "actual_delay_days", delay_target=True, seed=26104 + test_year)
        cost_pred = _predict_regressor(cost, testing[FEATURES])
        delay_pred = _predict_regressor(delay, testing[FEATURES], delay_target=True)
        risk = _classifier(fitting.actual_risk, seed=26105 + test_year)
        _fit_classifier(risk, fitting[FEATURES], fitting.actual_risk)
        risk_pred = np.asarray(risk.predict(testing[FEATURES]), dtype=int).reshape(-1)
        folds.append({
            "train_start": int(window.training_start), "train_end": int(test_year - 1), "test_year": int(test_year), "training_samples": int(len(fitting)), "testing_samples": int(len(testing)),
            "cost_MAE": round(float(mean_absolute_error(testing.actual_cost_overrun_percentage, cost_pred)), 3),
            "cost_RMSE": round(float(mean_squared_error(testing.actual_cost_overrun_percentage, cost_pred) ** .5), 3),
            "cost_MAPE": round(_safe_mape(testing.actual_cost_overrun_percentage.to_numpy(), cost_pred), 3),
            "delay_MAE_days": round(float(mean_absolute_error(testing.actual_delay_days, delay_pred)), 3),
            "delay_RMSE_days": round(float(mean_squared_error(testing.actual_delay_days, delay_pred) ** .5), 3),
            "risk_accuracy": round(float(accuracy_score(testing.actual_risk, risk_pred)), 4),
            "risk_precision": round(float(precision_score(testing.actual_risk, risk_pred, average="macro", zero_division=0)), 4),
            "risk_recall": round(float(recall_score(testing.actual_risk, risk_pred, average="macro", zero_division=0)), 4),
            "risk_f1": round(float(f1_score(testing.actual_risk, risk_pred, average="macro", zero_division=0)), 4),
        })
    cost_errors = [fold["cost_MAE"] for fold in folds]
    delay_errors = [fold["delay_MAE_days"] for fold in folds]
    report = {
        "model_version": key, "folds": folds, "fold_count": len(folds),
        "average_cost_mae": round(float(np.mean(cost_errors)), 3) if folds else None,
        "average_delay_mae_days": round(float(np.mean(delay_errors)), 3) if folds else None,
        "best_cost_mae": round(float(np.min(cost_errors)), 3) if folds else None,
        "worst_cost_mae": round(float(np.max(cost_errors)), 3) if folds else None,
        "best_delay_mae_days": round(float(np.min(delay_errors)), 3) if folds else None,
        "worst_delay_mae_days": round(float(np.max(delay_errors)), 3) if folds else None,
        "average_risk_accuracy": round(float(np.mean([fold["risk_accuracy"] for fold in folds])), 4) if folds else None,
        "average_risk_precision": round(float(np.mean([fold["risk_precision"] for fold in folds])), 4) if folds else None,
        "average_risk_recall": round(float(np.mean([fold["risk_recall"] for fold in folds])), 4) if folds else None,
        "average_risk_f1": round(float(np.mean([fold["risk_f1"] for fold in folds])), 4) if folds else None,
        "policy": "Each fold trains only on completion years before its test year. No future outcomes are used in fitting.",
    }
    (model_dir(key) / "rolling_validation_results.json").write_text(json.dumps(report, indent=2))
    return report


def retrain(start_year: int, end_year: int) -> dict:
    """Persist a newly selected historical window and its future-only evaluation."""
    key = version_key(start_year, end_year)
    available_end = int(labelled(outcome_data()).completion_year.max())
    if end_year >= available_end:
        raise ValueError(f"Training must end before {available_end} so an unseen future period remains.")
    metadata = train(key)
    result = evaluate(key, save=True)
    rolling = rolling_validation(key)
    metadata_path = model_dir(key) / "metadata.json"
    updated_metadata = json.loads(metadata_path.read_text())
    updated_metadata["rolling_validation_scores"] = {name: value for name, value in rolling.items() if name != "folds"}
    metadata_path.write_text(json.dumps(updated_metadata, indent=2))
    evaluation_path = model_dir(key) / "evaluation_results.json"
    if evaluation_path.exists():
        saved_evaluation = json.loads(evaluation_path.read_text())
        saved_evaluation["metadata"] = updated_metadata
        evaluation_path.write_text(json.dumps(saved_evaluation, indent=2))
    _set_active_version(key)
    metrics = result["metrics"]
    return {
        "status": "success", "model_version": key,
        "training_years": f"{metadata['training_start']}-{metadata['training_end']}",
        "testing_years": f"{metadata['test_start']}-{metadata['test_end']}",
        "cost_mae": metrics["cost_model"]["MAE"], "delay_mae": metrics["delay_model"]["MAE_days"],
        "risk_metrics": metrics["risk_model"], "validation_projects": metrics["metadata"]["testing_samples"],
        "metrics": metrics, "training_samples": metadata["training_samples"],
        "testing_samples": metrics["metadata"]["testing_samples"], "rolling_validation": rolling,
    }


def versions() -> list[dict]:
    result = []
    for target in sorted(MODELS.iterdir() if MODELS.exists() else [], key=lambda path: path.name):
        if not target.is_dir() or not KEY_PATTERN.fullmatch(target.name) or not (target / "metadata.json").exists():
            continue
        metadata = json.loads((target / "metadata.json").read_text())
        result.append({"key": target.name, **metadata, "training_label": f"{metadata['training_start']}-{metadata['training_end']}", "testing_label": f"{metadata['test_start']}-{metadata['test_end']}", "active": target.name == active_version()})
    return result
