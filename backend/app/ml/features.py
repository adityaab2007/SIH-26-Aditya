"""Leakage-safe features for longitudinal SIH26103 forecasting."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROJECT_HISTORY = ROOT / "data" / "project_history.csv"
CATEGORICAL_COLUMNS = ["sector", "ministry", "agency", "state"]
TEMPORAL_FEATURES = [
    "original_cost", "current_estimated_cost", "monthly_expenditure", "physical_progress_percentage",
    "monthly_cost_growth", "expenditure_velocity", "cost_revision_percentage", "progress_velocity",
    "progress_delay", "expected_vs_actual_progress", "schedule_slippage", "milestone_delay_rate",
    "sector_average_overrun", "agency_delay_history", *CATEGORICAL_COLUMNS,
]


def _as_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_project_history(path: Path | str = PROJECT_HISTORY) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"project_id": str})
    return _as_date(frame, ["month", "planned_start_date", "planned_completion_date", "revised_completion_date", "actual_completion_date"])


def engineer_temporal_features(history: pd.DataFrame) -> pd.DataFrame:
    """Engineer using only fields known at a snapshot or prior completed projects."""
    df = history.copy()
    df = _as_date(df, ["month", "planned_start_date", "planned_completion_date", "revised_completion_date", "actual_completion_date"])
    for col in ["original_cost", "current_estimated_cost", "actual_cost", "monthly_expenditure", "physical_progress_percentage", "milestone_delay_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["project_id", "month"]).reset_index(drop=True)
    grouped = df.groupby("project_id", group_keys=False)
    prior_cost = grouped["current_estimated_cost"].shift(1)
    prior_progress = grouped["physical_progress_percentage"].shift(1)
    prior_month = grouped["month"].shift(1)
    elapsed = ((df.month.dt.year - df.planned_start_date.dt.year) * 12 + (df.month.dt.month - df.planned_start_date.dt.month)).clip(lower=0)
    planned_duration = ((df.planned_completion_date.dt.year - df.planned_start_date.dt.year) * 12 + (df.planned_completion_date.dt.month - df.planned_start_date.dt.month)).replace(0, np.nan)
    month_gap = ((df.month.dt.year - prior_month.dt.year) * 12 + (df.month.dt.month - prior_month.dt.month)).replace(0, np.nan)
    df["monthly_cost_growth"] = np.where(prior_cost > 0, (df.current_estimated_cost - prior_cost) / prior_cost * 100, 0.0)
    df["expenditure_velocity"] = df.monthly_expenditure
    df["cost_revision_percentage"] = np.where(df.original_cost > 0, (df.current_estimated_cost - df.original_cost) / df.original_cost * 100, 0.0)
    df["progress_velocity"] = np.where(month_gap > 0, (df.physical_progress_percentage - prior_progress) / month_gap, 0.0)
    expected = (elapsed / planned_duration * 100).clip(upper=100)
    df["expected_vs_actual_progress"] = df.physical_progress_percentage - expected
    df["progress_delay"] = (-df.expected_vs_actual_progress).clip(lower=0)
    df["schedule_slippage"] = (df.revised_completion_date - df.planned_completion_date).dt.days.clip(lower=0).fillna(0)
    df["milestone_delay_rate"] = grouped["milestone_delay_days"].transform(lambda s: s.fillna(0).expanding().mean()).fillna(0)

    # Completion outcomes are visible only after the other project has finished.
    completed = df[df.actual_completion_date.notna()].copy()
    completed["final_overrun"] = np.where(completed.original_cost > 0, (completed.actual_cost - completed.original_cost) / completed.original_cost * 100, np.nan)
    completed["final_delay"] = (completed.actual_completion_date - completed.planned_completion_date).dt.days.clip(lower=0)
    sector_values, agency_values = [], []
    for row in df.itertuples(index=False):
        past = completed[completed.actual_completion_date < row.month]
        sector_values.append(past.loc[past.sector == row.sector, "final_overrun"].mean())
        agency_values.append(past.loc[past.agency == row.agency, "final_delay"].mean())
    df["sector_average_overrun"] = pd.Series(sector_values, index=df.index).fillna(0.0)
    df["agency_delay_history"] = pd.Series(agency_values, index=df.index).fillna(0.0)
    return df
