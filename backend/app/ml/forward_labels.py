"""Future-outcome labels attached to an earlier project snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_forward_labels(snapshots: pd.DataFrame) -> pd.DataFrame:
    required = {"project_id", "month", "original_cost", "actual_cost", "planned_completion_date", "actual_completion_date"}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = snapshots.copy()
    for c in ["month", "planned_completion_date", "actual_completion_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df.actual_cost = pd.to_numeric(df.actual_cost, errors="coerce")
    df.original_cost = pd.to_numeric(df.original_cost, errors="coerce")
    # Never label a snapshot after its final outcome became known.
    usable = df[df.actual_completion_date > df.month].copy()
    usable["future_cost_escalation_percentage"] = np.where(
        usable.original_cost > 0,
        (usable.actual_cost - usable.original_cost) / usable.original_cost * 100,
        np.nan,
    )
    usable["future_schedule_extension_days"] = (usable.actual_completion_date - usable.planned_completion_date).dt.days
    return usable
