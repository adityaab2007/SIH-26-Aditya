"""Utilities for the full longitudinal SIH26103 forecasting stage.

This module is intentionally separate from the current baseline training script. It creates
future-horizon labels only when multiple monthly snapshots per project are available, so the
prototype never fabricates future labels from a single current snapshot.
"""
from __future__ import annotations

import pandas as pd


def build_forward_labels(snapshots: pd.DataFrame, horizon_months: int = 6) -> pd.DataFrame:
    required = {"project_code", "snapshot_date", "revised_end_date", "revised_cost_cr"}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = snapshots.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["revised_end_date"] = pd.to_datetime(df["revised_end_date"], errors="coerce")
    df = df.sort_values(["project_code", "snapshot_date"])
    output = []

    for code, group in df.groupby("project_code"):
        group = group.reset_index(drop=True)
        for _, current in group.iterrows():
            target_date = current["snapshot_date"] + pd.DateOffset(months=horizon_months)
            future = group[group["snapshot_date"] >= target_date]
            if future.empty:
                continue
            future_row = future.iloc[0]
            if pd.isna(current["revised_end_date"]) or pd.isna(future_row["revised_end_date"]):
                schedule_shift = None
            else:
                schedule_shift = (future_row["revised_end_date"] - current["revised_end_date"]).days
            if pd.isna(current["revised_cost_cr"]) or pd.isna(future_row["revised_cost_cr"]) or current["revised_cost_cr"] == 0:
                cost_shift = None
            else:
                cost_shift = (future_row["revised_cost_cr"] - current["revised_cost_cr"]) / current["revised_cost_cr"] * 100
            record = current.to_dict()
            record.update({
                "label_horizon_months": horizon_months,
                "future_snapshot_date": future_row["snapshot_date"],
                "future_schedule_shift_days": schedule_shift,
                "future_schedule_slip_90d": None if schedule_shift is None else int(schedule_shift > 90),
                "future_cost_escalation_pct": cost_shift,
                "future_cost_jump_5pct": None if cost_shift is None else int(cost_shift > 5),
            })
            output.append(record)
    return pd.DataFrame(output)
