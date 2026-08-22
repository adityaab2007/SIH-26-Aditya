from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from backend.app.ml.forward_labels import build_forward_labels

FUTURE_ONLY_COLUMNS = {
    "future_snapshot_date",
    "future_schedule_shift_days",
    "future_schedule_slip_90d",
    "future_cost_escalation_pct",
    "future_cost_jump_5pct",
}


@dataclass(frozen=True)
class TemporalSplit:
    train_index: list[int]
    test_index: list[int]
    train_end: str
    test_start: str


def assert_no_target_leakage(features: Iterable[str], target: str) -> None:
    feature_set = set(features)
    forbidden = ({target} | FUTURE_ONLY_COLUMNS) & feature_set
    if forbidden:
        raise ValueError(f"Target/future leakage detected in features: {sorted(forbidden)}")


def chronological_holdout(
    frame: pd.DataFrame,
    date_col: str = "snapshot_date",
    test_fraction: float = 0.20,
) -> TemporalSplit | None:
    """Return a strict older-train/newer-test split.

    If the dataset has only one observation date, no temporal split is possible and
    callers must label any alternative evaluation as a single-snapshot baseline.
    """
    if date_col not in frame.columns:
        return None
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    valid_dates = sorted(d for d in dates.dropna().unique())
    if len(valid_dates) < 2:
        return None

    split_pos = max(1, min(len(valid_dates) - 1, int(len(valid_dates) * (1 - test_fraction))))
    cutoff = pd.Timestamp(valid_dates[split_pos])
    train_mask = dates < cutoff
    test_mask = dates >= cutoff
    if not train_mask.any() or not test_mask.any():
        return None

    return TemporalSplit(
        train_index=frame.index[train_mask].tolist(),
        test_index=frame.index[test_mask].tolist(),
        train_end=pd.Timestamp(dates[train_mask].max()).strftime("%Y-%m-%d"),
        test_start=pd.Timestamp(dates[test_mask].min()).strftime("%Y-%m-%d"),
    )


def forward_archive_status(path: Path | str, horizon_months: int = 2) -> dict:
    """Inspect whether the available monthly archive is large enough for model validation.

    Small official replay samples are useful to prove the label-generation mechanics but
    are not reported as statistically credible forecasting accuracy.
    """
    path = Path(path)
    if not path.exists():
        return {"available": False, "reason": "Monthly archive file not found", "label_rows": 0}

    snapshots = pd.read_csv(path, dtype={"project_code": str})
    if "revised_completion_date" in snapshots.columns and "revised_end_date" not in snapshots.columns:
        snapshots = snapshots.rename(columns={"revised_completion_date": "revised_end_date"})
    snapshots["snapshot_date"] = pd.to_datetime(snapshots.get("snapshot_date"), errors="coerce")
    labels = build_forward_labels(snapshots, horizon_months=horizon_months)
    schedule_rows = int(labels["future_schedule_shift_days"].notna().sum()) if "future_schedule_shift_days" in labels else 0
    cost_rows = int(labels["future_cost_escalation_pct"].notna().sum()) if "future_cost_escalation_pct" in labels else 0
    projects = int(labels["project_code"].nunique()) if not labels.empty else 0
    enough = min(schedule_rows, cost_rows) >= 30 and projects >= 10
    return {
        "available": enough,
        "horizon_months": horizon_months,
        "snapshot_rows": int(len(snapshots)),
        "projects": projects,
        "schedule_label_rows": schedule_rows,
        "cost_label_rows": cost_rows,
        "reason": None if enough else "Official replay sample is too small for statistically credible forward-model accuracy; ingest the expanded monthly PAIMANA/OCMS archive.",
        "label_generation_verified": not labels.empty,
    }
