#!/usr/bin/env python3
"""Build the deterministic, current-snapshot analytics table from PAIMANA rows."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paimana_projects_may_2026.csv"
OUTPUT = ROOT / "data" / "processed" / "model_dataset.csv"


def build_dataset(source: Path = RAW, output: Path = OUTPUT) -> pd.DataFrame:
    frame = pd.read_csv(source, dtype={"project_code": str})
    for column in ["snapshot_date", "original_end_date", "revised_end_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in ["original_cost_cr", "revised_cost_cr", "expenditure_cr", "physical_progress_pct"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    current_cost = frame["revised_cost_cr"].fillna(frame["original_cost_cr"])
    frame["days_to_original_deadline"] = (frame["original_end_date"] - frame["snapshot_date"]).dt.days
    frame["schedule_extension_days"] = (frame["revised_end_date"] - frame["original_end_date"]).dt.days
    frame["cost_escalation_pct"] = np.where(
        frame["original_cost_cr"] > 0,
        (frame["revised_cost_cr"] - frame["original_cost_cr"]) / frame["original_cost_cr"] * 100,
        np.nan,
    )
    frame["expenditure_to_original_pct"] = np.where(
        frame["original_cost_cr"] > 0,
        frame["expenditure_cr"] / frame["original_cost_cr"] * 100,
        np.nan,
    )
    frame["financial_progress_pct"] = np.where(current_cost > 0, frame["expenditure_cr"] / current_cost * 100, np.nan)
    frame["schedule_overrun_90d"] = np.where(frame["schedule_extension_days"].notna(), (frame["schedule_extension_days"] >= 90).astype(float), np.nan)
    frame["cost_overrun_5pct"] = np.where(frame["cost_escalation_pct"].notna(), (frame["cost_escalation_pct"] >= 5).astype(float), np.nan)
    frame["dq_expenditure_gt_revised"] = (frame["expenditure_cr"] > current_cost).astype(int)
    frame["dq_revised_date_before_original"] = (frame["revised_end_date"] < frame["original_end_date"]).astype(int)
    frame["dq_missing_revised_cost"] = frame["revised_cost_cr"].isna().astype(int)
    frame["dq_missing_revised_date"] = frame["revised_end_date"].isna().astype(int)
    frame["dq_missing_progress"] = frame["physical_progress_pct"].isna().astype(int)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, date_format="%Y-%m-%d")
    return frame


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}. Run scripts/seed_official_data.py first.")
    frame = build_dataset()
    print(f"Wrote {len(frame)} processed PAIMANA project rows to {OUTPUT}")


if __name__ == "__main__":
    sys.exit(main())
