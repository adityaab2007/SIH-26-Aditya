from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_CURRENT = ROOT / "data" / "raw" / "paimana_projects_may_2026.csv"
PROCESSED = ROOT / "data" / "processed" / "model_dataset.csv"

CATEGORICAL_COLUMNS = ["sector", "ministry"]
BASE_NUMERIC_COLUMNS = [
    "original_cost_cr",
    "revised_cost_cr",
    "expenditure_cr",
    "physical_progress_pct",
    "days_to_original_deadline",
    "expenditure_to_original_pct",
    "financial_progress_pct",
    "cost_escalation_pct",
    "schedule_extension_days",
]

SCHEDULE_FEATURES = [
    "original_cost_cr",
    "revised_cost_cr",
    "expenditure_cr",
    "physical_progress_pct",
    "days_to_original_deadline",
    "expenditure_to_original_pct",
    "financial_progress_pct",
    "cost_escalation_pct",
    "sector",
    "ministry",
]

COST_FEATURES = [
    "original_cost_cr",
    "expenditure_cr",
    "physical_progress_pct",
    "days_to_original_deadline",
    "expenditure_to_original_pct",
    "financial_progress_pct",
    "schedule_extension_days",
    "sector",
    "ministry",
]


def load_and_engineer(path: Path | str = RAW_CURRENT) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"project_code": str})
    for col in ["snapshot_date", "original_end_date", "revised_end_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["revised_cost_cr"] = pd.to_numeric(df["revised_cost_cr"], errors="coerce")
    df["expenditure_cr"] = pd.to_numeric(df["expenditure_cr"], errors="coerce")
    df["physical_progress_pct"] = pd.to_numeric(df["physical_progress_pct"], errors="coerce")

    df["days_to_original_deadline"] = (
        df["original_end_date"] - df["snapshot_date"]
    ).dt.days
    df["schedule_extension_days"] = (
        df["revised_end_date"] - df["original_end_date"]
    ).dt.days
    df["cost_escalation_pct"] = np.where(
        (df["revised_cost_cr"].notna()) & (df["original_cost_cr"] > 0),
        (df["revised_cost_cr"] - df["original_cost_cr"]) / df["original_cost_cr"] * 100,
        np.nan,
    )
    df["expenditure_to_original_pct"] = np.where(
        (df["expenditure_cr"].notna()) & (df["original_cost_cr"] > 0),
        df["expenditure_cr"] / df["original_cost_cr"] * 100,
        np.nan,
    )
    df["financial_progress_pct"] = np.where(
        (df["expenditure_cr"].notna())
        & (df["revised_cost_cr"].notna())
        & (df["revised_cost_cr"] > 0),
        df["expenditure_cr"] / df["revised_cost_cr"] * 100,
        np.where(
            (df["expenditure_cr"].notna()) & (df["original_cost_cr"] > 0),
            df["expenditure_cr"] / df["original_cost_cr"] * 100,
            np.nan,
        ),
    )

    # These labels describe the currently observed overrun state. They are used only
    # as a real-data baseline while the full OCMS/PAIMANA monthly archive is ingested.
    df["schedule_overrun_90d"] = np.where(
        df["schedule_extension_days"].notna(),
        (df["schedule_extension_days"] > 90).astype(int),
        np.nan,
    )
    df["cost_overrun_5pct"] = np.where(
        df["cost_escalation_pct"].notna(),
        (df["cost_escalation_pct"] > 5).astype(int),
        np.nan,
    )

    # Data-quality flags are not silently corrected; the UI surfaces them.
    df["dq_expenditure_gt_revised"] = (
        df["revised_cost_cr"].notna()
        & df["expenditure_cr"].notna()
        & (df["expenditure_cr"] > df["revised_cost_cr"] * 1.02)
    ).astype(int)
    df["dq_revised_date_before_original"] = (
        df["revised_end_date"].notna()
        & df["original_end_date"].notna()
        & (df["revised_end_date"] < df["original_end_date"])
    ).astype(int)
    df["dq_missing_revised_cost"] = df["revised_cost_cr"].isna().astype(int)
    df["dq_missing_revised_date"] = df["revised_end_date"].isna().astype(int)
    df["dq_missing_progress"] = df["physical_progress_pct"].isna().astype(int)

    return df


def save_processed(df: pd.DataFrame, path: Path | str = PROCESSED) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in ["snapshot_date", "original_end_date", "revised_end_date"]:
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    frame = load_and_engineer()
    saved = save_processed(frame)
    print(f"Saved {len(frame)} engineered real PAIMANA rows to {saved}")
