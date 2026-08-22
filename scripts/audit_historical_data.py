#!/usr/bin/env python3
"""Report official PAIMANA historical coverage without creating data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paimana_completed_archive"
PROCESSED = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"
OUTPUT = ROOT / "data" / "processed" / "historical_data_audit.json"


def main() -> None:
    frame = pd.read_csv(PROCESSED, dtype={"project_id": str})
    frame["completion_date"] = pd.to_datetime(frame["completion_date"], errors="coerce")
    year_counts = frame["completion_date"].dt.year.value_counts().sort_index()
    raw_years = sorted(int(path.name.split("_")[1].split("-")[0]) for path in RAW.glob("march_*.pdf"))
    processed_years = [int(year) for year in year_counts.index]
    report = {
        "source": "Official PAIMANA completed-project archive PDFs stored in data/raw/paimana_completed_archive",
        "raw_min_year": min(raw_years),
        "raw_max_year": max(raw_years),
        "processed_min_year": min(processed_years),
        "processed_max_year": max(processed_years),
        "raw_report_years": raw_years,
        "processed_completion_years": processed_years,
        "missing_years": sorted(set(raw_years).difference(processed_years)),
        "processed_records": int(len(frame)),
        "records_by_completion_year": {str(year): int(count) for year, count in year_counts.items()},
        "required_training_fields_missing": {key: int(value) for key, value in frame[["approved_cost_cr", "planned_commissioning_date", "reported_completion_expenditure_cr", "completion_date"]].isna().sum().items()},
        "reason": "2001-2008 PDFs existed locally but used an older Completed Projects table header and dotted row numbering; the ingestion mapping now supports both historical and newer PAIMANA layouts.",
        "training_policy": "Only records with approved cost, planned commissioning date, reported completion expenditure, and completion date are eligible for supervised time-window training.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
