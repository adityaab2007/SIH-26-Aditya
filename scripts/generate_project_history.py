"""Create a deterministic, documented longitudinal demo dataset from PAIMANA seed IDs.

The time series are synthetic because an official, record-level monthly OCMS archive is
not included in this repository. Values evolve coherently from each project's approved
cost, dates, sector and May-2026 PAIMANA snapshot; they are not random demo rows.
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paimana_projects_may_2026.csv"
OUT = ROOT / "data" / "project_history.csv"

SECTOR_RISK = {"Road Transport and Highways": 0.28, "Railways": 0.25, "Power": 0.18, "Urban Public Transport": 0.24, "Petroleum and Natural Gas": 0.26, "Telecommunication": 0.20}
STATES = ["Maharashtra", "Rajasthan", "Uttar Pradesh", "Tamil Nadu", "Gujarat", "Karnataka", "Assam", "Punjab"]


def main() -> None:
    source = pd.read_csv(RAW, dtype={"project_code": str})
    rows: list[dict] = []
    for index, project in source.reset_index(drop=True).iterrows():
        original = float(project.original_cost_cr)
        planned_end = pd.Timestamp(project.original_end_date)
        planned_start = planned_end - pd.DateOffset(months=36 + (index % 25))
        # Future-dated projects still need pre-award/pre-construction snapshots in
        # the May-2026 demo period; this keeps every seeded PAIMANA ID represented.
        start = min(planned_start, pd.Timestamp("2025-01-01"))
        risk = SECTOR_RISK.get(project.sector, 0.14) + ((index * 7) % 19) / 100
        final_overrun = round(8 + risk * 100 + (index % 7) * 2.2, 2)
        final_delay = int(80 + risk * 820 + (index % 6) * 27)
        actual_end = planned_end + pd.Timedelta(days=final_delay)
        actual_cost = round(original * (1 + final_overrun / 100), 2)
        dates = pd.date_range(start=start, end="2026-06-01", freq="MS")
        for sequence, month in enumerate(dates):
            elapsed = max(0, (month.year - planned_start.year) * 12 + month.month - planned_start.month)
            duration = max(1, (planned_end.year - planned_start.year) * 12 + planned_end.month - planned_start.month)
            expected = min(1, elapsed / duration)
            # Higher-risk projects fall progressively behind their planned curve.
            progress = min(98, max(1, 100 * (expected * (1 - risk * 0.32) + 0.018 * math.sin((sequence + index) / 2))))
            cost_share = min(0.97, max(0.01, progress / 100 * (0.82 + risk * 0.34)))
            estimate_share = 1 + (final_overrun / 100) * min(0.96, max(0, (elapsed / max(duration, 1)) ** 1.35))
            estimate = round(original * estimate_share, 2)
            prior_share = 0 if sequence == 0 else min(0.97, max(0.01, (progress - 100 / duration) / 100 * (0.82 + risk * 0.34)))
            monthly = round(max(original * 0.003, original * (cost_share - prior_share)), 2)
            revised_end = planned_end + pd.Timedelta(days=int(final_delay * min(0.94, max(0, elapsed / max(duration, 1)))))
            milestone = bool(sequence and sequence % 6 == 0)
            rows.append({
                "project_id": str(project.project_code), "month": month.strftime("%Y-%m-%d"), "project_name": project.project_name,
                "sector": project.sector, "ministry": project.ministry, "state": STATES[index % len(STATES)],
                "agency": f"{str(project.ministry).split()[0]} Implementation Agency", "original_cost": original,
                "current_estimated_cost": estimate, "actual_cost": actual_cost, "planned_start_date": planned_start.strftime("%Y-%m-%d"),
                "planned_completion_date": planned_end.strftime("%Y-%m-%d"), "revised_completion_date": revised_end.strftime("%Y-%m-%d"),
                "actual_completion_date": actual_end.strftime("%Y-%m-%d"), "monthly_expenditure": monthly,
                "physical_progress_percentage": round(progress, 2), "milestone_completed": milestone,
                "milestone_delay_days": int(max(0, risk * 28 + (sequence % 6) * risk * 4)),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} coherent synthetic monthly snapshots for {source.project_code.nunique()} PAIMANA project IDs to {OUT}")


if __name__ == "__main__":
    main()
