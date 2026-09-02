from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.app.ml.experiments import exp81_proxy_scientific_runner as base


def _proxy_rule_validation(resolved: pd.DataFrame, outcomes_raw: pd.DataFrame) -> dict:
    outcomes = base._prep(outcomes_raw)
    silver = resolved[
        resolved["identity_verified"].eq(True)
        & resolved["project_id"].notna()
        & resolved["canonical_project_id"].notna()
    ].copy()
    silver = silver.sort_values("snapshot_date").drop_duplicates("canonical_project_id", keep="last")
    silver = base._prep(silver)

    records = []
    for _, series in silver.iterrows():
        known = str(series.get("canonical_project_id")).strip()
        for allow_code in (True, False):
            candidate = base._candidate(series, outcomes, allow_code=allow_code)
            if candidate is None:
                continue
            outcome_index, rule = candidate
            candidate_id = outcomes_raw.loc[outcome_index].get("project_id")
            candidate_id = str(candidate_id).strip().removesuffix(".0") if pd.notna(candidate_id) else ""
            records.append(
                {
                    "canonical_project_id": known,
                    "rule": rule,
                    "correct": candidate_id == known,
                }
            )

    metrics = {}
    approved = set()
    if records:
        frame = pd.DataFrame(records).drop_duplicates(["canonical_project_id", "rule"], keep="last")
        for rule, group in frame.groupby("rule", sort=True):
            support = int(group["canonical_project_id"].nunique())
            precision = float(group["correct"].mean()) if support else 0.0
            status = (
                "APPROVED FOR PROXY AUTO-LINKING"
                if support >= 20 and precision >= 0.98
                else "NOT APPROVED"
            )
            metrics[rule] = {
                "support_projects": support,
                "precision_against_existing_verified_links": precision,
                "status": status,
            }
            if status.startswith("APPROVED"):
                approved.add(rule)

    return {
        "evidence_tier": "INTERNAL_SILVER_PROXY_NOT_HUMAN_GOLD",
        "gold_evidence_available": False,
        "rules": metrics,
        "approved_rules": sorted(approved),
        "warning": "Existing exact verified links are used only as an internal consistency proxy; this is not independent human gold validation.",
    }


base._proxy_rule_validation = _proxy_rule_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-end", type=int, required=True, choices=[2019, 2021])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base.run(args.training_end, args.output)
