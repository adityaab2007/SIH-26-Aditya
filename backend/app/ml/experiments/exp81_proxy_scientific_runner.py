from __future__ import annotations

import argparse
import json
import re
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import (
    OUTCOMES,
    TARGETS,
    _prepare_outcomes,
    assign_project_balanced_weights,
    build_training_dataset,
    engineer_as_of_features,
    load_monthly_snapshots,
    resolve_identities,
    risk_category,
    training_as_of_invariants,
)
from backend.app.ml.production_u1_delay_baseline import (
    train_window_with_promoted_cost_and_delay as train_current_production,
)

KEYS = ["canonical_project_id", "snapshot_date"]


def _text(value):
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def _code(value):
    return re.sub(r"[^a-z0-9]", "", _text(value)).upper()


def _title(value):
    value = " " + _text(value) + " "
    for source, target in {
        " rd ": " road ",
        " nh ": " national highway ",
        " rly ": " railway ",
        " stn ": " station ",
    }.items():
        value = value.replace(source, target)
    return re.sub(r"\s+", " ", value).strip()


def _norm(value):
    return _text(value)


def _cost_ok(left, right):
    left, right = pd.to_numeric(pd.Series([left, right]), errors="coerce")
    if pd.isna(left) or pd.isna(right):
        return False
    return abs(float(left) - float(right)) <= max(0.05, 0.01 * max(abs(float(left)), abs(float(right)), 1.0))


def _date_ok(left, right):
    left, right = pd.to_datetime(left, errors="coerce"), pd.to_datetime(right, errors="coerce")
    return bool(pd.notna(left) and pd.notna(right) and abs((left - right).days) <= 31)


def _prep(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_code"] = result.get("project_id", pd.Series("", index=result.index)).map(_code)
    result["_title"] = result.get("project_name", pd.Series("", index=result.index)).map(_title)
    result["_state"] = result.get("state", pd.Series("", index=result.index)).map(_norm).replace({"orissa": "odisha", "uttaranchal": "uttarakhand", "nct of delhi": "delhi"})
    result["_sector"] = result.get("sector", pd.Series("", index=result.index)).map(_norm)
    result["_agency"] = result.get("implementing_agency", pd.Series("", index=result.index)).map(_norm)
    return result


def _candidate(row: pd.Series, outcomes: pd.DataFrame, *, allow_code=True) -> tuple[int, str] | None:
    if allow_code and row["_code"]:
        exact = outcomes[outcomes["_code"].eq(row["_code"])]
        if len(exact) == 1:
            return int(exact.index[0]), "exact_project_code"
        if len(exact) > 1:
            return None

    if not row["_title"]:
        return None
    candidates = outcomes[
        outcomes["_title"].eq(row["_title"])
        & outcomes["_state"].eq(row["_state"])
        & outcomes["_sector"].eq(row["_sector"])
    ]
    candidates = candidates[
        [_cost_ok(row.get("approved_cost_cr"), item.get("approved_cost_cr")) for _, item in candidates.iterrows()]
    ] if len(candidates) else candidates
    if len(candidates) == 1:
        return int(candidates.index[0]), "title_state_sector_cost"
    if len(candidates) <= 1:
        return None

    planned = row.get("planned_completion_date", row.get("planned_commissioning_date"))
    refined = []
    for index, item in candidates.iterrows():
        same_agency = bool(row["_agency"] and row["_agency"] == item["_agency"])
        candidate_planned = item.get("planned_commissioning_date", item.get("planned_completion_date"))
        if same_agency and _date_ok(planned, candidate_planned):
            refined.append(int(index))
    if len(refined) == 1:
        return refined[0], "agency_state_sector_cost_date_title"
    return None


def _proxy_rule_validation(resolved: pd.DataFrame, outcomes_raw: pd.DataFrame) -> dict:
    outcomes = _prep(outcomes_raw)
    silver = resolved[
        resolved["identity_verified"].eq(True)
        & resolved["project_id"].notna()
        & resolved["canonical_project_id"].notna()
    ].copy()
    silver = silver.sort_values("snapshot_date").drop_duplicates("canonical_project_id", keep="last")
    silver = _prep(silver)

    records = []
    for row in silver.itertuples(index=False):
        series = pd.Series(row._asdict())
        known = str(series.get("canonical_project_id")).strip()
        for allow_code in (True, False):
            candidate = _candidate(series, outcomes, allow_code=allow_code)
            if candidate is None:
                continue
            outcome_index, rule = candidate
            candidate_id = outcomes_raw.loc[outcome_index].get("project_id")
            candidate_id = str(candidate_id).strip().removesuffix(".0") if pd.notna(candidate_id) else ""
            records.append({"rule": rule, "correct": candidate_id == known})

    metrics = {}
    approved = set()
    if records:
        frame = pd.DataFrame(records).drop_duplicates()
        for rule, group in frame.groupby("rule", sort=True):
            support = int(len(group))
            precision = float(group["correct"].mean()) if support else 0.0
            status = "APPROVED FOR PROXY AUTO-LINKING" if support >= 20 and precision >= 0.98 else "NOT APPROVED"
            metrics[rule] = {"support_projects": support, "precision_against_existing_verified_links": precision, "status": status}
            if status.startswith("APPROVED"):
                approved.add(rule)
    return {
        "evidence_tier": "INTERNAL_SILVER_PROXY_NOT_HUMAN_GOLD",
        "gold_evidence_available": False,
        "rules": metrics,
        "approved_rules": sorted(approved),
        "warning": "Existing exact verified links are used only as an internal consistency proxy; this is not independent human gold validation.",
    }


def _expand_with_approved_rules(approved_rules: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    snapshots = load_monthly_snapshots()
    outcomes_raw = pd.read_csv(OUTCOMES, dtype={"project_id": "string"}, low_memory=False)
    resolved, identity = resolve_identities(snapshots, outcomes_raw)
    prepared_outcomes = _prepare_outcomes(outcomes_raw)
    prepared_candidates = _prep(outcomes_raw)
    prepared_snapshots = _prep(snapshots)

    expanded = resolved.copy()
    expanded_identity = identity.copy()
    accepted = []
    unresolved_indices = resolved.index[~resolved["identity_verified"].fillna(False)]
    for row_index in unresolved_indices:
        candidate = _candidate(prepared_snapshots.loc[row_index], prepared_candidates, allow_code=True)
        if candidate is None:
            continue
        outcome_index, rule = candidate
        if rule not in approved_rules:
            continue
        accepted.append((int(row_index), int(outcome_index), rule))

    for row_index, outcome_index, rule in accepted:
        outcome = prepared_outcomes.loc[outcome_index]
        row = expanded.loc[row_index]
        completion = pd.to_datetime(outcome.get("completion_date"), errors="coerce")
        expenditure = pd.to_numeric(pd.Series([outcome.get("reported_completion_expenditure_cr")]), errors="coerce").iloc[0]
        approved_cost = pd.to_numeric(pd.Series([row.get("approved_cost_cr")]), errors="coerce").iloc[0]
        if pd.isna(approved_cost) or approved_cost <= 0:
            approved_cost = pd.to_numeric(pd.Series([outcome.get("approved_cost_cr")]), errors="coerce").iloc[0]
        planned = pd.to_datetime(row.get("planned_completion_date"), errors="coerce")
        if pd.isna(planned):
            planned = pd.to_datetime(outcome.get("planned_commissioning_date"), errors="coerce")
        cost_target = (
            (float(expenditure) - float(approved_cost)) / float(approved_cost) * 100.0
            if pd.notna(expenditure) and pd.notna(approved_cost) and approved_cost > 0
            else np.nan
        )
        delay_target = max(0, int((completion - planned).days)) if pd.notna(completion) and pd.notna(planned) else np.nan
        candidate_id = outcome.get("project_id")
        canonical = str(candidate_id).strip() if pd.notna(candidate_id) and str(candidate_id).strip() else f"exp81:{outcome_index}"

        expanded.at[row_index, "canonical_project_id"] = canonical
        expanded.at[row_index, "identity_method"] = f"exp81_proxy_{rule}"
        expanded.at[row_index, "identity_confidence"] = 0.97
        expanded.at[row_index, "identity_verified"] = True
        expanded.at[row_index, "completion_date"] = completion
        expanded.at[row_index, "completion_year"] = completion.year if pd.notna(completion) else pd.NA
        expanded.at[row_index, "reported_completion_expenditure_cr"] = expenditure
        expanded.at[row_index, "actual_cost_overrun_percentage"] = cost_target
        expanded.at[row_index, "actual_delay_days"] = delay_target
        expanded.at[row_index, "actual_risk"] = risk_category(float(delay_target)) if pd.notna(delay_target) else None

        mask = expanded_identity["row_index"].eq(row_index)
        expanded_identity.loc[mask, "canonical_project_id"] = canonical
        expanded_identity.loc[mask, "identity_method"] = f"exp81_proxy_{rule}"
        expanded_identity.loc[mask, "identity_confidence"] = 0.97
        expanded_identity.loc[mask, "identity_verified"] = True

    engineered = engineer_as_of_features(expanded, outcomes_raw)
    eligible = engineered[
        engineered["identity_verified"].eq(True)
        & engineered[TARGETS].notna().all(axis=1)
        & engineered["snapshot_date"].lt(engineered["completion_date"])
    ].copy()
    eligible["snapshot_quarter"] = eligible["snapshot_date"].dt.to_period("Q").astype(str)
    eligible = eligible.sort_values("snapshot_date").drop_duplicates(["canonical_project_id", "snapshot_quarter"], keep="last")
    eligible = assign_project_balanced_weights(eligible)
    invariants = training_as_of_invariants(eligible)
    if not invariants["passed"]:
        raise ValueError(f"Exp81 proxy as-of invariant failure: {invariants}")
    return eligible, expanded_identity, {"accepted_proxy_snapshots": len(accepted), "accepted_rules": pd.Series([item[2] for item in accepted], dtype=object).value_counts().to_dict()}


def _eligible_mask(frame: pd.DataFrame) -> pd.Series:
    raw = frame["cost_evaluation_eligible"]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_validation(root: Path, training_start: int, training_end: int) -> pd.DataFrame:
    frame = pd.read_csv(root / f"{training_start}_{training_end}" / "prediction_validation.csv", low_memory=False)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string").str.strip()
    frame = assign_project_balanced_weights(frame.loc[_eligible_mask(frame)].copy())
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _weighted_mae(frame: pd.DataFrame, actual: str, prediction) -> float:
    y = pd.to_numeric(frame[actual], errors="coerce").to_numpy(float)
    p = np.asarray(prediction, dtype=float)
    w = pd.to_numeric(frame["sample_weight"], errors="coerce").to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(w)
    return float(np.average(np.abs(y[mask] - p[mask]), weights=w[mask]))


def run(training_end: int, output: Path) -> dict:
    training_start, test_end = 2001, 2025
    baseline_data, baseline_identity = build_training_dataset()
    snapshots = load_monthly_snapshots()
    outcomes_raw = pd.read_csv(OUTCOMES, dtype={"project_id": "string"}, low_memory=False)
    resolved, _ = resolve_identities(snapshots, outcomes_raw)
    proxy = _proxy_rule_validation(resolved, outcomes_raw)
    approved = set(proxy["approved_rules"])
    expanded_data, expanded_identity, expansion = _expand_with_approved_rules(approved)

    base_test = baseline_data[baseline_data["completion_year"].between(training_end + 1, test_end)].copy()
    expanded_train = expanded_data[expanded_data["completion_year"].between(training_start, training_end)].copy()
    hybrid = pd.concat([expanded_train, base_test], ignore_index=True, sort=False)

    with tempfile.TemporaryDirectory(prefix="exp81_proxy_") as temp:
        root = Path(temp)
        base_root, candidate_root = root / "baseline", root / "candidate"
        train_current_production(training_start, training_end, test_end, data=baseline_data, identity=baseline_identity, artifact_root=base_root)
        train_current_production(training_start, training_end, test_end, data=hybrid, identity=expanded_identity, artifact_root=candidate_root)
        baseline = _load_validation(base_root, training_start, training_end)
        candidate = _load_validation(candidate_root, training_start, training_end)

    if not baseline[KEYS].equals(candidate[KEYS]):
        raise AssertionError("Exp81 proxy challenger changed held-out comparison keys")

    production_cost = _weighted_mae(baseline, "actual_cost_overrun_percentage", baseline["predicted_cost_overrun"])
    experiment_cost = _weighted_mae(baseline, "actual_cost_overrun_percentage", candidate["predicted_cost_overrun"])
    production_delay = _weighted_mae(baseline, "actual_delay_days", baseline["predicted_delay_days"])
    experiment_delay = _weighted_mae(baseline, "actual_delay_days", candidate["predicted_delay_days"])
    cost_gain = (production_cost - experiment_cost) / production_cost * 100.0 if production_cost else 0.0
    delay_gain = (production_delay - experiment_delay) / production_delay * 100.0 if production_delay else 0.0

    result = {
        "experiment_id": "exp_81",
        "experiment_sequence": 81,
        "window": f"2001-{training_end}",
        "status": "EXECUTION VALID - INTERNAL PROXY ONLY",
        "true_gold_result": "UNAVAILABLE: no manually audited gold linkage file exists in the repository",
        "proxy_validation": proxy,
        "proxy_expansion": expansion,
        "production_cost_mae": production_cost,
        "experiment_cost_mae": experiment_cost,
        "cost_improvement_percent": cost_gain,
        "production_delay_mae": production_delay,
        "experiment_delay_mae": experiment_delay,
        "delay_improvement_percent": delay_gain,
        "comparison_projects": int(baseline["canonical_project_id"].nunique()),
        "comparison_snapshots": int(len(baseline)),
        "verdict": "PROXY IMPROVED" if cost_gain > 0 and delay_gain > 0 else "PROXY MIXED / NO PROMOTION",
        "comparison_policy": "same original holdout keys; only proxy-approved training links are added",
        "scientific_limitation": "This can measure downstream sensitivity to internally validated linkage rules, but cannot replace independent human gold precision/recall evidence.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-end", type=int, required=True, choices=[2019, 2021])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.training_end, args.output)
