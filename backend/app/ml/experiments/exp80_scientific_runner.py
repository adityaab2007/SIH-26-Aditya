from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import backend.app.ml.production_exp35_baseline as production_exp35_baseline
from backend.app.ml.experiments.exp80_deterministic_linkage import resolve_links
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


def _eligible_mask(frame: pd.DataFrame) -> pd.Series:
    raw = frame["cost_evaluation_eligible"]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False)
    return raw.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_validation(root: Path, training_start: int, training_end: int) -> pd.DataFrame:
    frame = pd.read_csv(
        root / f"{training_start}_{training_end}" / "prediction_validation.csv",
        low_memory=False,
    )
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame["canonical_project_id"] = frame["canonical_project_id"].astype("string").str.strip()
    comparable = assign_project_balanced_weights(frame.loc[_eligible_mask(frame)].copy())
    return comparable.sort_values(KEYS, kind="stable").reset_index(drop=True)


def _weighted_mae(frame: pd.DataFrame, actual: str, prediction: str) -> float:
    y = pd.to_numeric(frame[actual], errors="coerce").to_numpy(float)
    p = pd.to_numeric(frame[prediction], errors="coerce").to_numpy(float)
    w = pd.to_numeric(frame["sample_weight"], errors="coerce").to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(w)
    return float(np.average(np.abs(y[mask] - p[mask]), weights=w[mask]))


def _bootstrap(frame: pd.DataFrame, actual: str, baseline_col: str, challenger_col: str, seed: int) -> dict:
    work = frame[["canonical_project_id", "sample_weight", actual, baseline_col, challenger_col]].copy()
    records = []
    for project_id, group in work.groupby("canonical_project_id", sort=False):
        weight = pd.to_numeric(group["sample_weight"], errors="coerce").to_numpy(float)
        actual_values = pd.to_numeric(group[actual], errors="coerce").to_numpy(float)
        base_values = pd.to_numeric(group[baseline_col], errors="coerce").to_numpy(float)
        challenger_values = pd.to_numeric(group[challenger_col], errors="coerce").to_numpy(float)
        records.append(
            (
                str(project_id),
                float(np.average(np.abs(actual_values - base_values), weights=weight)),
                float(np.average(np.abs(actual_values - challenger_values), weights=weight)),
            )
        )
    per_project = pd.DataFrame(records, columns=["project", "baseline", "challenger"])
    improvement = per_project["baseline"].to_numpy(float) - per_project["challenger"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(improvement), size=(5000, len(improvement)))
    boot = improvement[draws].mean(axis=1)
    return {
        "samples": 5000,
        "projects": int(len(per_project)),
        "ci95_absolute_improvement": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "probability_challenger_beats_baseline": float(np.mean(boot > 0)),
        "project_win_rate": float(np.mean(improvement > 0)),
    }


def _expand_supervised_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    snapshots = load_monthly_snapshots()
    outcomes_raw = pd.read_csv(OUTCOMES, dtype={"project_id": "string"}, low_memory=False)
    resolved, identity = resolve_identities(snapshots, outcomes_raw)
    prepared_outcomes = _prepare_outcomes(outcomes_raw)

    candidate_snapshots = snapshots.copy()
    candidate_snapshots["snapshot_id"] = candidate_snapshots.index.astype(int)
    candidate_outcomes = outcomes_raw.copy()
    candidate_outcomes["completion_id"] = candidate_outcomes.index.astype(int)
    links, diagnostics = resolve_links(candidate_snapshots, candidate_outcomes)

    unresolved = set(resolved.index[~resolved["identity_verified"].fillna(False)].astype(int))
    accepted = links[links["snapshot_id"].astype(int).isin(unresolved)].copy() if len(links) else links.copy()

    expanded = resolved.copy()
    expanded_identity = identity.copy()
    newly_linked_rows = []
    for link in accepted.itertuples(index=False):
        row_index = int(link.snapshot_id)
        outcome_index = int(link.completion_id)
        outcome = prepared_outcomes.loc[outcome_index]
        row = expanded.loc[row_index]

        completion = pd.to_datetime(outcome.get("completion_date"), errors="coerce")
        expenditure = pd.to_numeric(pd.Series([outcome.get("reported_completion_expenditure_cr")]), errors="coerce").iloc[0]
        approved = pd.to_numeric(pd.Series([row.get("approved_cost_cr")]), errors="coerce").iloc[0]
        if pd.isna(approved) or approved <= 0:
            approved = pd.to_numeric(pd.Series([outcome.get("approved_cost_cr")]), errors="coerce").iloc[0]
        planned = pd.to_datetime(row.get("planned_completion_date"), errors="coerce")
        if pd.isna(planned):
            planned = pd.to_datetime(outcome.get("planned_commissioning_date"), errors="coerce")

        cost_target = (
            (float(expenditure) - float(approved)) / float(approved) * 100.0
            if pd.notna(expenditure) and pd.notna(approved) and approved > 0
            else np.nan
        )
        delay_target = max(0, int((completion - planned).days)) if pd.notna(completion) and pd.notna(planned) else np.nan
        project_id = outcome.get("project_id")
        canonical = str(project_id).strip() if pd.notna(project_id) and str(project_id).strip() else f"exp80:{outcome_index}"

        expanded.at[row_index, "canonical_project_id"] = canonical
        expanded.at[row_index, "identity_method"] = f"exp80_{link.rule}"
        expanded.at[row_index, "identity_confidence"] = 0.99 if int(link.tier) == 1 else 0.97
        expanded.at[row_index, "identity_verified"] = True
        expanded.at[row_index, "completion_date"] = completion
        expanded.at[row_index, "completion_year"] = completion.year if pd.notna(completion) else pd.NA
        expanded.at[row_index, "reported_completion_expenditure_cr"] = expenditure
        expanded.at[row_index, "actual_cost_overrun_percentage"] = cost_target
        expanded.at[row_index, "actual_delay_days"] = delay_target
        expanded.at[row_index, "actual_risk"] = risk_category(float(delay_target)) if pd.notna(delay_target) else None

        mask = expanded_identity["row_index"].eq(row_index)
        expanded_identity.loc[mask, "canonical_project_id"] = canonical
        expanded_identity.loc[mask, "identity_method"] = f"exp80_{link.rule}"
        expanded_identity.loc[mask, "identity_confidence"] = 0.99 if int(link.tier) == 1 else 0.97
        expanded_identity.loc[mask, "identity_verified"] = True
        newly_linked_rows.append(row_index)

    engineered = engineer_as_of_features(expanded, outcomes_raw)
    eligible = engineered[
        engineered["identity_verified"].eq(True)
        & engineered[TARGETS].notna().all(axis=1)
        & engineered["snapshot_date"].lt(engineered["completion_date"])
    ].copy()
    eligible["snapshot_quarter"] = eligible["snapshot_date"].dt.to_period("Q").astype(str)
    eligible = eligible.sort_values("snapshot_date").drop_duplicates(
        ["canonical_project_id", "snapshot_quarter"], keep="last"
    )
    eligible = assign_project_balanced_weights(eligible)
    invariants = training_as_of_invariants(eligible)
    if not invariants["passed"]:
        raise ValueError(f"Exp80 as-of invariant failure: {invariants}")

    accepted_rows = expanded.loc[newly_linked_rows].copy() if newly_linked_rows else expanded.iloc[0:0].copy()
    diagnostics_payload = {
        "accepted_unresolved_snapshots": int(len(newly_linked_rows)),
        "accepted_unresolved_projects": int(accepted_rows["canonical_project_id"].nunique()) if len(accepted_rows) else 0,
        "matches_by_rule": accepted["rule"].value_counts().to_dict() if len(accepted) else {},
        "ambiguous_snapshots": int((diagnostics.get("status", pd.Series(dtype=object)) == "ambiguous").sum()) if len(diagnostics) else 0,
        "unmatched_snapshots": int((diagnostics.get("status", pd.Series(dtype=object)) == "unmatched").sum()) if len(diagnostics) else 0,
    }
    return eligible, expanded_identity, diagnostics_payload


def _train_experiment_challenger(
    training_start: int,
    training_end: int,
    test_end: int,
    data: pd.DataFrame,
    identity: pd.DataFrame,
    artifact_root: Path,
) -> None:
    """Retrain the production architecture on Exp80 data without production-only promotion guards.

    The fixed-window verification in production_exp35_baseline is intentionally a guard for
    production promotion. Exp80 changes the training cohort by design, so applying that guard
    to the challenger makes a valid experiment impossible. Baseline retraining still runs with
    the production guard enabled; only this experiment-only challenger call bypasses it.
    """
    selected_window = production_exp35_baseline._selected_window
    try:
        production_exp35_baseline._selected_window = lambda *_args, **_kwargs: False
        train_current_production(
            training_start,
            training_end,
            test_end,
            data=data,
            identity=identity,
            artifact_root=artifact_root,
        )
    finally:
        production_exp35_baseline._selected_window = selected_window


def run(training_end: int, output: Path) -> dict:
    training_start, test_end = 2001, 2025
    baseline_data, baseline_identity = build_training_dataset()
    expanded_data, expanded_identity, linkage = _expand_supervised_dataset()

    base_test = baseline_data[baseline_data["completion_year"].between(training_end + 1, test_end)].copy()
    expanded_train = expanded_data[expanded_data["completion_year"].between(training_start, training_end)].copy()
    hybrid = pd.concat([expanded_train, base_test], ignore_index=True, sort=False)

    with tempfile.TemporaryDirectory(prefix="exp80_") as temp:
        temp_root = Path(temp)
        baseline_root = temp_root / "baseline"
        challenger_root = temp_root / "challenger"
        train_current_production(
            training_start,
            training_end,
            test_end,
            data=baseline_data,
            identity=baseline_identity,
            artifact_root=baseline_root,
        )
        _train_experiment_challenger(
            training_start,
            training_end,
            test_end,
            data=hybrid,
            identity=expanded_identity,
            artifact_root=challenger_root,
        )
        baseline = _load_validation(baseline_root, training_start, training_end)
        challenger = _load_validation(challenger_root, training_start, training_end)

    if not baseline[KEYS].equals(challenger[KEYS]):
        raise AssertionError("Exp80 challenger changed the held-out comparison keys")

    comparison = baseline.copy()
    comparison["challenger_cost"] = challenger["predicted_cost_overrun"].to_numpy(float)
    comparison["challenger_delay"] = challenger["predicted_delay_days"].to_numpy(float)

    production_cost = _weighted_mae(comparison, "actual_cost_overrun_percentage", "predicted_cost_overrun")
    experiment_cost = _weighted_mae(comparison, "actual_cost_overrun_percentage", "challenger_cost")
    production_delay = _weighted_mae(comparison, "actual_delay_days", "predicted_delay_days")
    experiment_delay = _weighted_mae(comparison, "actual_delay_days", "challenger_delay")

    cost_gain = (production_cost - experiment_cost) / production_cost * 100.0 if production_cost else 0.0
    delay_gain = (production_delay - experiment_delay) / production_delay * 100.0 if production_delay else 0.0
    new_train = expanded_train.merge(
        baseline_data[KEYS].drop_duplicates(), on=KEYS, how="left", indicator=True
    )
    new_train = new_train[new_train["_merge"].eq("left_only")]

    result = {
        "experiment_id": "exp_80",
        "experiment_sequence": 80,
        "window": f"2001-{training_end}",
        "status": "EXECUTION VALID",
        "production_cost_mae": production_cost,
        "experiment_cost_mae": experiment_cost,
        "cost_improvement_percent": cost_gain,
        "production_delay_mae": production_delay,
        "experiment_delay_mae": experiment_delay,
        "delay_improvement_percent": delay_gain,
        "comparison_projects": int(comparison["canonical_project_id"].nunique()),
        "comparison_snapshots": int(len(comparison)),
        "new_training_projects": int(new_train["canonical_project_id"].nunique()),
        "new_training_snapshots": int(len(new_train)),
        "linkage": linkage,
        "cost_bootstrap": _bootstrap(
            comparison,
            "actual_cost_overrun_percentage",
            "predicted_cost_overrun",
            "challenger_cost",
            8001 + training_end,
        ),
        "delay_bootstrap": _bootstrap(
            comparison,
            "actual_delay_days",
            "predicted_delay_days",
            "challenger_delay",
            8002 + training_end,
        ),
        "verdict": "IMPROVED" if cost_gain > 0 and delay_gain > 0 else "MIXED / NO PROMOTION",
        "comparison_policy": "same original held-out project/snapshot keys; only training identities are expanded",
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
