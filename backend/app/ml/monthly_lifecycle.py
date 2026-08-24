"""Leakage-safe monthly PAIMANA trajectory construction and feature engineering."""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS = ROOT / "data" / "processed" / "paimana_monthly_snapshots.csv"
SNAPSHOTS_GZ = ROOT / "data" / "processed" / "paimana_monthly_snapshots.csv.gz"
OUTCOMES = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"
TRAJECTORIES = ROOT / "data" / "processed" / "paimana_project_trajectories.csv"
TRAINING_DATA = ROOT / "data" / "processed" / "paimana_snapshot_training_dataset.csv"
IDENTITY_AUDIT = ROOT / "data" / "processed" / "paimana_identity_audit.csv"

BASELINE_FEATURES = ["approved_cost_cr", "sector_average_delay", "sector_average_cost_overrun", "sector", "project_size_category"]
DIRECT_FEATURES = [
    "approved_cost_cr", "cumulative_expenditure_cr", "expenditure_ratio", "physical_progress",
    "schedule_slippage_days", "schedule_slippage_ratio", "elapsed_duration_days", "planned_duration_days",
    "duration_ratio", "expected_progress_percentage", "progress_deviation", "revised_cost_cr",
    "cost_escalation_percentage", "current_schedule_status", "sector", "project_size_category",
    "implementing_agency", "ministry",
]
TRAJECTORY_FEATURES = ["cost_growth_velocity_3m", "cost_growth_velocity_6m", "cost_acceleration", "progress_velocity_3m", "progress_velocity_6m", "progress_acceleration"]
PRIOR_FEATURES = [
    "sector_average_delay", "sector_average_cost_overrun", "sector_delay_rate", "sector_cost_overrun_rate",
    "agency_average_delay", "agency_average_cost_overrun", "agency_delay_rate", "agency_cost_overrun_rate",
]
CANDIDATE_FEATURES = list(dict.fromkeys(DIRECT_FEATURES + TRAJECTORY_FEATURES + PRIOR_FEATURES))
TARGETS = ["actual_cost_overrun_percentage", "actual_delay_days", "actual_risk"]
DATE_COLUMNS = ["snapshot_date", "approval_date", "planned_start_date", "planned_completion_date", "revised_completion_date", "actual_completion_date"]

_SNAPSHOT_SOURCE = "same official PAIMANA snapshot; no later report is consulted"
_TRAJECTORY_SOURCE = "current snapshot plus strictly earlier snapshots for the same canonical project"
_PRIOR_SOURCE = "completed projects with completion_date strictly earlier than the current snapshot_date"

AS_OF_FEATURE_LINEAGE: dict[str, dict] = {
    "approved_cost_cr": {"proven": True, "kind": "snapshot", "sources": ["approved_cost_cr"], "temporal_rule": _SNAPSHOT_SOURCE},
    "cumulative_expenditure_cr": {"proven": True, "kind": "snapshot", "sources": ["cumulative_expenditure_cr"], "temporal_rule": _SNAPSHOT_SOURCE},
    "expenditure_ratio": {"proven": True, "kind": "snapshot_derived", "sources": ["cumulative_expenditure_cr", "approved_cost_cr"], "temporal_rule": _SNAPSHOT_SOURCE},
    "physical_progress": {"proven": True, "kind": "snapshot", "sources": ["physical_progress"], "temporal_rule": _SNAPSHOT_SOURCE},
    "schedule_slippage_days": {"proven": True, "kind": "snapshot_derived", "sources": ["revised_completion_date", "planned_completion_date"], "temporal_rule": _SNAPSHOT_SOURCE},
    "schedule_slippage_ratio": {"proven": True, "kind": "snapshot_derived", "sources": ["revised_completion_date", "planned_completion_date", "planned_start_date", "approval_date"], "temporal_rule": _SNAPSHOT_SOURCE},
    "elapsed_duration_days": {"proven": True, "kind": "snapshot_derived", "sources": ["snapshot_date", "planned_start_date", "approval_date"], "temporal_rule": _SNAPSHOT_SOURCE},
    "planned_duration_days": {"proven": True, "kind": "snapshot_derived", "sources": ["planned_start_date", "approval_date", "planned_completion_date"], "temporal_rule": _SNAPSHOT_SOURCE},
    "duration_ratio": {"proven": True, "kind": "snapshot_derived", "sources": ["snapshot_date", "planned_start_date", "approval_date", "planned_completion_date"], "temporal_rule": _SNAPSHOT_SOURCE},
    "expected_progress_percentage": {"proven": True, "kind": "snapshot_derived", "sources": ["duration_ratio"], "temporal_rule": _SNAPSHOT_SOURCE},
    "progress_deviation": {"proven": True, "kind": "snapshot_derived", "sources": ["physical_progress", "expected_progress_percentage"], "temporal_rule": _SNAPSHOT_SOURCE},
    "revised_cost_cr": {"proven": True, "kind": "snapshot", "sources": ["revised_cost_cr"], "temporal_rule": _SNAPSHOT_SOURCE},
    "cost_escalation_percentage": {"proven": True, "kind": "snapshot_derived", "sources": ["revised_cost_cr", "approved_cost_cr"], "temporal_rule": _SNAPSHOT_SOURCE},
    "current_schedule_status": {"proven": True, "kind": "snapshot", "sources": ["current_schedule_status"], "temporal_rule": _SNAPSHOT_SOURCE},
    "sector": {"proven": True, "kind": "snapshot", "sources": ["sector"], "temporal_rule": _SNAPSHOT_SOURCE},
    "project_size_category": {"proven": True, "kind": "snapshot_derived", "sources": ["approved_cost_cr"], "temporal_rule": _SNAPSHOT_SOURCE},
    "implementing_agency": {"proven": True, "kind": "snapshot", "sources": ["implementing_agency"], "temporal_rule": _SNAPSHOT_SOURCE},
    "ministry": {"proven": True, "kind": "snapshot", "sources": ["ministry"], "temporal_rule": _SNAPSHOT_SOURCE},
}
for _feature in TRAJECTORY_FEATURES:
    AS_OF_FEATURE_LINEAGE[_feature] = {
        "proven": True,
        "kind": "trajectory",
        "sources": ["snapshot_date", "revised_cost_cr" if _feature.startswith("cost_") else "physical_progress"],
        "temporal_rule": _TRAJECTORY_SOURCE,
    }
for _feature in PRIOR_FEATURES:
    AS_OF_FEATURE_LINEAGE[_feature] = {
        "proven": True,
        "kind": "historical_prior",
        "sources": ["completion_date", "actual_delay_days", "actual_cost_overrun_percentage"],
        "temporal_rule": _PRIOR_SOURCE,
    }


def as_of_feature_evidence(feature_names: list[str] | None = None) -> dict[str, dict]:
    names = feature_names or CANDIDATE_FEATURES
    return {name: dict(AS_OF_FEATURE_LINEAGE[name]) for name in names if name in AS_OF_FEATURE_LINEAGE}


def load_monthly_snapshots(snapshot_path: Path | None = None) -> pd.DataFrame:
    """Load the official processed monthly snapshots without archive ingestion.

    The uncompressed CSV is preferred for local development. A tracked gzip
    artifact is supported for clones where the uncompressed file is too large
    for normal Git storage. Neither path triggers PDF discovery or parsing.
    """
    requested = Path(snapshot_path) if snapshot_path is not None else None
    candidates = [requested] if requested is not None else [SNAPSHOTS, SNAPSHOTS_GZ]
    if requested == SNAPSHOTS:
        candidates.append(SNAPSHOTS_GZ)
    for path in candidates:
        if path is not None and path.exists():
            return pd.read_csv(path, dtype={"project_id": "string"}, low_memory=False)
    raise FileNotFoundError(
        "Official processed monthly PAIMANA dataset is unavailable in this checkout: "
        "data/processed/paimana_monthly_snapshots.csv (or .csv.gz). "
        "Run the separate local-only monthly PAIMANA data refresh before retraining."
    )


def normalize_name(value: object) -> str:
    safe = "" if value is None or pd.isna(value) else str(value)
    text = re.sub(r"[^a-z0-9]+", " ", safe.lower()).strip()
    return re.sub(r"\b(project|phase|stage)\b", lambda m: m.group(0), text)


def _clean_id(value: object) -> str | None:
    value = "" if value is None or pd.isna(value) else str(value)
    value = value.strip().upper().removesuffix(".0")
    return value if re.fullmatch(r"[A-Z]?\d{8,9}", value) else None


def risk_category(delay_days: float) -> str:
    if delay_days < 90:
        return "LOW"
    if delay_days < 365:
        return "MEDIUM"
    if delay_days < 730:
        return "HIGH"
    return "CRITICAL"


def _prepare_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy(); data["project_id"] = data.project_id.map(_clean_id)
    data["name_key"] = data.project_name.map(normalize_name)
    for column in ["completion_date", "planned_commissioning_date"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["actual_cost_overrun_percentage"] = np.where(
        data.approved_cost_cr.gt(0),
        (data.reported_completion_expenditure_cr - data.approved_cost_cr) / data.approved_cost_cr * 100,
        np.nan,
    )
    data["actual_delay_days"] = (data.completion_date - data.planned_commissioning_date).dt.days.clip(lower=0)
    data["actual_risk"] = data.actual_delay_days.map(lambda value: risk_category(float(value)) if pd.notna(value) else None)
    data["completion_year"] = data.completion_date.dt.year.astype("Int64")
    return data


def resolve_identities(snapshots: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve exact IDs first; exact normalized-name matches require uniqueness."""
    snap = snapshots.copy(); outcome = _prepare_outcomes(outcomes)
    snap["project_id"] = snap.project_id.map(_clean_id); snap["name_key"] = snap.project_name.map(normalize_name)
    id_counts = outcome.dropna(subset=["project_id"]).groupby("project_id").size()
    name_counts = outcome[outcome.name_key.ne("")].groupby("name_key").size()
    outcome_id = outcome.dropna(subset=["project_id"]).drop_duplicates("project_id").set_index("project_id")
    outcome_name = outcome[outcome.name_key.ne("")].drop_duplicates("name_key").set_index("name_key")
    rows = []
    for index, row in snap.iterrows():
        code = row.project_id; name_key = row.name_key; match = None; method = "unresolved"; verified = False; confidence = 0.0
        if code and id_counts.get(code, 0) == 1:
            match = outcome_id.loc[code]; method = "exact_official_project_id"; verified = True; confidence = 1.0
            canonical = code
        elif name_key and name_counts.get(name_key, 0) == 1:
            candidate = outcome_name.loc[name_key]
            approved = pd.to_numeric(pd.Series([row.get("approved_cost_cr")]), errors="coerce").iloc[0]
            cost_matches = pd.notna(approved) and pd.notna(candidate.approved_cost_cr) and np.isclose(float(approved), float(candidate.approved_cost_cr), rtol=0, atol=0.05)
            if cost_matches:
                match = candidate; method = "exact_name_and_approved_cost"; verified = True; confidence = 0.95
            else:
                method = "exact_name_cost_mismatch"
            canonical = candidate.project_id if verified and candidate.project_id else f"legacy:{sha1(name_key.encode()).hexdigest()[:16]}"
        else:
            canonical = code or f"legacy:{sha1((name_key + '|' + normalize_name(row.get('implementing_agency'))).encode()).hexdigest()[:16]}"
            if name_key and name_counts.get(name_key, 0) > 1:
                method = "ambiguous_exact_name"
        snap.at[index, "canonical_project_id"] = canonical; snap.at[index, "identity_method"] = method
        snap.at[index, "identity_confidence"] = confidence; snap.at[index, "identity_verified"] = bool(verified)
        if match is not None:
            completion = match["completion_date"]
            expenditure = pd.to_numeric(pd.Series([match["reported_completion_expenditure_cr"]]), errors="coerce").iloc[0]
            approved = pd.to_numeric(pd.Series([row.get("approved_cost_cr")]), errors="coerce").iloc[0]
            if pd.isna(approved) or approved <= 0:
                approved = pd.to_numeric(pd.Series([match["approved_cost_cr"]]), errors="coerce").iloc[0]
            planned = pd.to_datetime(row.get("planned_completion_date"), errors="coerce")
            if pd.isna(planned):
                planned = match["planned_commissioning_date"]
            cost_target = (expenditure - approved) / approved * 100 if pd.notna(expenditure) and pd.notna(approved) and approved > 0 else np.nan
            delay_target = max(0, (completion - planned).days) if pd.notna(completion) and pd.notna(planned) else np.nan
            snap.at[index, "completion_date"] = completion
            snap.at[index, "completion_year"] = completion.year if pd.notna(completion) else pd.NA
            snap.at[index, "reported_completion_expenditure_cr"] = expenditure
            snap.at[index, "actual_cost_overrun_percentage"] = cost_target
            snap.at[index, "actual_delay_days"] = delay_target
            snap.at[index, "actual_risk"] = risk_category(float(delay_target)) if pd.notna(delay_target) else None
        rows.append({"row_index": int(index), "project_id": code, "project_name": row.project_name, "canonical_project_id": canonical,
                     "identity_method": method, "identity_confidence": confidence, "identity_verified": bool(verified)})
    return snap, pd.DataFrame(rows)


def _slope_as_of(group: pd.DataFrame, value_column: str, window_months: int) -> pd.Series:
    values = pd.to_numeric(group[value_column], errors="coerce"); dates = group.snapshot_date
    result = pd.Series(np.nan, index=group.index, dtype=float)
    for position, index in enumerate(group.index):
        current_date = dates.loc[index]; current_value = values.loc[index]
        if pd.isna(current_date) or pd.isna(current_value):
            continue
        prior_indices = group.index[:position]
        eligible = [i for i in prior_indices if pd.notna(values.loc[i]) and 0 < (current_date - dates.loc[i]).days <= window_months * 31]
        if not eligible:
            continue
        earlier = eligible[0]; months = (current_date - dates.loc[earlier]).days / 30.4375
        result.loc[index] = (current_value - values.loc[earlier]) / months if months > 0 else np.nan
    return result


def _historical_priors(frame: pd.DataFrame, outcomes: pd.DataFrame, minimum: int = 3) -> pd.DataFrame:
    result = frame.copy()
    known = result[
        result.identity_verified.eq(True)
        & result[["completion_date", "actual_delay_days", "actual_cost_overrun_percentage"]].notna().all(axis=1)
    ].sort_values("snapshot_date").drop_duplicates("canonical_project_id", keep="first").copy()
    for column in PRIOR_FEATURES:
        result[column] = np.nan
    known = known.dropna(subset=["completion_date", "actual_delay_days", "actual_cost_overrun_percentage"]).copy()

    def lookup(snapshot_dates: pd.Series, history: pd.DataFrame) -> tuple[np.ndarray, ...]:
        size = len(snapshot_dates)
        empty = (np.zeros(size, dtype=int),) + tuple(np.full(size, np.nan) for _ in range(4))
        if history.empty:
            return empty
        ordered = history.sort_values("completion_date")
        dates = ordered.completion_date.to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(dates, snapshot_dates.to_numpy(dtype="datetime64[ns]"), side="left") - 1
        valid = positions >= 0
        counts = np.where(valid, positions + 1, 0)
        delay = ordered.actual_delay_days.to_numpy(float)
        cost = ordered.actual_cost_overrun_percentage.to_numpy(float)
        cumulatives = [np.cumsum(delay), np.cumsum(cost), np.cumsum(delay >= 90), np.cumsum(cost > 0)]
        values = []
        for cumulative in cumulatives:
            output = np.full(size, np.nan); output[valid] = cumulative[positions[valid]] / counts[valid]
            values.append(output)
        return counts, *values

    global_stats = lookup(result.snapshot_date, known)
    missing_group = "__PAIMANA_MISSING__"
    sector_groups = result["sector"].astype("string").fillna(missing_group)
    for value, indices in sector_groups.groupby(sector_groups).groups.items():
        local = known[known.sector.eq(value)] if value != missing_group and "sector" in known else known.iloc[0:0]
        stats = lookup(result.loc[indices, "snapshot_date"], local)
        global_slice = tuple(array[result.index.get_indexer(indices)] for array in global_stats)
        selected = tuple(np.where(stats[0] >= minimum, stats[i], global_slice[i]) for i in range(1, 5))
        for suffix, values in zip(("average_delay", "average_cost_overrun", "delay_rate", "cost_overrun_rate"), selected):
            result.loc[indices, f"sector_{suffix}"] = values

    agency_groups = result["implementing_agency"].astype("string").fillna(missing_group)
    for value, indices in agency_groups.groupby(agency_groups).groups.items():
        local = known[known.implementing_agency.eq(value)] if value != missing_group and "implementing_agency" in known else known.iloc[0:0]
        stats = lookup(result.loc[indices, "snapshot_date"], local)
        fallback = [result.loc[indices, f"sector_{suffix}"].to_numpy(float) for suffix in ("average_delay", "average_cost_overrun", "delay_rate", "cost_overrun_rate")]
        selected = tuple(np.where(stats[0] >= minimum, stats[i], fallback[i - 1]) for i in range(1, 5))
        for suffix, values in zip(("average_delay", "average_cost_overrun", "delay_rate", "cost_overrun_rate"), selected):
            result.loc[indices, f"agency_{suffix}"] = values
    return result


def engineer_as_of_features(frame: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in DATE_COLUMNS + ["completion_date"]:
        if column not in data:
            data[column] = pd.NaT
        data[column] = pd.to_datetime(data[column], errors="coerce")
    for column in ["approved_cost_cr", "revised_cost_cr", "cumulative_expenditure_cr", "physical_progress"]:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data = data.sort_values(["canonical_project_id", "snapshot_date"]).reset_index(drop=True)
    data["project_size_category"] = pd.cut(data.approved_cost_cr, [-np.inf, 500, 2000, 10000, np.inf], labels=["small", "medium", "large", "mega"]).astype("string")
    data["expenditure_ratio"] = np.where(data.approved_cost_cr.gt(0), data.cumulative_expenditure_cr / data.approved_cost_cr, np.nan)
    data["cost_escalation_percentage"] = np.where(data.approved_cost_cr.gt(0), (data.revised_cost_cr - data.approved_cost_cr) / data.approved_cost_cr * 100, np.nan)
    lifecycle_start = data.planned_start_date.fillna(data.approval_date)
    data["lifecycle_start_source"] = np.where(data.planned_start_date.notna(), "planned_start_date", np.where(data.approval_date.notna(), "official_approval_date_proxy", None))
    data["planned_duration_days"] = (data.planned_completion_date - lifecycle_start).dt.days
    data["elapsed_duration_days"] = (data.snapshot_date - lifecycle_start).dt.days
    data["duration_ratio"] = np.where(data.planned_duration_days.gt(0), data.elapsed_duration_days / data.planned_duration_days, np.nan)
    current_target = data.revised_completion_date.fillna(data.planned_completion_date)
    data["schedule_slippage_days"] = (current_target - data.planned_completion_date).dt.days
    data["schedule_slippage_ratio"] = np.where(data.planned_duration_days.gt(0), data.schedule_slippage_days / data.planned_duration_days, np.nan)
    data["expected_progress_percentage"] = np.where(data.duration_ratio.notna(), np.minimum(100, np.maximum(0, 100 * data.duration_ratio)), np.nan)
    data["progress_deviation"] = data.physical_progress - data.expected_progress_percentage
    for _, group in data.groupby("canonical_project_id", sort=False):
        for months in (3, 6):
            data.loc[group.index, f"cost_growth_velocity_{months}m"] = _slope_as_of(group, "revised_cost_cr", months)
            data.loc[group.index, f"progress_velocity_{months}m"] = _slope_as_of(group, "physical_progress", months)
    data["cost_acceleration"] = data.cost_growth_velocity_3m - data.cost_growth_velocity_6m
    data["progress_acceleration"] = data.progress_velocity_3m - data.progress_velocity_6m
    data = _historical_priors(data, outcomes)
    data["lifecycle_stage"] = pd.cut(data.duration_ratio, [-np.inf, .30, .60, .90, np.inf], labels=["early", "mid", "late", "very_late"]).astype("string")
    return data


def assign_project_balanced_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign equal total training mass to each project in the final sampled cohort."""
    weighted = frame.copy()
    counts = weighted.groupby("canonical_project_id").canonical_project_id.transform("size")
    weighted["sample_weight"] = 1.0 / counts.clip(lower=1)
    sums = weighted.groupby("canonical_project_id").sample_weight.sum()
    if not sums.empty and not np.allclose(sums.to_numpy(dtype=float), 1.0, rtol=0, atol=1e-10):
        raise AssertionError("Per-project sample weights must sum to one after final snapshot sampling.")
    return weighted


def training_as_of_invariants(frame: pd.DataFrame) -> dict:
    snapshot = pd.to_datetime(frame.get("snapshot_date"), errors="coerce")
    completion = pd.to_datetime(frame.get("completion_date"), errors="coerce")
    post_completion = int((snapshot.notna() & completion.notna() & snapshot.ge(completion)).sum())
    unverified = int((~frame.get("identity_verified", pd.Series(False, index=frame.index)).fillna(False).astype(bool)).sum())
    return {
        "rows_checked": int(len(frame)),
        "post_or_at_completion_rows": post_completion,
        "unverified_identity_rows": unverified,
        "passed": post_completion == 0 and unverified == 0,
        "rules": [
            "every supervised snapshot must precede the linked completion date",
            "every supervised row must have an identity-verified completed outcome",
            _TRAJECTORY_SOURCE,
            _PRIOR_SOURCE,
        ],
    }


def build_training_dataset(snapshot_path: Path | None = None, outcome_path: Path = OUTCOMES) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = load_monthly_snapshots(snapshot_path)
    outcomes = pd.read_csv(outcome_path, dtype={"project_id": "string"}, low_memory=False)
    resolved, identity = resolve_identities(snapshots, outcomes); engineered = engineer_as_of_features(resolved, outcomes)
    trajectories = engineered.copy(); eligible = engineered[
        engineered.identity_verified.eq(True)
        & engineered[TARGETS].notna().all(axis=1)
        & engineered.snapshot_date.lt(engineered.completion_date)
    ].copy()
    # Deterministic quarterly sampling limits autocorrelation and compute cost.
    eligible["snapshot_quarter"] = eligible.snapshot_date.dt.to_period("Q").astype(str)
    eligible = eligible.sort_values("snapshot_date").drop_duplicates(["canonical_project_id", "snapshot_quarter"], keep="last")
    # Weight only after all filtering/sampling. Otherwise projects with denser raw
    # monthly histories receive less total weight than projects with sparse histories.
    eligible = assign_project_balanced_weights(eligible)
    invariants = training_as_of_invariants(eligible)
    if not invariants["passed"]:
        raise ValueError(f"Lifecycle as-of invariant failure: {invariants}")
    TRAJECTORIES.parent.mkdir(parents=True, exist_ok=True); trajectories.to_csv(TRAJECTORIES, index=False, date_format="%Y-%m-%d")
    eligible.to_csv(TRAINING_DATA, index=False, date_format="%Y-%m-%d"); identity.to_csv(IDENTITY_AUDIT, index=False)
    return eligible, identity