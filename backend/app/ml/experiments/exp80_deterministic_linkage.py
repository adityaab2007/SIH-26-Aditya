from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd

EXPERIMENT_ID = "exp_80"
EXPERIMENT_SEQUENCE = 80
OUTCOME_FIELDS = {
    "actual_cost_overrun_percentage",
    "actual_delay_days",
    "completion_date",
    "reported_completion_expenditure_cr",
}


def _text(value):
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_code(value):
    return re.sub(r"[^a-z0-9]", "", _text(value)).upper()


def normalize_title(value):
    text = " " + _text(value) + " "
    replacements = {
        " rd ": " road ",
        " nh ": " national highway ",
        " rly ": " railway ",
        " stn ": " station ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def normalize_state(value):
    text = _text(value)
    aliases = {
        "orissa": "odisha",
        "uttaranchal": "uttarakhand",
        "nct of delhi": "delhi",
    }
    return aliases.get(text, text)


def normalize_sector(value):
    return _text(value).replace("&", "and")


def normalize_agency(value):
    text = _text(value)
    aliases = {
        "nhai": "national highways authority of india",
        "ircon": "ircon international limited",
    }
    return aliases.get(text, text)


def parse_cost(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def parse_date(value):
    return pd.to_datetime(value, errors="coerce")


def cost_compatible(left, right, rel_tol=0.01, abs_tol=0.05):
    left, right = parse_cost(left), parse_cost(right)
    return bool(
        pd.notna(left)
        and pd.notna(right)
        and abs(float(left) - float(right))
        <= max(abs_tol, rel_tol * max(abs(float(left)), abs(float(right)), 1.0))
    )


def date_compatible(left, right, tolerance_days=31):
    left, right = parse_date(left), parse_date(right)
    return bool(pd.notna(left) and pd.notna(right) and abs((left - right).days) <= tolerance_days)


def _prep(frame):
    result = frame.copy()
    result["_code"] = result.get(
        "project_id", result.get("project_code", pd.Series(index=result.index, dtype=object))
    ).map(normalize_code)
    result["_title"] = result.get(
        "project_name", result.get("title", pd.Series(index=result.index, dtype=object))
    ).map(normalize_title)
    result["_state"] = result.get("state", pd.Series("", index=result.index)).map(normalize_state)
    result["_sector"] = result.get("sector", pd.Series("", index=result.index)).map(normalize_sector)
    result["_agency"] = result.get(
        "implementing_agency", result.get("agency", pd.Series("", index=result.index))
    ).map(normalize_agency)
    return result


def resolve_links(snapshots, outcomes):
    """Resolve only unique, deterministic identity candidates.

    The rules are deliberately hierarchical. A non-unique title/state/sector/cost
    candidate set is *not* accepted; it is refined using agency and planned date.
    No completion outcome or target field participates in matching.
    """
    snapshot_rows, outcome_rows = _prep(snapshots), _prep(outcomes)
    accepted = []
    diagnostics = []

    for snapshot_index, row in snapshot_rows.iterrows():
        candidates = []

        if row["_code"]:
            exact_code = outcome_rows[outcome_rows._code.eq(row["_code"])]
            if len(exact_code):
                candidates = [(1, index, "exact_project_code") for index in exact_code.index]

        if not candidates and row["_title"]:
            title_candidates = outcome_rows[
                outcome_rows._title.eq(row["_title"])
                & outcome_rows._state.eq(row["_state"])
                & outcome_rows._sector.eq(row["_sector"])
            ]
            if len(title_candidates):
                compatible = title_candidates[
                    [
                        cost_compatible(row.get("approved_cost_cr"), candidate.get("approved_cost_cr"))
                        for _, candidate in title_candidates.iterrows()
                    ]
                ]
                if len(compatible) == 1:
                    candidates = [(2, compatible.index[0], "title_state_sector_cost")]
                elif len(compatible) > 1:
                    refined = []
                    for index, candidate in compatible.iterrows():
                        same_agency = bool(row["_agency"] and candidate["_agency"] == row["_agency"])
                        planned_snapshot = row.get(
                            "planned_completion_date", row.get("planned_commissioning_date")
                        )
                        planned_outcome = candidate.get(
                            "planned_commissioning_date", candidate.get("planned_completion_date")
                        )
                        if same_agency and date_compatible(planned_snapshot, planned_outcome):
                            refined.append(index)
                    if len(refined) == 1:
                        candidates = [(3, refined[0], "agency_state_sector_cost_date_title")]
                    else:
                        candidates = [(2, index, "title_state_sector_cost") for index in compatible.index]

        unique = {index: (tier, rule) for tier, index, rule in candidates}
        snapshot_id = row.get("snapshot_id", snapshot_index)
        if len(unique) == 1:
            outcome_index, (tier, rule) = next(iter(unique.items()))
            outcome_row = outcome_rows.loc[outcome_index]
            accepted.append(
                {
                    "snapshot_id": snapshot_id,
                    "completion_id": outcome_row.get(
                        "completion_id", outcome_row.get("project_id", outcome_index)
                    ),
                    "rule": rule,
                    "tier": tier,
                    "matched_fields": ["project_code"]
                    if tier == 1
                    else ["project_title", "state", "sector", "approved_cost"]
                    + (["agency", "planned_commissioning_date"] if tier == 3 else []),
                    "confidence_category": "VERY_HIGH" if tier == 1 else "HIGH",
                    "ambiguity_count": 1,
                }
            )
        else:
            diagnostics.append(
                {
                    "snapshot_id": snapshot_id,
                    "status": "ambiguous" if len(unique) > 1 else "unmatched",
                    "ambiguity_count": len(unique),
                }
            )

    return pd.DataFrame(accepted), pd.DataFrame(diagnostics)


def linkage_diagnostics(original_links, new_links, diagnostics):
    return {
        "original_matched_projects": int(original_links.completion_id.nunique()) if len(original_links) else 0,
        "original_matched_snapshots": int(len(original_links)),
        "newly_matched_projects": int(
            len(set(new_links.completion_id) - set(original_links.completion_id))
        )
        if len(new_links)
        else 0,
        "newly_matched_snapshots": int(max(0, len(new_links) - len(original_links))),
        "total_matched_projects_after_expansion": int(new_links.completion_id.nunique()) if len(new_links) else 0,
        "total_matched_snapshots_after_expansion": int(len(new_links)),
        "ambiguous_snapshots": int((diagnostics.status == "ambiguous").sum()) if len(diagnostics) else 0,
        "unmatched_snapshots": int((diagnostics.status == "unmatched").sum()) if len(diagnostics) else 0,
        "matches_by_rule": dict(Counter(new_links.rule)) if len(new_links) else {},
        "matches_by_tier": {str(key): int(value) for key, value in Counter(new_links.tier).items()}
        if len(new_links)
        else {},
        "uniqueness_rate": float(new_links.ambiguity_count.eq(1).mean()) if len(new_links) else 0.0,
    }


def assert_no_outcome_leakage(columns):
    bad = OUTCOME_FIELDS.intersection(columns)
    if bad:
        raise ValueError("Outcome leakage in identity matching: " + ",".join(sorted(bad)))
