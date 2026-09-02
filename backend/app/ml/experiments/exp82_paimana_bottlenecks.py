from __future__ import annotations

import re

import pandas as pd

EXPERIMENT_ID = "exp_82"
EXPERIMENT_SEQUENCE = 82

TEXT_COLUMN_PATTERNS = (
    "remark",
    "reason",
    "bottleneck",
    "issue",
    "status",
    "narrative",
    "comment",
    "observation",
)

CATEGORIES = {
    "land_acquisition": [r"\bland acquisition\b", r"\bland.*acquir"],
    "forest_clearance": [r"\bforest clearance\b", r"\bforest.*clear"],
    "environmental_clearance": [r"\benvironment(?:al)? clearance\b", r"\benvironment.*clear"],
    "funding": [r"\bfund(?:ing|s)? constraint", r"\bshortage of funds\b", r"\bfunds? not available\b"],
    "litigation": [r"\blitigation\b", r"\bcourt case\b", r"\bstay order\b"],
    "contractor": [r"\bcontractor\b.*(?:delay|issue|problem|slow|default)", r"\bpoor contractor performance\b"],
    "utility_shifting": [r"\butility shifting\b", r"\bshifting of utilities\b"],
    "railway_crossing": [r"\brail(?:way)? crossing\b", r"\brailway approval\b"],
    "dpr": [r"\bdpr\b", r"\bdetailed project report\b"],
    "approval_sanction": [r"\bapproval delay\b", r"\bsanction delay\b", r"\bawaiting approval\b"],
    "tender_procurement": [r"\btender\b", r"\bprocurement\b"],
    "law_order": [r"\blaw and order\b"],
    "public_obstruction": [r"\blocal obstruction\b", r"\bpublic protest\b"],
    "design": [r"\bdesign (?:issue|change|delay)\b"],
    "material_shortage": [r"\bmaterial shortage\b"],
    "manpower_shortage": [r"\bmanpower shortage\b", r"\blabou?r shortage\b"],
    "geological_site": [r"\bgeolog(?:y|ical)\b", r"\bsite condition\b"],
}

NEGATION = re.compile(r"\b(?:no|not|without|resolved|cleared)\b.{0,25}$")


def discover_text_columns(columns):
    return [column for column in columns if any(pattern in str(column).lower() for pattern in TEXT_COLUMN_PATTERNS)]


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def category_active(text, patterns):
    normalized = normalize_text(text)
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match and not NEGATION.search(normalized[max(0, match.start() - 30) : match.start()]):
            return 1
    return 0


def _asof_first_seen_date(frame, *, project_col, date_col, active_col):
    """Return the first observed active date using current/past rows only.

    A full-group ``transform('min')`` is intentionally not used because that
    leaks a future first occurrence backward into earlier lifecycle snapshots.
    """
    occurrence_number = frame.groupby(project_col, sort=False)[active_col].cumsum()
    first_marker = frame[date_col].where(frame[active_col].eq(1) & occurrence_number.eq(1))
    return first_marker.groupby(frame[project_col], sort=False).ffill()


def add_bottleneck_features(
    frame,
    project_col="canonical_project_id",
    date_col="snapshot_date",
    text_columns=None,
):
    x = frame.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    columns = text_columns or discover_text_columns(x.columns)
    if not columns:
        raise ValueError("No PAIMANA narrative/delay-reason columns found in actual snapshot schema.")

    x["_issue_text"] = x[columns].fillna("").astype(str).agg(" | ".join, axis=1).map(normalize_text)
    x = x.sort_values([project_col, date_col], kind="stable").copy()

    for category, patterns in CATEGORIES.items():
        active = f"issue_{category}_active"
        seen = f"issue_{category}_seen_before"
        months = f"issue_{category}_months_active"
        since = f"issue_{category}_months_since_first_seen"

        x[active] = x["_issue_text"].map(lambda text: category_active(text, patterns))
        x[seen] = x.groupby(project_col, sort=False)[active].cummax()
        x[months] = x.groupby(project_col, sort=False)[active].cumsum()

        first_seen = _asof_first_seen_date(
            x,
            project_col=project_col,
            date_col=date_col,
            active_col=active,
        )
        x[since] = ((x[date_col] - first_seen).dt.days / 30.4375).where(first_seen.notna())

    active_columns = [column for column in x if column.endswith("_active")]
    seen_columns = [column for column in x if column.endswith("_seen_before")]
    months_active_columns = [column for column in x if column.endswith("_months_active")]

    x["issue_text_present"] = x["_issue_text"].ne("").astype(int)
    x["issue_count_active"] = x[active_columns].sum(axis=1)
    x["issue_count_seen"] = x[seen_columns].sum(axis=1)
    x["number_of_distinct_bottleneck_categories"] = x["issue_count_seen"]
    x["repeated_bottleneck_flag"] = (x[months_active_columns].max(axis=1) >= 2).astype(int)

    any_active = x["issue_count_active"].gt(0).astype(int)
    x["_any_bottleneck_active"] = any_active
    first_bottleneck = _asof_first_seen_date(
        x,
        project_col=project_col,
        date_col=date_col,
        active_col="_any_bottleneck_active",
    )
    x["months_since_first_bottleneck"] = (
        (x[date_col] - first_bottleneck).dt.days / 30.4375
    ).where(first_bottleneck.notna())

    return x.drop(columns=["_issue_text", "_any_bottleneck_active"]), columns


def assert_asof_only(frame, project_col="canonical_project_id", date_col="snapshot_date"):
    ordered = frame.sort_values([project_col, date_col])
    if ordered.groupby(project_col)[date_col].apply(lambda series: not series.is_monotonic_increasing).any():
        raise ValueError("Temporal ordering failure")
    return True
