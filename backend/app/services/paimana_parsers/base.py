from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

import pandas as pd

CANONICAL_COLUMNS = [
    "project_id", "project_name", "snapshot_date", "financial_year", "report_month",
    "sector", "ministry", "implementing_agency", "state", "approved_cost_cr",
    "revised_cost_cr", "cumulative_expenditure_cr", "approval_date", "planned_start_date",
    "planned_completion_date", "revised_completion_date", "actual_completion_date",
    "current_schedule_status", "physical_progress", "financial_progress",
    "milestone_status", "source_report", "source_url", "source_page", "parser_version",
]


@dataclass(frozen=True)
class ParseContext:
    financial_year: str
    report_month: pd.Timestamp
    source_report: str
    source_url: str


@dataclass
class ParseResult:
    frame: pd.DataFrame
    parser_version: str
    warnings: list[str]


class ReportParser(Protocol):
    version: str

    @classmethod
    def matches(cls, text: str) -> bool: ...

    @classmethod
    def parse(cls, text: str, context: ParseContext) -> ParseResult: ...


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def normalize_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame[CANONICAL_COLUMNS]


def number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    if value in {"", "-", "(-)", "N.A.", "NA"}:
        return None
    value = value.strip("()[]")
    try:
        return float(value)
    except ValueError:
        return None


def month_date(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip("()[]")
    if value in {"", "-", "/", "N.A."}:
        return None
    value = re.sub(r"^(\d{1,2})-(\d{4})$", r"\1/\2", value)
    parsed = pd.to_datetime(value, format="%m/%Y", errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def base_row(context: ParseContext, *, page: int) -> dict:
    return {
        "snapshot_date": context.report_month.strftime("%Y-%m-%d"),
        "financial_year": context.financial_year,
        "report_month": context.report_month.strftime("%B"),
        "source_report": context.source_report,
        "source_url": context.source_url,
        "source_page": page,
    }
