"""Parsers for code-bearing classic and redesigned PAIMANA project tables."""
from __future__ import annotations

import re

from .base import ParseContext, ParseResult, base_row, month_date, normalize_frame, number


def _identity(block: str) -> tuple[str | None, str | None, str | None]:
    code = re.search(r"[\[(]([A-Z]?\d{8,9})[\])]", block)
    tail = re.search(r"[\[(][A-Z]?\d{8,9}[\])][ \t]*,?[ \t]*([^,\n]+)(?:,([^,\n]+))?", block)
    agency = tail.group(1).strip(" ,-") if tail else None
    state = tail.group(2).strip(" ,-") if tail and tail.group(2) else None
    if agency and (month_date(agency) or re.match(r"^\d+\s+", agency)):
        agency = None
        state = None
    return (
        code.group(1) if code else None,
        agency,
        state,
    )


class ClassicProjectParser:
    version = "classic-code-v3"
    _start = re.compile(
        r"^\s*(?P<seq>\d+)\s+(?P<name>.+?)\s+(?P<approval>\d{1,2}/\d{4})\s+"
        r"(?P<original>[\d,]+(?:\.\d+)?)\s+(?P<anticipated>[\d,]+(?:\.\d+)?)\s+"
        r"(?P<expenditure>[\d,]+(?:\.\d+)?)\s+(?P<planned>\d{1,2}/\d{4}|-)\s+"
        r"(?P<anticipated_date>\d{1,2}/\d{4}|-)", re.I,
    )

    @classmethod
    def matches(cls, text: str) -> bool:
        lowered = text.lower()
        classic_heading = "sector wise details" in lowered or "detail of ongoing projects" in lowered
        compact_heading = bool(re.search(
            r"S\.No\s+Project\s+Approval\s+Cost\s+Cost\s+ture\s+Revised", text, re.I,
        )) and "cumm." in lowered
        return (classic_heading or compact_heading) and "date of" in lowered and not re.search(
            r"Original\s*/.*\n.*Revised\s*/.*\n.*Anticipated", text, re.I,
        )

    @classmethod
    def parse(cls, text: str, context: ParseContext) -> ParseResult:
        rows: list[dict] = []
        current_sector = None
        marker = re.search(r"Sector Wise Details|Detail of ongoing Projects|S\.No\s+Project", text, re.I)
        section = text[marker.start():] if marker else text
        section = re.split(r"\n\s*(?:STATUS OF CENTRAL|List of Projects Without|ANNEXURE\s+-?\s*X)", section, maxsplit=1, flags=re.I)[0]
        pages = section.split("\f")
        for page_no, page in enumerate(pages, 1):
            lines = page.splitlines()
            for index, raw in enumerate(lines):
                stripped = raw.strip()
                if re.fullmatch(r"[A-Z][A-Z &/()\-]{2,}", stripped) and not any(x in stripped for x in ("DETAIL", "TOTAL", "DATE", "PROJECT")):
                    current_sector = stripped.title()
                match = cls._start.match(raw)
                if not match:
                    continue
                block = "\n".join([match.group("name"), *lines[index + 1:index + 5]])
                code, agency, state = _identity(block)
                name = re.split(r"\s+-\s+[\[(]?[A-Z]?\d{8,9}", block)[0]
                name = re.sub(r"\s+", " ", name).strip(" ,-\n")
                original = number(match.group("original"))
                anticipated = number(match.group("anticipated"))
                expenditure = number(match.group("expenditure"))
                financial = expenditure / anticipated * 100 if expenditure is not None and anticipated else None
                rows.append({
                    **base_row(context, page=page_no), "project_id": code, "project_name": name,
                    "sector": current_sector, "implementing_agency": agency, "state": state,
                    "approved_cost_cr": original, "revised_cost_cr": anticipated,
                    "approval_date": month_date(match.group("approval")),
                    "cumulative_expenditure_cr": expenditure,
                    "planned_completion_date": month_date(match.group("planned")),
                    "revised_completion_date": month_date(match.group("anticipated_date")),
                    "financial_progress": round(financial, 4) if financial is not None else None,
                    "parser_version": cls.version,
                })
        return ParseResult(normalize_frame(rows), cls.version, [] if rows else ["No classic ongoing-project rows matched"])


class RedesignedProjectParser:
    version = "redesigned-code-v1"
    _start = re.compile(
        r"^\s*(?P<seq>\d+)\s+(?P<name>.+?)\s+(?P<approval>\d{1,2}/\d{4})\s+"
        r"(?P<planned>\d{1,2}/\d{4}|-)\s+(?P<original>[\d,]+(?:\.\d+)?)\s+"
        r"(?P<expenditure>[\d,]+(?:\.\d+)?)\s+(?P<milestones>\d+\s*/\s*\d+)", re.I,
    )

    @classmethod
    def matches(cls, text: str) -> bool:
        return "detail of ongoing projects" in text.lower() and bool(re.search(
            r"Original\s*/.*\n.*Revised\s*/.*\n.*Anticipated", text, re.I,
        ))

    @classmethod
    def parse(cls, text: str, context: ParseContext) -> ParseResult:
        rows: list[dict] = []
        current_sector = None
        marker = re.search(r"Detail of ongoing Projects", text, re.I)
        section = text[marker.start():] if marker else text
        section = re.split(r"\n\s*(?:STATUS OF CENTRAL|ANNEXURE\s+-?\s*X)", section, maxsplit=1, flags=re.I)[0]
        for page_no, page in enumerate(section.split("\f"), 1):
            lines = page.splitlines()
            for index, raw in enumerate(lines):
                stripped = raw.strip()
                if re.fullmatch(r"[A-Z][A-Z &/()\-]{2,}", stripped) and not any(x in stripped for x in ("DETAIL", "TOTAL", "DATE", "PROJECT")):
                    current_sector = stripped.title()
                match = cls._start.match(raw)
                if not match:
                    continue
                continuation = "\n".join(lines[index + 1:index + 6])
                block = match.group("name") + "\n" + continuation
                code, agency, state = _identity(block)
                revised_costs = [number(v) for v in re.findall(r"[\[(]([\d,]+(?:\.\d+)?)[\])]", continuation)]
                revised_cost = next((v for v in reversed(revised_costs) if v is not None and v >= 0), None)
                revised_dates = [month_date(v) for v in re.findall(r"[\[(](\d{1,2}/\d{4})[\])]", continuation)]
                revised_date = next((v for v in reversed(revised_dates) if v), None)
                original = number(match.group("original")); expenditure = number(match.group("expenditure"))
                effective = revised_cost or original
                financial = expenditure / effective * 100 if expenditure is not None and effective else None
                name = re.split(r"\s+-\s+[\[(]?[A-Z]?\d{8,9}", block)[0]
                rows.append({
                    **base_row(context, page=page_no), "project_id": code,
                    "project_name": re.sub(r"\s+", " ", name).strip(" ,-\n"), "sector": current_sector,
                    "implementing_agency": agency, "state": state, "approved_cost_cr": original,
                    "approval_date": month_date(match.group("approval")),
                    "revised_cost_cr": revised_cost, "cumulative_expenditure_cr": expenditure,
                    "planned_completion_date": month_date(match.group("planned")),
                    "revised_completion_date": revised_date, "financial_progress": round(financial, 4) if financial is not None else None,
                    "milestone_status": match.group("milestones").replace(" ", ""), "parser_version": cls.version,
                })
        return ParseResult(normalize_frame(rows), cls.version, [] if rows else ["No redesigned project rows matched"])
