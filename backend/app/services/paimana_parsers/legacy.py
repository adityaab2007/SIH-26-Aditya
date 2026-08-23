"""Parser for observed legacy Sector-Wise analysis tables (2001-era layout)."""
from __future__ import annotations

import re

from .base import ParseContext, ParseResult, base_row, month_date, normalize_frame, number


class LegacySectorParser:
    version = "legacy-sector-v1"
    _start = re.compile(
        r"^\s*(?P<seq>\d+)\.?\s+(?P<name>.+?)\s+(?P<approval>\d{1,2}/\d{4})\s+"
        r"(?P<original>[\d,]+(?:\.\d+)?)\s+(?P<overrun>-?[\d,]+(?:\.\d+)?)\s+"
        r"(?P<expenditure>[\d,]+(?:\.\d+)?)\s+(?P<planned>\d{1,2}/\d{4}|-)\s+"
        r"(?P<delay>-?\d+|-)\s+(?P<milestones>\d+\s*/\s*\d+)", re.I,
    )

    @classmethod
    def matches(cls, text: str) -> bool:
        lowered = text.lower()
        return "sector-wise analysis of projects" in lowered and "[anticipated]" in lowered

    @classmethod
    def parse(cls, text: str, context: ParseContext) -> ParseResult:
        rows: list[dict] = []
        current_sector = None
        for page_no, page in enumerate(text.split("\f"), 1):
            lines = page.splitlines()
            for index, raw in enumerate(lines):
                line = raw.rstrip()
                stripped = line.strip()
                if re.fullmatch(r"[A-Z][A-Z &/()\-]{2,}", stripped) and not any(
                    word in stripped for word in ("TOTAL", "SECTOR", "PROJECT", "COST", "DATE")
                ):
                    current_sector = stripped.title()
                    continue
                match = cls._start.match(line)
                if not match:
                    continue
                continuation = " ".join(x.strip() for x in lines[index + 1:index + 5])
                project_id_match = re.search(r"[\[(]([A-Z]?\d{8,9})[\])]", continuation)
                agency_match = re.search(r"\(([A-Z][A-Z0-9 &.\-/]{1,50})\)", continuation)
                anticipated_cost = next((number(v) for v in re.findall(r"\[([\d,]+(?:\.\d+)?)\]", continuation) if number(v) is not None), None)
                anticipated_date = next((month_date(v) for v in re.findall(r"\[(\d{1,2}/\d{4})\]", continuation) if month_date(v)), None)
                name_parts = [match.group("name").strip()]
                for extra in lines[index + 1:index + 4]:
                    fragment = extra[:42].strip()
                    if fragment and not fragment.startswith(("(", "[")) and not re.match(r"^(Total|\d+\.)", fragment, re.I):
                        name_parts.append(fragment)
                original = number(match.group("original"))
                expenditure = number(match.group("expenditure"))
                revised = anticipated_cost
                financial = expenditure / revised * 100 if expenditure is not None and revised else None
                rows.append({
                    **base_row(context, page=page_no),
                    "project_id": project_id_match.group(1) if project_id_match else None,
                    "project_name": re.sub(r"\s+", " ", " ".join(name_parts)).strip(" ,-"),
                    "sector": current_sector,
                    "implementing_agency": agency_match.group(1).strip() if agency_match else None,
                    "approved_cost_cr": original,
                    "approval_date": month_date(match.group("approval")),
                    "revised_cost_cr": revised,
                    "cumulative_expenditure_cr": expenditure,
                    "planned_completion_date": month_date(match.group("planned")),
                    "revised_completion_date": anticipated_date,
                    "physical_progress": None,
                    "financial_progress": round(financial, 4) if financial is not None else None,
                    "milestone_status": match.group("milestones").replace(" ", ""),
                    "current_schedule_status": "delayed" if number(match.group("delay")) and number(match.group("delay")) > 0 else None,
                    "parser_version": cls.version,
                })
        return ParseResult(normalize_frame(rows), cls.version, [] if rows else ["No legacy project rows matched"])
