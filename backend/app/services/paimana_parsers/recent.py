"""Parser for the current PAIMANA Project List layout."""
from __future__ import annotations

import re

from .base import ParseContext, ParseResult, base_row, month_date, normalize_frame, number


class RecentProjectListParser:
    version = "recent-project-list-v3"

    @classmethod
    def matches(cls, text: str) -> bool:
        return "project list: ongoing projects as of" in text.lower()

    @classmethod
    def parse(cls, text: str, context: ParseContext) -> ParseResult:
        marker = text.lower().find("project list: ongoing projects as of")
        pages = text[marker:].split("\f") if marker >= 0 else []
        start_re = re.compile(r"^(?P<head>.*?)\s{2,}(?P<approval>\d{1,2}[-/]\d{4})\s+(?P<completion>\d{1,2}[-/]\d{4}|/)\s+(?P<cost>[\d,]+(?:\.\d+)?)\s+(?P<expenditure>[\d,]+(?:\.\d+)?)\s+(?P<progress>\d+(?:\.\d+)?|-)\s*$")
        rows: list[dict] = []
        for page_offset, page in enumerate(pages, 1):
            lines = page.splitlines(); starts = []
            for index, line in enumerate(lines):
                match = start_re.match(line)
                sequence = re.search(r"(?<!\S)(\d{1,4})\s+(.+)$", match.group("head")) if match else None
                if match and sequence:
                    starts.append((index, match, sequence))
            for position, (line_index, match, sequence) in enumerate(starts):
                block = lines[line_index + 1:starts[position + 1][0] if position + 1 < len(starts) else len(lines)]
                joined = "\n".join(block)
                code_match = re.search(r"\(([A-Z]?\d{8,9})\s*\)", joined)
                if not code_match:
                    continue
                parenthetical_values = re.findall(r"\(([^{}()]{2,80})\)", joined)
                agency = next((value.strip() for value in parenthetical_values if not (
                    re.fullmatch(r"[A-Z]?\d{8,9}", value.strip())
                    or re.fullmatch(r"(?:N\.A\.|\d{1,2}[-/]\d{4}|[A-Za-z]{3}-\d{2}|[\d,.]+)", value.strip(), re.I)
                )), None)
                braces = re.findall(r"\{([^}]+)\}", joined); parentheses = re.findall(r"\(([^)]+)\)", joined)
                anticipated_date = next((month_date(x) for x in braces if month_date(x)), None)
                anticipated_cost = next((number(x) for x in braces if number(x) is not None), None)
                revised_date = next((month_date(x) for x in parentheses if month_date(x)), None)
                original = number(match.group("cost")); expenditure = number(match.group("expenditure")); progress = number(match.group("progress"))
                effective = anticipated_cost or original
                financial = expenditure / effective * 100 if expenditure is not None and effective else None
                rows.append({
                    **base_row(context, page=page_offset), "project_id": code_match.group(1),
                    "project_name": re.sub(r"\s+", " ", sequence.group(2)).strip(),
                    "implementing_agency": agency,
                    "approved_cost_cr": original, "revised_cost_cr": anticipated_cost,
                    "approval_date": month_date(match.group("approval")),
                    "cumulative_expenditure_cr": expenditure, "planned_completion_date": month_date(match.group("completion")),
                    "revised_completion_date": revised_date or anticipated_date, "physical_progress": progress,
                    "financial_progress": round(financial, 4) if financial is not None else None,
                    "parser_version": cls.version,
                })
        return ParseResult(normalize_frame(rows), cls.version, [] if rows else ["No recent project-code rows matched"])
