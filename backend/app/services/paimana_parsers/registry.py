from __future__ import annotations

from .base import ParseContext, ParseResult, ReportParser, empty_frame
from .classic import ClassicProjectParser, RedesignedProjectParser
from .legacy import LegacySectorParser
from .recent import RecentProjectListParser

PARSERS: tuple[type[ReportParser], ...] = (
    RecentProjectListParser,
    RedesignedProjectParser,
    ClassicProjectParser,
    LegacySectorParser,
)


def detect_parser(text: str) -> type[ReportParser] | None:
    return next((parser for parser in PARSERS if parser.matches(text)), None)


def parse_report(text: str, context: ParseContext) -> ParseResult:
    parser = detect_parser(text)
    if parser is None:
        return ParseResult(empty_frame(), "unrecognized", ["No registered parser matched the report structure"])
    return parser.parse(text, context)
