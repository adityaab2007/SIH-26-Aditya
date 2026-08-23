"""Version-aware parsers for official PAIMANA monthly Flash Reports."""

from .base import CANONICAL_COLUMNS, ParseContext, ParseResult
from .registry import detect_parser, parse_report

__all__ = ["CANONICAL_COLUMNS", "ParseContext", "ParseResult", "detect_parser", "parse_report"]
