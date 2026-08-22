"""Canonical SIH26103 future-target entrypoint.

The implementation remains in ``forward_labels`` for backward compatibility.
"""
from .forward_labels import build_forward_labels

__all__ = ["build_forward_labels"]
