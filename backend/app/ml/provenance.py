"""Immutable provenance helpers for lifecycle model artifacts."""
from __future__ import annotations

from hashlib import sha256
import json
import os
import subprocess
import uuid
from pathlib import Path

import pandas as pd


def new_run_id() -> str:
    return uuid.uuid4().hex


def git_commit_sha(root: Path) -> str | None:
    """Return the source commit when available without making provenance mandatory on .git."""
    explicit = os.getenv("GITHUB_SHA") or os.getenv("VERCEL_GIT_COMMIT_SHA")
    if explicit:
        return explicit.strip() or None
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def feature_schema_fingerprint(features: list[str]) -> str:
    payload = json.dumps(list(features), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(payload).hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash the exact tabular cohort independent of incoming row/column order.

    The hash intentionally includes sample weights and targets when present so a
    weighting or target-construction change invalidates the fingerprint too.
    """
    if frame.empty:
        return sha256(b"empty-dataframe").hexdigest()
    columns = sorted(str(column) for column in frame.columns)
    stable = frame.loc[:, columns].copy()
    sort_columns = [name for name in ("canonical_project_id", "snapshot_date", "completion_year", "project_id") if name in stable]
    if sort_columns:
        stable = stable.sort_values(sort_columns, kind="mergesort", na_position="last")
    else:
        stable = stable.sort_values(columns, kind="mergesort", na_position="last")
    payload = stable.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        na_rep="<NA>",
        float_format="%.12g",
        lineterminator="\n",
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprints(directory: Path, names: list[str]) -> dict[str, str | None]:
    return {name: file_sha256(directory / name) for name in names}
