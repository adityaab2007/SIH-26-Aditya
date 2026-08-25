"""Canonical registry for experiment evidence and promotion decisions.

The checked-in registry contains policy and historical evidence. New experiment
runs write one immutable entry file each, avoiding a shared-file merge hotspot
when multiple teammates run different experiments in parallel.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY = ROOT / "reports" / "experiments" / "registry.json"
DEFAULT_ENTRY_ROOT = ROOT / "reports" / "experiments" / "registry_entries"
VALID_DECISIONS = {"PENDING", "ACCEPTED", "REJECTED"}


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "production_policy": {
            "frontend_model_role": "production",
            "experiments_are_never_auto_promoted": True,
            "promotion_requires_explicit_acceptance": True,
        },
        "experiments": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Experiment registry evidence is unreadable: {path}") from exc


def _deduplicate(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    for entry in entries:
        identity = entry.get("run_id") or entry.get("evidence_id")
        key = (str(entry.get("experiment_id")), str(identity or ""))
        if identity and key in positions:
            result[positions[key]] = entry
        else:
            if identity:
                positions[key] = len(result)
            result.append(entry)
    result.sort(key=lambda item: (str(item.get("experiment_id")), str(item.get("created_at") or ""), str(item.get("run_id") or item.get("evidence_id") or "")))
    return result


def load_registry(
    path: Path = DEFAULT_REGISTRY,
    *,
    entries_root: Path | None = None,
) -> dict[str, Any]:
    payload = _empty_registry() if not path.exists() else _read_json(path)
    payload.setdefault("schema_version", 1)
    payload.setdefault("production_policy", _empty_registry()["production_policy"])
    entries = list(payload.get("experiments") or [])

    root = DEFAULT_ENTRY_ROOT if entries_root is None and path == DEFAULT_REGISTRY else entries_root
    if root and root.exists():
        for entry_path in sorted(root.glob("*/*.json")):
            entries.append(_read_json(entry_path))
    payload["experiments"] = _deduplicate(entries)
    return payload


def write_registry(payload: dict[str, Any], path: Path = DEFAULT_REGISTRY) -> None:
    """Atomically replace a registry file (used for policy/history maintenance)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _entry_filename(entry: dict[str, Any]) -> str:
    identity = str(entry.get("run_id") or entry.get("evidence_id") or "").strip()
    if not identity:
        raise ValueError("experiment registry entry requires run_id or evidence_id")
    if any(token in identity for token in ("/", "\\", "..")):
        raise ValueError("invalid experiment registry identity")
    return identity + ".json"


def record_experiment(
    entry: dict[str, Any],
    *,
    path: Path | None = None,
    entries_root: Path = DEFAULT_ENTRY_ROOT,
) -> dict[str, Any]:
    """Persist one run without forcing parallel experiments to edit one file.

    ``path`` is supported for tests/tools that intentionally maintain a single
    standalone registry. Normal experiment runs should omit it and receive an
    immutable per-run entry under ``registry_entries/<experiment_id>/``.
    """
    required = {"experiment_id", "name", "status", "decision"}
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError("experiment registry entry missing: " + ", ".join(missing))
    if entry["decision"] not in VALID_DECISIONS:
        raise ValueError(f"invalid experiment decision: {entry['decision']}")

    if path is not None:
        payload = load_registry(path, entries_root=None)
        experiments = list(payload.get("experiments") or [])
        identity = entry.get("run_id") or entry.get("evidence_id")
        experiments = [
            item for item in experiments
            if not identity or (item.get("run_id") or item.get("evidence_id")) != identity
        ]
        experiments.append(dict(entry))
        payload["experiments"] = _deduplicate(experiments)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_registry(payload, path)
        return payload

    experiment_id = str(entry["experiment_id"]).strip().lower().replace(" ", "_")
    if not experiment_id or any(token in experiment_id for token in ("/", "\\", "..")):
        raise ValueError("invalid experiment_id")
    destination = entries_root / experiment_id / _entry_filename(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(entry, indent=2, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text() != serialized:
            raise FileExistsError(f"Experiment evidence is immutable once written: {destination}")
    else:
        destination.write_text(serialized)
    return load_registry(DEFAULT_REGISTRY, entries_root=entries_root)


def decision_from_improvement(improvement_percentage: float | None, threshold: float = 10.0) -> str:
    """Default cost-MAE decision rule used only when an experiment opts into it."""
    if improvement_percentage is None:
        return "PENDING"
    return "ACCEPTED" if float(improvement_percentage) >= float(threshold) else "REJECTED"
