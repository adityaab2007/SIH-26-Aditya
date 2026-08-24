"""Canonical registry for experiment evidence and promotion decisions."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY = ROOT / "reports" / "experiments" / "registry.json"
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


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    if not path.exists():
        return _empty_registry()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Experiment registry is unreadable: {path}") from exc
    payload.setdefault("schema_version", 1)
    payload.setdefault("production_policy", _empty_registry()["production_policy"])
    payload.setdefault("experiments", [])
    return payload


def write_registry(payload: dict[str, Any], path: Path = DEFAULT_REGISTRY) -> None:
    """Atomically replace the registry so parallel crashes cannot half-write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def record_experiment(
    entry: dict[str, Any],
    *,
    path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    required = {"experiment_id", "name", "status", "decision"}
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError("experiment registry entry missing: " + ", ".join(missing))
    if entry["decision"] not in VALID_DECISIONS:
        raise ValueError(f"invalid experiment decision: {entry['decision']}")

    payload = load_registry(path)
    experiments = list(payload.get("experiments") or [])
    run_id = entry.get("run_id")
    evidence_id = entry.get("evidence_id")

    def same(item: dict[str, Any]) -> bool:
        if run_id and item.get("run_id") == run_id:
            return True
        if evidence_id and item.get("evidence_id") == evidence_id:
            return True
        return False

    experiments = [item for item in experiments if not same(item)]
    experiments.append(dict(entry))
    experiments.sort(key=lambda item: (str(item.get("experiment_id")), str(item.get("created_at") or "")))
    payload["experiments"] = experiments
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_registry(payload, path)
    return payload


def decision_from_improvement(improvement_percentage: float | None, threshold: float = 10.0) -> str:
    """Default cost-MAE decision rule used only when an experiment opts into it."""
    if improvement_percentage is None:
        return "PENDING"
    return "ACCEPTED" if float(improvement_percentage) >= float(threshold) else "REJECTED"
