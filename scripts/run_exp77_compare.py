"""Run Experiment 77 against fresh production baselines for required windows."""
import json
import math
import os
from pathlib import Path

from backend.app.ml.experiments import adapter_exp77
from backend.app.ml.experiments.batch_compare import run_batch_comparison

DEFAULT_WINDOWS = ((2001, 2019), (2001, 2021))
REQUIRED = (
    "production_cost_mae",
    "experiment_cost_mae",
    "cost_improvement_percentage",
    "production_delay_mae",
    "experiment_delay_mae",
    "delay_improvement_percentage",
    "verdict",
)


def _selected_windows():
    raw = os.environ.get("EXPERIMENT_WINDOW", "").strip()
    if not raw:
        return DEFAULT_WINDOWS
    normalized = raw.replace("_", "-")
    parts = normalized.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid EXPERIMENT_WINDOW={raw!r}; expected 2001-2019 or 2001-2021")
    window = (int(parts[0]), int(parts[1]))
    if window not in DEFAULT_WINDOWS:
        raise ValueError(f"Unsupported experiment window: {window}")
    return (window,)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
    except ImportError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    out = Path("audit_outputs/exp77")
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for start, end in _selected_windows():
        key = f"{start}_{end}"
        payload = run_batch_comparison(adapter_exp77, start, end)
        overall = payload.get("overall_comparison") or {}
        missing = [k for k in REQUIRED if overall.get(k) is None]
        if missing:
            raise RuntimeError(f"{key} incomplete: {missing}")
        safe_payload = _safe(payload)
        (out / f"{key}.json").write_text(json.dumps(safe_payload, indent=2, allow_nan=False) + "\n")
        results[key] = _safe(overall)
        print(
            f"{key}: cost {overall['production_cost_mae']} -> {overall['experiment_cost_mae']} "
            f"({overall['cost_improvement_percentage']}%); delay {overall['production_delay_mae']} -> "
            f"{overall['experiment_delay_mae']} ({overall['delay_improvement_percentage']}%); "
            f"verdict={overall['verdict']}"
        )
    summary = _safe({"experiment_id": "exp_77", "windows": results, "promotion_allowed": False})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
