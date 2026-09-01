"""Run Experiment 13 against fresh production baselines for required windows."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from backend.app.ml.experiments import adapter_exp13
from backend.app.ml.experiments.batch_compare import run_batch_comparison

DEFAULT_WINDOWS = ((2001, 2019), (2001, 2021))
REQUIRED_METRICS = (
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


def _validate_window(window: str, payload: dict) -> dict:
    overall = payload.get("overall_comparison") or {}
    missing = [key for key in REQUIRED_METRICS if overall.get(key) is None]
    if missing:
        raise RuntimeError(f"{window} comparison incomplete; missing: {', '.join(missing)}")
    return overall


def main() -> None:
    output = Path("audit_outputs") / "exp13"
    output.mkdir(parents=True, exist_ok=True)
    results = {}

    for start, end in _selected_windows():
        window = f"{start}_{end}"
        result = run_batch_comparison(adapter_exp13, start, end)
        payload = _safe({
            "experiment_id": "exp_13",
            "experiment_name": "Recency-Weighted Project Training",
            "experiment_scope": "cost_delay",
            "window": window,
            **result,
        })
        overall = _validate_window(window, payload)
        (output / f"{window}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        results[window] = payload
        print(
            f"{start}-{end}: cost {overall['production_cost_mae']} -> {overall['experiment_cost_mae']} "
            f"({overall['cost_improvement_percentage']}%); delay {overall['production_delay_mae']} -> "
            f"{overall['experiment_delay_mae']} ({overall['delay_improvement_percentage']}%); "
            f"verdict={overall['verdict']}"
        )

    verdicts = [(value.get("overall_comparison") or {}).get("verdict") for value in results.values()]
    if all(value == "PROMOTION CANDIDATE" for value in verdicts):
        overall_verdict = "PROMOTION CANDIDATE"
    elif all(value == "REGRESSION / DO NOT PROMOTE" for value in verdicts):
        overall_verdict = "REGRESSION / DO NOT PROMOTE"
    else:
        overall_verdict = "MIXED / NEEDS REVIEW"

    summary = _safe({
        "experiment_id": "exp_13",
        "experiment_name": "Recency-Weighted Project Training",
        "experiment_scope": "cost_delay",
        "windows": results,
        "overall_verdict": overall_verdict,
        "promotion_allowed": False,
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"WINDOW VERDICT: {overall_verdict}")


if __name__ == "__main__":
    main()
