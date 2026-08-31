"""Run Experiment 13 against fresh production baselines for both required windows."""
from __future__ import annotations

import json
import math
from pathlib import Path

from backend.app.services.lifecycle_model_comparison_service import retrain_and_compare

WINDOWS = ((2001, 2019), (2001, 2021))
REQUIRED_METRICS = (
    "production_cost_mae",
    "experiment_cost_mae",
    "cost_improvement_percentage",
    "production_delay_mae",
    "experiment_delay_mae",
    "delay_improvement_percentage",
    "verdict",
)


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

    for start, end in WINDOWS:
        window = f"{start}_{end}"
        result = retrain_and_compare(start, end, "exp_13")
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
            f"{start}-{end}: "
            f"cost {overall['production_cost_mae']} -> {overall['experiment_cost_mae']} "
            f"({overall['cost_improvement_percentage']}%); "
            f"delay {overall['production_delay_mae']} -> {overall['experiment_delay_mae']} "
            f"({overall['delay_improvement_percentage']}%); "
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
        "comparison_contract": {
            "windows": ["2001_2019", "2001_2021"],
            "fresh_production_baseline": True,
            "same_held_out_cohort": True,
            "metrics": list(REQUIRED_METRICS),
        },
        "windows": results,
        "overall_verdict": overall_verdict,
        "promotion_allowed": False,
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"OVERALL VERDICT: {overall_verdict}")


if __name__ == "__main__":
    main()
