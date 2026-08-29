"""Run the complete Experiment 13 audit for the two required windows."""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.lifecycle_model_comparison_service import retrain_and_compare


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
    except ImportError:
        pass
    return value


def main() -> None:
    output = Path("audit_outputs") / "exp13"
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for start, end in ((2001, 2019), (2001, 2021)):
        result = retrain_and_compare(start, end, "exp_13")
        payload = _safe({"experiment_id": "exp_13", "experiment_name": "Recency-Weighted Project Training", "experiment_scope": "cost_delay", "window": f"{start}_{end}", **result})
        (output / f"{start}_{end}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        results[f"{start}_{end}"] = payload
        overall = payload.get("overall_comparison") or {}
        print(f"{start}-{end}: cost {overall.get('production_cost_mae')} -> {overall.get('experiment_cost_mae')} ({overall.get('cost_improvement_percentage')}%); delay {overall.get('production_delay_mae')} -> {overall.get('experiment_delay_mae')} ({overall.get('delay_improvement_percentage')}%); verdict={overall.get('verdict')}")
    verdicts = [((value.get("overall_comparison") or {}).get("verdict")) for value in results.values()]
    overall_verdict = "PROMOTION CANDIDATE" if all(value == "PROMOTION CANDIDATE" for value in verdicts) else "REGRESSION / DO NOT PROMOTE" if all(value == "REGRESSION / DO NOT PROMOTE" for value in verdicts) else "MIXED / NEEDS REVIEW"
    summary = {"experiment_id": "exp_13", "experiment_name": "Recency-Weighted Project Training", "experiment_scope": "cost_delay", "windows": results, "overall_verdict": overall_verdict, "promotion_allowed": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"OVERALL VERDICT: {overall_verdict}")


if __name__ == "__main__":
    main()
