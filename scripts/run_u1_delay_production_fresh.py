"""Freshly train U1 Delay production and verify Cost stays unchanged."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.production_u1_delay_baseline import train_window_with_promoted_cost_and_delay

EXPECTED = {
    2019: {"cost": 27.801, "base_delay": 470.612, "u1_delay": 438.098},
    2021: {"cost": 26.079, "base_delay": 407.571, "u1_delay": 359.379},
}
TOLERANCE = 0.02


def _close(actual: float, expected: float) -> bool:
    return abs(float(actual) - float(expected)) <= TOLERANCE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--test-end", type=int, default=2025)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.start != 2001 or a.end not in EXPECTED or a.test_end != 2025:
        raise ValueError("U1 Delay production verification supports only 2001-2019 and 2001-2021 through 2025")

    data, identity = build_training_dataset()
    data = data.copy()
    data["completion_year"] = pd.to_numeric(data["completion_year"], errors="coerce")
    with tempfile.TemporaryDirectory(prefix=f"u1-delay-production-{a.end}-") as td:
        result = train_window_with_promoted_cost_and_delay(
            a.start, a.end, a.test_end, data=data, identity=identity, artifact_root=Path(td)
        )

    metrics = result["lifecycle"]["metrics"]
    promo = result["promotion"]
    contract = result["metadata"]["cost_evaluation_contract"]
    payload = {
        "window": f"{a.start}_{a.end}",
        "test_end": a.test_end,
        "cost_mae": metrics["cost"]["MAE"],
        "base_delay_mae": promo["previous_delay_mae"],
        "u1_delay_mae": metrics["delay"]["MAE"],
        "delay_improvement_percentage": promo["delay_improvement_percentage"],
        "comparison_test_projects": contract["test_projects"],
        "comparison_test_snapshots": contract["test_snapshots"],
        "production_cost_baseline": result["metadata"]["production_cost_baseline"],
        "production_delay_baseline": result["metadata"]["production_delay_baseline"],
        "promoted_delay_from_experiment": result["metadata"]["promoted_delay_from_experiment"],
        "cost_retained": promo["cost_retained"],
        "risk_retained": promo["risk_retained"],
    }

    expected = EXPECTED[a.end]
    if not _close(payload["cost_mae"], expected["cost"]):
        raise RuntimeError(f"Production Cost changed: {payload['cost_mae']} vs expected {expected['cost']}")
    if not _close(payload["base_delay_mae"], expected["base_delay"]):
        raise RuntimeError(f"Exp61 Delay comparator changed: {payload['base_delay_mae']} vs expected {expected['base_delay']}")
    if not _close(payload["u1_delay_mae"], expected["u1_delay"]):
        raise RuntimeError(f"U1 Delay did not reproduce: {payload['u1_delay_mae']} vs expected {expected['u1_delay']}")
    if payload["delay_improvement_percentage"] <= 0:
        raise RuntimeError("U1 Delay production promotion did not improve Delay")
    if payload["cost_retained"] is not True or payload["risk_retained"] is not True:
        raise RuntimeError("U1 Delay promotion failed target-isolation guard")
    if a.end == 2021 and (payload["comparison_test_projects"], payload["comparison_test_snapshots"]) != (721, 11200):
        raise RuntimeError("Verified 2001-2021 production cohort changed")

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    prefix = f"U1_DELAY_PRODUCTION_{a.start}_{a.end}"
    print(f"{prefix}_COST_MAE={payload['cost_mae']}")
    print(f"{prefix}_BASE_DELAY_MAE={payload['base_delay_mae']}")
    print(f"{prefix}_DELAY_MAE={payload['u1_delay_mae']}")
    print(f"{prefix}_DELAY_IMPROVEMENT_PERCENT={payload['delay_improvement_percentage']}")
    print(f"{prefix}_PROJECTS={payload['comparison_test_projects']}")
    print(f"{prefix}_SNAPSHOTS={payload['comparison_test_snapshots']}")


if __name__ == "__main__":
    main()
