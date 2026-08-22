#!/usr/bin/env python3
"""Run the isolated log cost-target experiment without changing production."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.ml.experiments.log_cost_target import run_experiment


if __name__ == "__main__":
    result = run_experiment()
    print(result["evolution"])
