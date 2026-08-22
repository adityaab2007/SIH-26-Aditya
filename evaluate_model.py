#!/usr/bin/env python3
"""Evaluate one registered real PAIMANA historical simulation model."""
from __future__ import annotations
import argparse
import json
from backend.app.ml.real_time_windows import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["2001_2015", "2015_2021"])
args = parser.parse_args()
result = evaluate(args.model)
print(json.dumps(result["metrics"], indent=2))
