#!/usr/bin/env python3
"""Train one real PAIMANA historical simulation window."""
from __future__ import annotations
import argparse
from backend.app.ml.real_time_windows import WINDOWS, train

parser = argparse.ArgumentParser()
parser.add_argument("--start-year", type=int, required=True)
parser.add_argument("--end-year", type=int, required=True)
args = parser.parse_args()
key = next((key for key, window in WINDOWS.items() if window.training_start == args.start_year and window.training_end == args.end_year), None)
if key is None:
    parser.error("Supported windows are 2001-2015 and 2015-2021.")
print(train(key))
