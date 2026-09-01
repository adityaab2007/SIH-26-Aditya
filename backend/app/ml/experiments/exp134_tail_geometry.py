"""Exp134 — Nonlinear Saturation and Tail Geometry."""
from __future__ import annotations
import argparse
from backend.app.ml.experiments.internal_signal_challengers import ROOT, run_experiment, self_test
EXP_ID = "exp134"
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="models/monthly_lifecycle/experiments/exp134/ci"); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(EXP_ID); return
    run_experiment(EXP_ID, ROOT/a.output_dir)
if __name__ == "__main__": main()
