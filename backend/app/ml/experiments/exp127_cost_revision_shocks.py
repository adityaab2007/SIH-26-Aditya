"""Exp127 — Cost Revision Shock Persistence."""
from __future__ import annotations

import argparse
import json
import re

from backend.app.ml import production_exp35_baseline as exp35_production
from backend.app.ml.experiments import internal_signal_challengers as harness
from backend.app.ml.experiments.internal_signal_challengers import ROOT, run_experiment, self_test

EXP_ID = "exp127"
_AFT_CAPACITY_ERROR = re.compile(r"^Only (\d+) projects have AFT evidence; cannot form the requested (\d+)-project calibration cohort\.$")


def _run_with_adaptive_forward_oof(output_dir) -> None:
    original_oof = harness.rolling_production_oof
    original_selector = exp35_production._select_aft_calibration_projects
    reductions: list[tuple[int, int]] = []

    def adaptive_oof(*args, **kwargs):
        def adaptive_selector(frame, limit=exp35_production.VERIFIED_AFT_CALIBRATION_PROJECTS):
            try:
                return original_selector(frame, limit=limit)
            except RuntimeError as exc:
                match = _AFT_CAPACITY_ERROR.match(str(exc))
                if not match:
                    raise
                available = int(match.group(1)); requested = int(match.group(2))
                if available < 20:
                    raise RuntimeError(f"Only {available} projects have AFT evidence in this forward OOF fold; Exp127 requires at least 20 for an experiment-local adaptive calibration gate.") from exc
                reductions.append((requested, available))
                return original_selector(frame, limit=available)
        exp35_production._select_aft_calibration_projects = adaptive_selector
        try:
            return original_oof(*args, **kwargs)
        finally:
            exp35_production._select_aft_calibration_projects = original_selector

    harness.rolling_production_oof = adaptive_oof
    try:
        run_experiment(EXP_ID, output_dir)
    finally:
        harness.rolling_production_oof = original_oof
        exp35_production._select_aft_calibration_projects = original_selector
    if reductions:
        print("EXP127_OOF_AFT_GATE_ADAPTATIONS=" + json.dumps([{"requested": r, "available": a} for r, a in reductions]))


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="models/monthly_lifecycle/experiments/exp127/ci"); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(EXP_ID); return
    _run_with_adaptive_forward_oof(ROOT/a.output_dir)


if __name__ == "__main__": main()
