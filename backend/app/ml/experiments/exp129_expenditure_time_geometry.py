"""Exp129 — Expenditure-Time Efficiency Geometry."""
from __future__ import annotations
import argparse, json, re
from backend.app.ml import production_exp35_baseline as exp35_production, production_exp61_baseline as exp61_production, production_u1_delay_baseline as u1_production
from backend.app.ml.experiments import internal_signal_challengers as harness, nextgen_common
from backend.app.ml.experiments.internal_signal_challengers import ROOT, run_experiment, self_test
EXP_ID="exp129"; _AFT_CAPACITY_ERROR=re.compile(r"^Only (\d+) projects have AFT evidence; cannot form the requested (\d+)-project calibration cohort\.$")
def _run(output_dir):
    original_oof=harness.rolling_production_oof; original35=exp35_production._select_aft_calibration_projects; original61=exp61_production._select_aft_calibration_projects; original_u1=u1_production._select_aft_calibration_projects; original_nextgen=nextgen_common._select_aft_calibration_projects; reductions=[]
    def adaptive_oof(*args,**kwargs):
        def selector(frame,limit=exp35_production.VERIFIED_AFT_CALIBRATION_PROJECTS):
            try: return original35(frame,limit=limit)
            except RuntimeError as exc:
                m=_AFT_CAPACITY_ERROR.match(str(exc))
                if not m: raise
                available,requested=map(int,m.groups())
                if available<20: raise RuntimeError(f"Only {available} projects have AFT evidence in this forward OOF fold; Exp129 requires at least 20 for an experiment-local adaptive calibration gate.") from exc
                reductions.append((requested,available)); return original35(frame,limit=available)
        exp35_production._select_aft_calibration_projects=selector; exp61_production._select_aft_calibration_projects=selector; u1_production._select_aft_calibration_projects=selector; nextgen_common._select_aft_calibration_projects=selector
        try: return original_oof(*args,**kwargs)
        finally: exp35_production._select_aft_calibration_projects=original35; exp61_production._select_aft_calibration_projects=original61; u1_production._select_aft_calibration_projects=original_u1; nextgen_common._select_aft_calibration_projects=original_nextgen
    harness.rolling_production_oof=adaptive_oof
    try: run_experiment(EXP_ID,output_dir)
    finally: harness.rolling_production_oof=original_oof; exp35_production._select_aft_calibration_projects=original35; exp61_production._select_aft_calibration_projects=original61; u1_production._select_aft_calibration_projects=original_u1; nextgen_common._select_aft_calibration_projects=original_nextgen
    if reductions: print("EXP129_OOF_AFT_GATE_ADAPTATIONS="+json.dumps([{"requested":r,"available":a} for r,a in reductions]))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="models/monthly_lifecycle/experiments/exp129/ci"); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(EXP_ID); return
    _run(ROOT/a.output_dir)
if __name__=="__main__": main()
