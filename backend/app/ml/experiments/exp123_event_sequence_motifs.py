"""Exp123 — causal execution event-sequence motifs for Cost and Delay."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from backend.app.ml import production_exp35_baseline as exp35_production
from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.experiments.prediction_ledger import build_prediction_ledger, write_prediction_ledger
from backend.app.ml.experiments.scientific_challenger_utils import (
    WINDOWS, attach_features, assert_same_keys, fit_bounded_residual_correction,
    fresh_production_window, lifecycle_metrics, paired_project_bootstrap,
    print_base_contract, production_hashes, rolling_production_oof, save_json,
    verdict_from_windows, weighted_mae,
)

ROOT = Path(__file__).resolve().parents[4]
EXP_ID = "exp123"
DETERIORATION = {"C+", "S+", "P0", "E-", "R0"}
RECOVERY = {"C-", "S-", "P+", "E+"}
RAW = ["revised_cost_cr", "cost_escalation_percentage", "schedule_slippage_days", "physical_progress", "expenditure_ratio"]
BASE_SEQ_FEATURES = ["exp123_deterioration_streak", "exp123_recovery_streak", "exp123_event_entropy", "exp123_transition_entropy", "exp123_event_count_6", "exp123_months_since_deterioration"] + [f"exp123_latest_{token.replace('+','up').replace('-','down')}" for token in ["C+","C-","S+","S-","P0","P+","E-","E+","R0"]]
_AFT_CAPACITY_ERROR = re.compile(r"^Only (\d+) projects have AFT evidence; cannot form the requested (\d+)-project calibration cohort\.$")


def _q(values: pd.Series, q: float, floor: float) -> float:
    x = pd.to_numeric(values, errors="coerce").abs(); x = x[np.isfinite(x) & x.gt(0)]
    return max(float(x.quantile(q)) if len(x) else 0.0, floor)


def learn_event_thresholds(frame: pd.DataFrame, training_end: int) -> dict:
    train = frame[pd.to_numeric(frame["completion_year"], errors="coerce").le(training_end)].copy(); train["snapshot_date"] = pd.to_datetime(train["snapshot_date"], errors="coerce"); train = train.sort_values(["canonical_project_id", "snapshot_date"], kind="mergesort")
    g = train.groupby("canonical_project_id", sort=False)
    dc = g["cost_escalation_percentage"].diff(); ds = g["schedule_slippage_days"].diff(); dp = g["physical_progress"].diff() if "physical_progress" in train else pd.Series(np.nan, index=train.index); de = g["expenditure_ratio"].diff()
    return {"cost_change": _q(dc,0.75,0.5), "schedule_change": _q(ds,0.75,7.0), "progress_change": _q(dp,0.50,0.5), "expenditure_change": _q(de,0.50,0.01), "stagnation_abs_progress":0.10, "report_gap_days":62.0, "learned_only_from_completion_year_lte":int(training_end)}


def _entropy(tokens: list[str]) -> float:
    tokens=[x for x in tokens if x and x!="NONE"]
    if not tokens: return 0.0
    counts=Counter(tokens); total=float(sum(counts.values())); return float(-sum((n/total)*math.log(n/total+1e-12) for n in counts.values()))


def build_sequence_features(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    data=frame.copy(); data["snapshot_date"]=pd.to_datetime(data["snapshot_date"], errors="coerce")
    for c in RAW:
        if c not in data: data[c]=np.nan
        data[c]=pd.to_numeric(data[c], errors="coerce")
    data=data.sort_values(["canonical_project_id","snapshot_date"], kind="mergesort").copy(); rows=[]
    for project_id,g in data.groupby("canonical_project_id", sort=False):
        history=[]; prior=None; last_bad_date=None; bad_streak=0; good_streak=0
        for _,row in g.sort_values("snapshot_date",kind="mergesort").iterrows():
            event="NONE"
            if prior is not None:
                gap=(row.snapshot_date-prior.snapshot_date).days if pd.notna(row.snapshot_date) and pd.notna(prior.snapshot_date) else 0
                dc=row.cost_escalation_percentage-prior.cost_escalation_percentage; ds=row.schedule_slippage_days-prior.schedule_slippage_days; dp=row.physical_progress-prior.physical_progress; de=row.expenditure_ratio-prior.expenditure_ratio
                if gap>thresholds["report_gap_days"]: event="R0"
                elif pd.notna(dc) and dc>=thresholds["cost_change"]: event="C+"
                elif pd.notna(ds) and ds>=thresholds["schedule_change"]: event="S+"
                elif pd.notna(dp) and abs(dp)<=thresholds["stagnation_abs_progress"]: event="P0"
                elif pd.notna(de) and de<=-thresholds["expenditure_change"]: event="E-"
                elif pd.notna(dc) and dc<=-thresholds["cost_change"]: event="C-"
                elif pd.notna(ds) and ds<=-thresholds["schedule_change"]: event="S-"
                elif pd.notna(dp) and dp>=thresholds["progress_change"]: event="P+"
                elif pd.notna(de) and de>=thresholds["expenditure_change"]: event="E+"
            if event!="NONE": history.append(event)
            if event in DETERIORATION: bad_streak+=1; good_streak=0; last_bad_date=row.snapshot_date
            elif event in RECOVERY: good_streak+=1; bad_streak=0
            recent=history[-8:]; transitions=[f"{a}>{b}" for a,b in zip(recent[:-1],recent[1:])]; latest2=">".join(history[-2:]) if len(history)>=2 else ""; latest3=">".join(history[-3:]) if len(history)>=3 else ""; months_since=((row.snapshot_date-last_bad_date).days/30.4375) if last_bad_date is not None and pd.notna(row.snapshot_date) else 99.0
            item={"canonical_project_id":project_id,"snapshot_date":row.snapshot_date,"exp123_primary_event":event,"exp123_latest_motif2":latest2,"exp123_latest_motif3":latest3,"exp123_sequence_tail":">".join(history[-12:]),"exp123_deterioration_streak":float(bad_streak),"exp123_recovery_streak":float(good_streak),"exp123_event_entropy":_entropy(recent),"exp123_transition_entropy":_entropy(transitions),"exp123_event_count_6":float(len(history[-6:])),"exp123_months_since_deterioration":float(months_since),"exp123_available":True}
            for token in ["C+","C-","S+","S-","P0","P+","E-","E+","R0"]: item[f"exp123_latest_{token.replace('+','up').replace('-','down')}"]=float(event==token)
            rows.append(item); prior=row
    result=pd.DataFrame(rows)
    if result.duplicated(["canonical_project_id","snapshot_date"]).any(): raise AssertionError("Exp123 feature generation duplicated keys")
    return result


def select_motifs(oof: pd.DataFrame, actual: str, production_col: str, max_motifs: int=6) -> list[str]:
    work=oof.copy(); work["__residual"]=pd.to_numeric(oof[actual],errors="coerce")-pd.to_numeric(oof[production_col],errors="coerce")
    candidates=pd.concat([work["exp123_latest_motif2"],work["exp123_latest_motif3"]]).dropna().astype(str); candidates=[m for m in candidates.unique().tolist() if m and ">" in m]
    min_support=max(20,int(math.ceil(work.canonical_project_id.nunique()*0.02))); overall=float(work["__residual"].mean()); ranked=[]
    for motif in candidates:
        mask=work["exp123_latest_motif2"].eq(motif)|work["exp123_latest_motif3"].eq(motif); projects=int(work.loc[mask,"canonical_project_id"].nunique())
        if projects<min_support: continue
        effect=abs(float(work.loc[mask,"__residual"].mean())-overall); ranked.append((effect*math.sqrt(projects),projects,motif))
    ranked.sort(reverse=True); return [m for _,_,m in ranked[:max_motifs]]


def add_motif_features(frame: pd.DataFrame, motifs: list[str], prefix: str):
    out=frame.copy(); features=list(BASE_SEQ_FEATURES); tail=out["exp123_sequence_tail"].fillna("").astype(str)
    for i,motif in enumerate(motifs):
        col=f"{prefix}_motif_{i}"; count=f"{prefix}_motif_count_{i}"; out[col]=(out["exp123_latest_motif2"].eq(motif)|out["exp123_latest_motif3"].eq(motif)).astype(float); out[count]=tail.map(lambda s,m=motif:float(s.count(m))); features += [col,count]
    return out,features


def _rolling_oof_with_adaptive_aft_gate(
    data: pd.DataFrame,
    identity: pd.DataFrame,
    *,
    training_start: int,
    training_end: int,
    root: Path,
) -> pd.DataFrame:
    """Run strict forward OOF while adapting only the audit gate to fold capacity.

    Exp35's fixed 688-project AFT gate is an audit contract for the verified
    production holdout. Early one-year OOF folds can contain fewer than 688
    AFT-evidence projects. For those folds only, use every available AFT-evidence
    project (minimum 20) so the same production architecture can be evaluated
    without consulting future folds or outcomes. The selector is restored
    immediately, so the final production baseline remains untouched.
    """
    original_selector = exp35_production._select_aft_calibration_projects
    reductions: list[tuple[int, int]] = []

    def adaptive_selector(frame: pd.DataFrame, limit: int = exp35_production.VERIFIED_AFT_CALIBRATION_PROJECTS):
        try:
            return original_selector(frame, limit=limit)
        except RuntimeError as exc:
            match = _AFT_CAPACITY_ERROR.match(str(exc))
            if not match:
                raise
            available = int(match.group(1))
            requested = int(match.group(2))
            if available < 20:
                raise RuntimeError(
                    f"Only {available} projects have AFT evidence in this forward OOF fold; "
                    "Exp123 requires at least 20 for an experiment-local adaptive calibration gate."
                ) from exc
            reductions.append((requested, available))
            return original_selector(frame, limit=available)

    exp35_production._select_aft_calibration_projects = adaptive_selector
    try:
        oof = rolling_production_oof(
            data,
            identity,
            training_start=training_start,
            training_end=training_end,
            root=root,
        )
    finally:
        exp35_production._select_aft_calibration_projects = original_selector

    if reductions:
        print(
            "EXP123_OOF_AFT_GATE_ADAPTATIONS="
            + json.dumps([{"requested": req, "available": avail} for req, avail in reductions])
        )
    return oof


def _target_result(score,baseline,candidate,actual,end):
    b=weighted_mae(score,actual,baseline); c=weighted_mae(score,actual,candidate)
    return {"base_mae":b,"experiment_mae":c,"improvement_pct":(b-c)/b*100.0 if b else 0.0,"bootstrap":paired_project_bootstrap(score,actual=actual,baseline=baseline,challenger=candidate,samples=5000,seed=12300+end+(1 if actual.endswith('days') else 0)),"lifecycle":lifecycle_metrics(score,actual=actual,baseline=baseline,challenger=candidate)}


def run(output_dir: Path, training_end: int | None = None) -> dict:
    print_base_contract(); before=production_hashes(); data,identity=build_training_dataset(); data=data.copy(); data["snapshot_date"]=pd.to_datetime(data["snapshot_date"],errors="coerce"); windows=[]
    selected_windows=[w for w in WINDOWS if training_end is None or w[1] == training_end]
    if not selected_windows:
        raise ValueError(f"Unsupported Exp123 training_end={training_end}; expected one of {[w[1] for w in WINDOWS]}")
    for start,end,test_end in selected_windows:
        thresholds=learn_event_thresholds(data,end); seq=build_sequence_features(data,thresholds)
        with tempfile.TemporaryDirectory(prefix=f"exp123-{end}-") as td:
            td=Path(td); prod=fresh_production_window(data,identity,training_start=start,training_end=end,test_end=test_end,artifact_root=td/"baseline"); oof=_rolling_oof_with_adaptive_aft_gate(data,identity,training_start=start,training_end=end,root=td/"oof")
            cols=BASE_SEQ_FEATURES+["exp123_latest_motif2","exp123_latest_motif3","exp123_sequence_tail","exp123_available"]
            oof=attach_features(oof,seq,cols); score=attach_features(prod.comparable,seq,cols); assert_same_keys(prod.comparable,score)
            cost_motifs=select_motifs(oof,"actual_cost_overrun_percentage","predicted_cost_overrun"); delay_motifs=select_motifs(oof,"actual_delay_days","predicted_delay_days")
            cost_oof,cost_features=add_motif_features(oof,cost_motifs,"exp123_cost"); cost_score,_=add_motif_features(score,cost_motifs,"exp123_cost"); delay_oof,delay_features=add_motif_features(oof,delay_motifs,"exp123_delay"); delay_score,_=add_motif_features(score,delay_motifs,"exp123_delay")
            cost_candidate,cost_diag,cost_model=fit_bounded_residual_correction(cost_oof,cost_score,features=cost_features,actual="actual_cost_overrun_percentage",production_col="predicted_cost_overrun",available_col="exp123_available",seed=12301+end)
            delay_candidate,delay_diag,delay_model=fit_bounded_residual_correction(delay_oof,delay_score,features=delay_features,actual="actual_delay_days",production_col="predicted_delay_days",available_col="exp123_available",seed=12302+end)
            cost_base=score["predicted_cost_overrun"].to_numpy(float); delay_base=score["predicted_delay_days"].to_numpy(float); cost_result=_target_result(score,cost_base,cost_candidate,"actual_cost_overrun_percentage",end); delay_result=_target_result(score,delay_base,delay_candidate,"actual_delay_days",end)
            window_name=f"{start}_{end}"; ledger=build_prediction_ledger(score,experiment_id=EXP_ID,window=window_name,production_cost_prediction=cost_base,experiment_cost_prediction=cost_candidate,production_delay_prediction=delay_base,experiment_delay_prediction=delay_candidate,extra_columns=["lifecycle_stage"]); ledger_dir=output_dir/window_name; write_prediction_ledger(ledger,ledger_dir,overwrite=True)
            if cost_model is not None: joblib.dump(cost_model,ledger_dir/"cost_residual_model.pkl")
            if delay_model is not None: joblib.dump(delay_model,ledger_dir/"delay_residual_model.pkl")
            windows.append({"window":window_name,"status":"EXECUTION VALID","projects":int(score.canonical_project_id.nunique()),"snapshots":int(len(score)),"base_feature_count":25,"thresholds":thresholds,"cost_motifs":cost_motifs,"delay_motifs":delay_motifs,"cost_feature_count":len(cost_features),"delay_feature_count":len(delay_features),"cost":cost_result|{"residual_correction":cost_diag},"delay":delay_result|{"residual_correction":delay_diag},"production_cost_baseline":prod.result["metadata"].get("production_cost_baseline"),"production_delay_baseline":prod.result["metadata"].get("production_delay_baseline")})
    if before!=production_hashes(): raise AssertionError("Exp123 modified tracked production artifacts")
    cost_windows=[{"status":w["status"],"improvement_pct":w["cost"]["improvement_pct"],"bootstrap":w["cost"]["bootstrap"]} for w in windows]; delay_windows=[{"status":w["status"],"improvement_pct":w["delay"]["improvement_pct"],"bootstrap":w["delay"]["bootstrap"]} for w in windows]; cv=verdict_from_windows(cost_windows); dv=verdict_from_windows(delay_windows)
    payload={"experiment":EXP_ID,"name":"Execution Event-Sequence Motifs","base_pipeline":"25-FEATURE MONTHLY LIFECYCLE","windows":windows,"cost_verdict":cv,"delay_verdict":dv,"production_artifacts_unchanged":True,"final_recommendation":{"cost":"KEEP" if cv in {"STRONG PROMOTION CANDIDATE","PROMISING"} else "REJECT","delay":"KEEP" if dv in {"STRONG PROMOTION CANDIDATE","PROMISING"} else "REJECT"}}
    save_json(ROOT/"reports"/"exp123_final_report.json",payload); lines=["# Exp123 Final Report","",f"Cost verdict: **{cv}**  ",f"Delay verdict: **{dv}**","","## Results","","| Window | Base Cost | Exp Cost | Cost Δ% | Base Delay | Exp Delay | Delay Δ% |","|---|---:|---:|---:|---:|---:|---:|"]
    for w in windows: lines.append(f"| {w['window']} | {w['cost']['base_mae']:.6f} | {w['cost']['experiment_mae']:.6f} | {w['cost']['improvement_pct']:.4f}% | {w['delay']['base_mae']:.6f} | {w['delay']['experiment_mae']:.6f} | {w['delay']['improvement_pct']:.4f}% |")
    lines += ["","Motif thresholds and ranking are learned from training/OOF evidence only. Appending future reports cannot rewrite earlier features. Production artifacts remain hash-identical."]; (ROOT/"reports"/"exp123_final_report.md").write_text("\n".join(lines)+"\n"); print(json.dumps(payload,indent=2,allow_nan=False)); return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="models/monthly_lifecycle/experiments/exp123/ci")
    p.add_argument("--training-end",type=int,choices=[2019,2021],default=None)
    a=p.parse_args()
    run(ROOT/a.output_dir, training_end=a.training_end)

if __name__=="__main__": main()
