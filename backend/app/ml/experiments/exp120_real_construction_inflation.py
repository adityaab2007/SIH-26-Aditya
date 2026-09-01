"""Exp120 — real construction-inflation exposure (Cost only).

Only versioned OEA/DPIIT WPI observations are accepted. If the verified external
bundle is absent or lacks the future comparison period, the experiment returns
INSUFFICIENT VERIFIED EXTERNAL DATA and exact production Cost/Delay predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.experiments.prediction_ledger import build_prediction_ledger, write_prediction_ledger
from backend.app.ml.experiments.scientific_challenger_utils import (
    WINDOWS, assert_same_keys, fit_bounded_residual_correction, fresh_production_window,
    lifecycle_metrics, paired_project_bootstrap, print_base_contract, production_hashes,
    rolling_production_oof, save_json, verdict_from_windows, weighted_mae,
)

ROOT = Path(__file__).resolve().parents[4]
EXP_ID = "exp120"
EXTERNAL_ROOT = ROOT / "data" / "external" / "exp120_wpi"
WPI_FILE = EXTERNAL_ROOT / "wpi_monthly.csv"
MANIFEST = EXTERNAL_ROOT / "source_manifest.json"
SERIES = ["all_commodities", "manufactured_products", "fuel_power", "steel", "cement"]
FEATURES = [
    "exp120_all_yoy", "exp120_all_6m", "exp120_manufactured_yoy", "exp120_fuel_yoy",
    "exp120_steel_yoy", "exp120_cement_yoy", "exp120_all_volatility_12m",
    "exp120_all_acceleration", "exp120_cost_x_inflation",
]


def load_verified_wpi() -> tuple[pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if manifest.get("source_institution") != "Office of the Economic Adviser, DPIIT, Government of India":
        return pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "official OEA source manifest missing/invalid"}
    if not WPI_FILE.exists():
        return pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "cleaned versioned WPI bundle not present; no values are fabricated"}
    data = pd.read_csv(WPI_FILE)
    required = {"date", "all_commodities"}
    if not required.issubset(data.columns):
        return pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": f"WPI bundle missing required columns {sorted(required - set(data.columns))}"}
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    for c in SERIES:
        if c not in data: data[c] = np.nan
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if data.empty:
        return data, {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "WPI bundle has no valid monthly observations"}
    return data, {"status": "VERIFIED", "coverage_start": str(data.date.min().date()), "coverage_end": str(data.date.max().date()), "rows": int(len(data))}


def build_wpi_features(rows: pd.DataFrame, wpi: pd.DataFrame) -> pd.DataFrame:
    base = rows[["canonical_project_id", "snapshot_date", "approved_cost_cr"]].copy(); base["snapshot_date"] = pd.to_datetime(base["snapshot_date"], errors="coerce")
    if wpi.empty:
        for f in FEATURES: base[f] = np.nan
        base["external_data_available"] = False; base["external_feature_timestamp"] = pd.NaT; return base
    x = wpi.copy()
    for s in SERIES: x[f"{s}_yoy"] = x[s].pct_change(12, fill_method=None) * 100.0
    x["all_commodities_6m"] = x.all_commodities.pct_change(6, fill_method=None) * 100.0
    monthly = x.all_commodities.pct_change(fill_method=None) * 100.0
    x["all_volatility_12m"] = monthly.rolling(12, min_periods=6).std()
    x["all_acceleration"] = x["all_commodities_yoy"] - x["all_commodities_yoy"].shift(3)
    right = x.rename(columns={"date": "external_feature_timestamp"}).sort_values("external_feature_timestamp")
    joined = pd.merge_asof(base.sort_values("snapshot_date"), right, left_on="snapshot_date", right_on="external_feature_timestamp", direction="backward")
    joined["exp120_all_yoy"] = joined["all_commodities_yoy"]; joined["exp120_all_6m"] = joined["all_commodities_6m"]
    joined["exp120_manufactured_yoy"] = joined["manufactured_products_yoy"]; joined["exp120_fuel_yoy"] = joined["fuel_power_yoy"]
    joined["exp120_steel_yoy"] = joined["steel_yoy"]; joined["exp120_cement_yoy"] = joined["cement_yoy"]
    joined["exp120_all_volatility_12m"] = joined["all_volatility_12m"]; joined["exp120_all_acceleration"] = joined["all_acceleration"]
    joined["exp120_cost_x_inflation"] = pd.to_numeric(joined.approved_cost_cr, errors="coerce") * pd.to_numeric(joined.exp120_all_yoy, errors="coerce")
    joined["external_data_available"] = joined.external_feature_timestamp.notna() & joined.exp120_all_yoy.notna()
    if (joined.loc[joined.external_data_available, "external_feature_timestamp"] > joined.loc[joined.external_data_available, "snapshot_date"]).any(): raise AssertionError("Future WPI observation entered an earlier snapshot")
    return joined.sort_index()


def _coverage_by_year(frame: pd.DataFrame) -> dict:
    years = pd.to_datetime(frame.snapshot_date, errors="coerce").dt.year
    return {str(int(y)): {"rows": int((years==y).sum()), "available": int(((years==y) & frame.external_data_available).sum())} for y in sorted(years.dropna().unique())}


def run(output_dir: Path) -> dict:
    print_base_contract(); before = production_hashes(); wpi, source = load_verified_wpi()
    data, identity = build_training_dataset(); data=data.copy(); data["snapshot_date"]=pd.to_datetime(data.snapshot_date, errors="coerce"); features = build_wpi_features(data, wpi); windows=[]
    for start,end,test_end in WINDOWS:
        with tempfile.TemporaryDirectory(prefix=f"exp120-{end}-") as td:
            td=Path(td); prod=fresh_production_window(data,identity,training_start=start,training_end=end,test_end=test_end,artifact_root=td/"baseline")
            score = prod.comparable.merge(features[["canonical_project_id","snapshot_date",*FEATURES,"external_data_available","external_feature_timestamp"]], on=["canonical_project_id","snapshot_date"], how="left", validate="one_to_one"); score["external_data_available"] = score.external_data_available.fillna(False)
            assert_same_keys(prod.comparable, score); baseline=score.predicted_cost_overrun.to_numpy(float); delay_baseline=score.predicted_delay_days.to_numpy(float)
            coverage_projects=int(score.loc[score.external_data_available,"canonical_project_id"].nunique()); coverage_ratio=coverage_projects/max(int(score.canonical_project_id.nunique()),1)
            diag={"status":"INSUFFICIENT VERIFIED EXTERNAL DATA","selected_scale":0.0,"reason":"holdout WPI coverage below 10%"}; model=None; candidate=baseline.copy()
            if source.get("status")=="VERIFIED" and coverage_ratio>=0.10:
                oof=rolling_production_oof(data,identity,training_start=start,training_end=end,root=td/"oof")
                oof=oof.merge(features[["canonical_project_id","snapshot_date",*FEATURES,"external_data_available"]],on=["canonical_project_id","snapshot_date"],how="left",validate="one_to_one"); oof["external_data_available"]=oof.external_data_available.fillna(False)
                candidate,diag,model=fit_bounded_residual_correction(oof,score,features=FEATURES,actual="actual_cost_overrun_percentage",production_col="predicted_cost_overrun",available_col="external_data_available",seed=12000+end)
            base_mae=weighted_mae(score,"actual_cost_overrun_percentage",baseline); exp_mae=weighted_mae(score,"actual_cost_overrun_percentage",candidate); gain=(base_mae-exp_mae)/base_mae*100 if base_mae else 0.0
            bootstrap=paired_project_bootstrap(score,actual="actual_cost_overrun_percentage",baseline=baseline,challenger=candidate,samples=5000,seed=12000+end); stages=lifecycle_metrics(score,actual="actual_cost_overrun_percentage",baseline=baseline,challenger=candidate); status="EXECUTION VALID" if diag.get("status")=="EXECUTION VALID" else "INSUFFICIENT VERIFIED EXTERNAL DATA"
            window=f"{start}_{end}"; ledger=build_prediction_ledger(score,experiment_id=EXP_ID,window=window,production_cost_prediction=baseline,experiment_cost_prediction=candidate,extra_columns=["lifecycle_stage","external_data_available","external_feature_timestamp"]); ledger_dir=output_dir/window; write_prediction_ledger(ledger,ledger_dir,overwrite=True)
            if model is not None: joblib.dump(model,ledger_dir/"cost_residual_model.pkl")
            if not np.array_equal(delay_baseline, score.predicted_delay_days.to_numpy(float)): raise AssertionError("Exp120 changed Delay")
            windows.append({"window":window,"status":status,"base_mae":base_mae,"experiment_mae":exp_mae,"improvement_pct":gain,"projects":int(score.canonical_project_id.nunique()),"snapshots":int(len(score)),"external_project_coverage":coverage_ratio,"external_rows":int(score.external_data_available.sum()),"yearly_coverage":_coverage_by_year(score),"base_feature_count":25,"experimental_feature_count":len(FEATURES),"total_challenger_signal_count":25+len(FEATURES),"delay_prediction_equality_asserted":True,"residual_correction":diag,"bootstrap":bootstrap,"lifecycle":stages,"production_cost_baseline":prod.result["metadata"].get("production_cost_baseline"),"production_delay_baseline":prod.result["metadata"].get("production_delay_baseline")})
    if before!=production_hashes(): raise AssertionError("Exp120 modified tracked production artifacts")
    verdict=verdict_from_windows(windows); payload={"experiment":EXP_ID,"name":"Real Construction-Inflation Exposure Model","target":"Cost","source":source,"windows":windows,"verdict":verdict,"production_artifacts_unchanged":True,"final_recommendation":"KEEP" if verdict in {"STRONG PROMOTION CANDIDATE","PROMISING"} else ("NEEDS DATA" if verdict=="INVALID / INSUFFICIENT DATA" else "REJECT")}
    save_json(ROOT/"reports"/"exp120_final_report.json",payload); lines=["# Exp120 Final Report","",f"Verdict: **{verdict}**","",f"External source status: **{source.get('status')}**","","| Window | Base Cost MAE | Exp120 Cost MAE | Improvement | Status |","|---|---:|---:|---:|---|"]
    for w in windows: lines.append(f"| {w['window']} | {w['base_mae']:.6f} | {w['experiment_mae']:.6f} | {w['improvement_pct']:.4f}% | {w['status']} |")
    lines += ["","No WPI observation after a snapshot is allowed into that snapshot. Rows without verified WPI retain exact production prediction. Delay is unchanged."]; (ROOT/"reports"/"exp120_final_report.md").write_text("\n".join(lines)+"\n"); print(json.dumps(payload,indent=2,allow_nan=False)); return payload


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="models/monthly_lifecycle/experiments/exp120/ci"); a=p.parse_args(); run(ROOT/a.output_dir)

if __name__=="__main__": main()
