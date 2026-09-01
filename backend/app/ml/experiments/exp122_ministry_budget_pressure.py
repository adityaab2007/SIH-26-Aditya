"""Exp122 — Ministry capital-budget pressure (Cost + Delay).

Only versioned official Union Budget / Demands for Grants evidence is accepted.
Ministry matching is deterministic and audited. Rows without a trustworthy
mapping or an as-of published budget retain exact current-production predictions.
"""
from __future__ import annotations

import argparse
import json
import re
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
EXP_ID = "exp122"
EXTERNAL_ROOT = ROOT / "data" / "external" / "exp122_union_budget"
BUDGET_FILE = EXTERNAL_ROOT / "union_budget_ministry_year.csv"
MAPPING_FILE = EXTERNAL_ROOT / "ministry_mapping.csv"
MANIFEST = EXTERNAL_ROOT / "source_manifest.json"
FEATURES = [
    "exp122_capital_budget_cr",
    "exp122_capital_budget_yoy",
    "exp122_total_budget_yoy",
    "exp122_capital_share",
    "exp122_approved_to_capital_budget",
    "exp122_revised_to_capital_budget",
    "exp122_portfolio_to_capital_budget",
    "exp122_capital_budget_volatility_3y",
]


def normalize_ministry(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _fiscal_year_start(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    return pd.Series(np.where(d.dt.month.ge(4), d.dt.year, d.dt.year - 1), index=dates.index, dtype="Int64")


def load_verified_budget() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    expected = "Ministry of Finance, Government of India — Union Budget"
    if manifest.get("source_institution") != expected:
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "official Union Budget source manifest missing/invalid"}
    if not BUDGET_FILE.exists() or not MAPPING_FILE.exists():
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": "reviewed budget table and/or ministry mapping not present; no values are fabricated"}

    budget = pd.read_csv(BUDGET_FILE)
    mapping = pd.read_csv(MAPPING_FILE)
    budget_required = {"fiscal_year_start", "published_date", "official_budget_ministry", "capital_be_cr", "total_be_cr"}
    map_required = {"raw_ministry", "official_budget_ministry", "match_method", "match_confidence"}
    if not budget_required.issubset(budget.columns):
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": f"budget table missing columns {sorted(budget_required-set(budget.columns))}"}
    if not map_required.issubset(mapping.columns):
        return pd.DataFrame(), pd.DataFrame(), {"status": "INSUFFICIENT VERIFIED EXTERNAL DATA", "reason": f"mapping table missing columns {sorted(map_required-set(mapping.columns))}"}

    budget["fiscal_year_start"] = pd.to_numeric(budget["fiscal_year_start"], errors="coerce").astype("Int64")
    budget["published_date"] = pd.to_datetime(budget["published_date"], errors="coerce")
    budget["official_budget_ministry"] = budget["official_budget_ministry"].astype(str).str.strip()
    for c in ["capital_be_cr", "total_be_cr"]:
        budget[c] = pd.to_numeric(budget[c], errors="coerce")
    budget = budget.dropna(subset=["fiscal_year_start", "published_date", "official_budget_ministry"]).copy()
    if budget.duplicated(["fiscal_year_start", "official_budget_ministry"]).any():
        raise ValueError("Exp122 budget table has duplicate fiscal-year/ministry rows")
    budget = budget.sort_values(["official_budget_ministry", "fiscal_year_start"])
    g = budget.groupby("official_budget_ministry", sort=False)
    budget["capital_budget_yoy"] = g["capital_be_cr"].pct_change(fill_method=None) * 100.0
    budget["total_budget_yoy"] = g["total_be_cr"].pct_change(fill_method=None) * 100.0
    budget["capital_share"] = np.where(budget.total_be_cr.gt(0), budget.capital_be_cr / budget.total_be_cr, np.nan)
    budget["capital_budget_volatility_3y"] = g["capital_budget_yoy"].rolling(3, min_periods=2).std().reset_index(level=0, drop=True)

    mapping["raw_ministry"] = mapping["raw_ministry"].astype(str).str.strip()
    mapping["normalized_ministry"] = mapping["raw_ministry"].map(normalize_ministry)
    mapping["official_budget_ministry"] = mapping["official_budget_ministry"].astype(str).str.strip()
    mapping["match_method"] = mapping["match_method"].astype(str).str.strip()
    mapping["match_confidence"] = pd.to_numeric(mapping["match_confidence"], errors="coerce")
    if "effective_from" not in mapping: mapping["effective_from"] = "1900-01-01"
    if "effective_to" not in mapping: mapping["effective_to"] = "2100-12-31"
    mapping["effective_from"] = pd.to_datetime(mapping["effective_from"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    mapping["effective_to"] = pd.to_datetime(mapping["effective_to"], errors="coerce").fillna(pd.Timestamp("2100-12-31"))
    if mapping.duplicated(["normalized_ministry", "effective_from", "effective_to"]).any():
        raise ValueError("Exp122 ministry mapping contains duplicate effective-range keys")
    if mapping["match_method"].str.contains("fuzzy", case=False, na=False).any():
        raise ValueError("Exp122 refuses fuzzy ministry mappings")
    if (mapping["match_confidence"].fillna(0) < 0.95).any():
        raise ValueError("Exp122 requires reviewed ministry mapping confidence >= 0.95")

    info = {
        "status": "VERIFIED",
        "budget_rows": int(len(budget)),
        "mapping_rows": int(len(mapping)),
        "coverage_start_fy": int(budget.fiscal_year_start.min()) if len(budget) else None,
        "coverage_end_fy": int(budget.fiscal_year_start.max()) if len(budget) else None,
    }
    return budget, mapping, info


def _mapping_for_rows(base: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["normalized_ministry"] = out["ministry"].map(normalize_ministry)
    out["official_budget_ministry"] = pd.NA
    out["external_match_method"] = "unmatched"
    out["external_match_confidence"] = np.nan
    if mapping.empty:
        return out
    by_name = {k: v.copy() for k, v in mapping.groupby("normalized_ministry", sort=False)}
    for idx, row in out.iterrows():
        candidates = by_name.get(row["normalized_ministry"])
        if candidates is None:
            continue
        t = row["snapshot_date"]
        valid = candidates[(candidates.effective_from <= t) & (candidates.effective_to >= t)]
        if len(valid) != 1:
            continue
        hit = valid.iloc[0]
        out.at[idx, "official_budget_ministry"] = hit.official_budget_ministry
        out.at[idx, "external_match_method"] = hit.match_method
        out.at[idx, "external_match_confidence"] = float(hit.match_confidence)
    return out


def build_budget_features(rows: pd.DataFrame, budget: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    cols = ["canonical_project_id", "snapshot_date", "ministry", "approved_cost_cr", "revised_cost_cr"]
    base = rows.reindex(columns=cols).copy()
    base["snapshot_date"] = pd.to_datetime(base["snapshot_date"], errors="coerce")
    base["fiscal_year_start"] = _fiscal_year_start(base["snapshot_date"])
    base = _mapping_for_rows(base, mapping)
    if budget.empty:
        for f in FEATURES: base[f] = np.nan
        base["external_data_available"] = False
        base["external_feature_timestamp"] = pd.NaT
        return base

    right = budget.rename(columns={"published_date": "external_feature_timestamp"}).copy()
    joined = base.merge(right, on=["fiscal_year_start", "official_budget_ministry"], how="left", validate="many_to_one")
    as_of_ok = joined.external_feature_timestamp.notna() & joined.snapshot_date.notna() & joined.external_feature_timestamp.le(joined.snapshot_date)
    joined["exp122_capital_budget_cr"] = joined.capital_be_cr.where(as_of_ok)
    joined["exp122_capital_budget_yoy"] = joined.capital_budget_yoy.where(as_of_ok)
    joined["exp122_total_budget_yoy"] = joined.total_budget_yoy.where(as_of_ok)
    joined["exp122_capital_share"] = joined.capital_share.where(as_of_ok)
    joined["exp122_capital_budget_volatility_3y"] = joined.capital_budget_volatility_3y.where(as_of_ok)
    denom = pd.to_numeric(joined.exp122_capital_budget_cr, errors="coerce").replace(0, np.nan)
    joined["exp122_approved_to_capital_budget"] = pd.to_numeric(joined.approved_cost_cr, errors="coerce") / denom
    joined["exp122_revised_to_capital_budget"] = pd.to_numeric(joined.revised_cost_cr, errors="coerce") / denom
    portfolio = joined.groupby(["snapshot_date", "official_budget_ministry"], dropna=False)["approved_cost_cr"].transform(lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1))
    joined["exp122_portfolio_to_capital_budget"] = portfolio / denom
    joined["external_data_available"] = as_of_ok & joined.exp122_capital_budget_cr.notna()
    joined["external_feature_timestamp"] = joined.external_feature_timestamp.where(joined.external_data_available)
    if (joined.loc[joined.external_data_available, "external_feature_timestamp"] > joined.loc[joined.external_data_available, "snapshot_date"]).any():
        raise AssertionError("Future Union Budget publication entered an earlier snapshot")
    if len(joined) != len(base):
        raise AssertionError("Exp122 external join duplicated project/snapshot rows")
    return joined


def _coverage_by_year(frame: pd.DataFrame) -> dict:
    years = pd.to_datetime(frame.snapshot_date, errors="coerce").dt.year
    return {str(int(y)): {"rows": int((years==y).sum()), "available": int(((years==y) & frame.external_data_available).sum())} for y in sorted(years.dropna().unique())}


def _recommend(verdict: str) -> str:
    if verdict in {"STRONG PROMOTION CANDIDATE", "PROMISING"}: return "KEEP"
    if verdict == "INVALID / INSUFFICIENT DATA": return "NEEDS DATA"
    return "REJECT"


def run(output_dir: Path) -> dict:
    print_base_contract()
    before = production_hashes()
    budget, mapping, source = load_verified_budget()
    data, identity = build_training_dataset()
    data = data.copy(); data["snapshot_date"] = pd.to_datetime(data.snapshot_date, errors="coerce")
    feature_frame = build_budget_features(data, budget, mapping)
    cost_windows, delay_windows, combined_windows = [], [], []

    feature_cols = ["canonical_project_id", "snapshot_date", *FEATURES, "external_data_available", "external_feature_timestamp", "external_match_method", "external_match_confidence", "official_budget_ministry", "normalized_ministry"]
    for start, end, test_end in WINDOWS:
        with tempfile.TemporaryDirectory(prefix=f"exp122-{end}-") as td:
            td = Path(td)
            prod = fresh_production_window(data, identity, training_start=start, training_end=end, test_end=test_end, artifact_root=td/"baseline")
            score = prod.comparable.merge(feature_frame[feature_cols], on=["canonical_project_id", "snapshot_date"], how="left", validate="one_to_one")
            score["external_data_available"] = score.external_data_available.fillna(False)
            score["external_match_method"] = score.external_match_method.fillna("unmatched")
            assert_same_keys(prod.comparable, score)
            base_cost = score.predicted_cost_overrun.to_numpy(float)
            base_delay = score.predicted_delay_days.to_numpy(float)
            cost_candidate, delay_candidate = base_cost.copy(), base_delay.copy()
            cost_diag = {"status":"INSUFFICIENT VERIFIED EXTERNAL DATA","selected_scale":0.0}
            delay_diag = {"status":"INSUFFICIENT VERIFIED EXTERNAL DATA","selected_scale":0.0}
            cost_model = delay_model = None
            coverage_projects = int(score.loc[score.external_data_available, "canonical_project_id"].nunique())
            coverage_ratio = coverage_projects / max(int(score.canonical_project_id.nunique()), 1)

            if source.get("status") == "VERIFIED" and coverage_ratio >= 0.10:
                oof = rolling_production_oof(data, identity, training_start=start, training_end=end, root=td/"oof")
                oof = oof.merge(feature_frame[["canonical_project_id","snapshot_date",*FEATURES,"external_data_available"]], on=["canonical_project_id","snapshot_date"], how="left", validate="one_to_one")
                oof["external_data_available"] = oof.external_data_available.fillna(False)
                cost_candidate, cost_diag, cost_model = fit_bounded_residual_correction(oof, score, features=FEATURES, actual="actual_cost_overrun_percentage", production_col="predicted_cost_overrun", available_col="external_data_available", seed=12200+end)
                delay_candidate, delay_diag, delay_model = fit_bounded_residual_correction(oof, score, features=FEATURES, actual="actual_delay_days", production_col="predicted_delay_days", available_col="external_data_available", seed=12300+end)

            cost_base_mae = weighted_mae(score, "actual_cost_overrun_percentage", base_cost)
            cost_exp_mae = weighted_mae(score, "actual_cost_overrun_percentage", cost_candidate)
            delay_base_mae = weighted_mae(score, "actual_delay_days", base_delay)
            delay_exp_mae = weighted_mae(score, "actual_delay_days", delay_candidate)
            cost_gain = (cost_base_mae-cost_exp_mae)/cost_base_mae*100.0 if cost_base_mae else 0.0
            delay_gain = (delay_base_mae-delay_exp_mae)/delay_base_mae*100.0 if delay_base_mae else 0.0
            cost_status = "EXECUTION VALID" if cost_diag.get("status") == "EXECUTION VALID" else "INSUFFICIENT VERIFIED EXTERNAL DATA"
            delay_status = "EXECUTION VALID" if delay_diag.get("status") == "EXECUTION VALID" else "INSUFFICIENT VERIFIED EXTERNAL DATA"
            cost_boot = paired_project_bootstrap(score, actual="actual_cost_overrun_percentage", baseline=base_cost, challenger=cost_candidate, samples=5000, seed=12200+end)
            delay_boot = paired_project_bootstrap(score, actual="actual_delay_days", baseline=base_delay, challenger=delay_candidate, samples=5000, seed=12300+end)
            window = f"{start}_{end}"
            ledger = build_prediction_ledger(score, experiment_id=EXP_ID, window=window, production_cost_prediction=base_cost, experiment_cost_prediction=cost_candidate, production_delay_prediction=base_delay, experiment_delay_prediction=delay_candidate, extra_columns=["lifecycle_stage","external_data_available","external_match_method","external_match_confidence","external_feature_timestamp","official_budget_ministry"])
            ledger_dir = output_dir/window; write_prediction_ledger(ledger, ledger_dir, overwrite=True)
            if cost_model is not None: joblib.dump(cost_model, ledger_dir/"cost_residual_model.pkl")
            if delay_model is not None: joblib.dump(delay_model, ledger_dir/"delay_residual_model.pkl")

            cost_result = {"window":window,"status":cost_status,"base_mae":cost_base_mae,"experiment_mae":cost_exp_mae,"improvement_pct":cost_gain,"bootstrap":cost_boot,"lifecycle":lifecycle_metrics(score,actual="actual_cost_overrun_percentage",baseline=base_cost,challenger=cost_candidate),"residual_correction":cost_diag}
            delay_result = {"window":window,"status":delay_status,"base_mae":delay_base_mae,"experiment_mae":delay_exp_mae,"improvement_pct":delay_gain,"bootstrap":delay_boot,"lifecycle":lifecycle_metrics(score,actual="actual_delay_days",baseline=base_delay,challenger=delay_candidate),"residual_correction":delay_diag}
            cost_windows.append(cost_result); delay_windows.append(delay_result)
            unmatched = score.loc[~score.external_data_available, "normalized_ministry"].dropna().astype(str).value_counts().head(20).to_dict()
            combined_windows.append({"window":window,"projects":int(score.canonical_project_id.nunique()),"snapshots":int(len(score)),"external_project_coverage":coverage_ratio,"external_rows":int(score.external_data_available.sum()),"yearly_coverage":_coverage_by_year(score),"unmatched_or_uncovered_ministries":unmatched,"base_feature_count":25,"experimental_feature_count":len(FEATURES),"total_challenger_signal_count":25+len(FEATURES),"production_cost_baseline":prod.result["metadata"].get("production_cost_baseline"),"production_delay_baseline":prod.result["metadata"].get("production_delay_baseline")})

    if before != production_hashes():
        raise AssertionError("Exp122 modified tracked production artifacts")
    cost_verdict = verdict_from_windows(cost_windows); delay_verdict = verdict_from_windows(delay_windows)
    payload = {"experiment":EXP_ID,"name":"Ministry Capital-Budget Pressure","targets":["Cost","Delay"],"source":source,"cost_windows":cost_windows,"delay_windows":delay_windows,"coverage":combined_windows,"cost_verdict":cost_verdict,"delay_verdict":delay_verdict,"production_artifacts_unchanged":True,"final_recommendation":{"cost":_recommend(cost_verdict),"delay":_recommend(delay_verdict)}}
    save_json(ROOT/"reports"/"exp122_final_report.json", payload)
    lines = ["# Exp122 Final Report","",f"Cost verdict: **{cost_verdict}**",f"Delay verdict: **{delay_verdict}**","",f"External source status: **{source.get('status')}**","","| Window | Base Cost MAE | Exp122 Cost MAE | Cost improvement | Base Delay MAE | Exp122 Delay MAE | Delay improvement |","|---|---:|---:|---:|---:|---:|---:|"]
    for c,d in zip(cost_windows,delay_windows):
        lines.append(f"| {c['window']} | {c['base_mae']:.6f} | {c['experiment_mae']:.6f} | {c['improvement_pct']:.4f}% | {d['base_mae']:.6f} | {d['experiment_mae']:.6f} | {d['improvement_pct']:.4f}% |")
    lines += ["","Budget observations must be officially published on or before each snapshot. Ministry mapping is deterministic; fuzzy matching is rejected. Unmatched rows retain exact production predictions."]
    (ROOT/"reports"/"exp122_final_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(payload, indent=2, allow_nan=False)); return payload


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", default="models/monthly_lifecycle/experiments/exp122/ci"); args = p.parse_args(); run(ROOT/args.output_dir)


if __name__ == "__main__": main()
