#!/usr/bin/env python3
"""Reproducible end-to-end official PAIMANA monthly lifecycle pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.ml.monthly_lifecycle import build_training_dataset
from backend.app.ml.monthly_training import train_required_windows, train_window
from backend.app.services.paimana_ingestion_service import (
    AUDIT_PATH, MANIFEST_PATH, archive_coverage, build_monthly_history,
    discover_archive_reports, download_archive_reports,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "monthly_lifecycle_upgrade_report.md"


def write_report(comparison: dict | None = None) -> None:
    audit = json.loads(AUDIT_PATH.read_text()) if AUDIT_PATH.exists() else {}
    outcome_audit_path = ROOT / "data" / "processed" / "paimana_completed_outcomes_audit.json"
    outcome_audit = json.loads(outcome_audit_path.read_text()) if outcome_audit_path.exists() else {}
    identity_path = ROOT / "data" / "processed" / "paimana_identity_audit.csv"
    trajectory_path = ROOT / "data" / "processed" / "paimana_project_trajectories.csv"
    import pandas as pd
    identity = pd.read_csv(identity_path) if identity_path.exists() else pd.DataFrame()
    trajectories = pd.read_csv(trajectory_path, usecols=["canonical_project_id"], low_memory=False) if trajectory_path.exists() else pd.DataFrame()
    missing_months = ", ".join(f"{item['month']} {item['financial_year']}" for item in audit.get("missing_months", [])) or "none"
    downloaded = audit.get("reports_downloaded_or_cached", 0); parsed = audit.get("reports_parsed", 0)
    parser_success = parsed / downloaded * 100 if downloaded else 0
    identity_counts = identity.identity_method.value_counts().to_dict() if not identity.empty else {}
    lines = ["# PAIMANA monthly lifecycle forecasting upgrade", "", "## Data", "",
             f"- Official reports discovered: {audit.get('reports_discovered', 0)}",
             f"- Financial years: {audit.get('financial_year_count', 0)} ({', '.join(audit.get('financial_years', []))})",
             f"- Known missing months: {missing_months}",
             f"- Reports parsed with project rows: {audit.get('reports_parsed', 0)}",
             f"- Reports downloaded/cached: {audit.get('reports_downloaded_or_cached', 0)}; download failures: {audit.get('reports_discovered', 0) - audit.get('reports_downloaded_or_cached', 0)}",
             f"- Downloaded reports with no recognized project rows: {audit.get('reports_without_rows', 0)}",
             f"- Monthly observations: {audit.get('monthly_observations', 0)}",
             f"- Unique reported project codes: {audit.get('unique_reported_project_codes', 0)}",
             f"- Canonical trajectories generated: {trajectories.canonical_project_id.nunique() if not trajectories.empty else 0}",
             f"- Parser row-success rate among downloaded reports: {parser_success:.1f}% (summary-only/unrecognized reports remain audited)",
             f"- Parser coverage: {json.dumps(audit.get('parser_report_counts', {}), sort_keys=True)}",
             f"- Observations by financial year: {json.dumps(audit.get('observations_by_financial_year', {}), sort_keys=True)}",
             "", "## Identity", "",
             f"- Snapshot rows audited: {len(identity)}",
             f"- Identity-verified rows: {int(identity.identity_verified.sum()) if not identity.empty else 0}",
             f"- Ambiguous exact-name rows excluded: {int(identity.identity_method.eq('ambiguous_exact_name').sum()) if not identity.empty else 0}",
             f"- Identity methods: {json.dumps(identity_counts, sort_keys=True)}",
             f"- Canonical completed outcomes: {outcome_audit.get('canonical_outcomes', 0)} ({outcome_audit.get('canonical_outcomes_with_official_id', 0)} with unique official IDs)",
             "", "No fuzzy name matching is used. Unverified or ambiguous rows are excluded from supervised training.", ""]
    if comparison:
        for item in comparison.get("windows", []):
            baseline = item["baseline"]["metrics"]; lifecycle = item["lifecycle"]["metrics"]
            improved = lifecycle["cost"]["MAE"] < baseline["cost"]["MAE"] and lifecycle["delay"]["MAE"] < baseline["delay"]["MAE"]
            metadata = item["metadata"]
            audit_rows = metadata["feature_availability"]["features"]
            retained = [row["feature"] for row in audit_rows if row["decision"] == "keep"]
            rejected = [f"{row['feature']} ({row['reason']})" for row in audit_rows if row["decision"] == "remove"]
            feature_table = ["| Feature | Available | Missing | Years | Projects | As-of safe | Decision | Reason |", "|---|---:|---:|---:|---:|---|---|---|"]
            feature_table.extend(
                f"| {row['feature']} | {row['availability_percentage']}% | {row['missing_percentage']}% | {row['temporal_year_coverage']} | {row['project_coverage']} | {'yes' if row['safely_as_of_available'] else 'no'} | {row['decision']} | {row['reason']} |"
                for row in audit_rows
            )
            cost_gain = (baseline["cost"]["MAE"] - lifecycle["cost"]["MAE"]) / baseline["cost"]["MAE"] * 100
            delay_gain = (baseline["delay"]["MAE"] - lifecycle["delay"]["MAE"]) / baseline["delay"]["MAE"] * 100
            lines.extend([f"## Window {item['window']}", "", "### Baseline versus lifecycle", "",
                          f"Training: {metadata['training_snapshots']} snapshots / {metadata['unique_training_projects']} projects. Test: {metadata['test_snapshots']} snapshots / {metadata['unique_test_projects']} projects.", "",
                          f"Selected regressors: cost={metadata['selected_algorithms']['cost']}; delay={metadata['selected_algorithms']['delay']}. Risk uses the documented Random Forest classifier.", "",
                          "| Metric | Five-feature baseline | Monthly lifecycle |", "|---|---:|---:|",
                          f"| Cost MAE | {baseline['cost']['MAE']} | {lifecycle['cost']['MAE']} |",
                          f"| Cost RMSE | {baseline['cost']['RMSE']} | {lifecycle['cost']['RMSE']} |",
                          f"| Cost R2 | {baseline['cost']['R2']} | {lifecycle['cost']['R2']} |",
                          f"| Delay MAE | {baseline['delay']['MAE']} | {lifecycle['delay']['MAE']} |",
                          f"| Delay RMSE | {baseline['delay']['RMSE']} | {lifecycle['delay']['RMSE']} |",
                          f"| Delay R2 | {baseline['delay']['R2']} | {lifecycle['delay']['R2']} |",
                          f"| Risk accuracy | {baseline['risk']['accuracy']} | {lifecycle['risk']['accuracy']} |",
                          f"| Risk macro F1 | {baseline['risk']['macro_f1']} | {lifecycle['risk']['macro_f1']} |", "",
                          f"Primary MAE improvement: cost {cost_gain:.1f}%; delay {delay_gain:.1f}%.", "",
                          "### Feature audit", "",
                          f"Retained ({len(retained)}): {', '.join(retained)}.", "",
                          f"Rejected ({len(rejected)}): {'; '.join(rejected)}.", "",
                          *feature_table, "",
                          "### Lifecycle-stage evaluation", "", "```json", json.dumps(item["lifecycle"]["lifecycle_stages"], indent=2), "```", "",
                          "### Ablations", "", "```json", json.dumps(item["ablations"], indent=2), "```", "",
                          "### SHAP / importance", "", "```json", json.dumps(item["shap"], indent=2), "```", "",
                          f"Conclusion for this window: {'both cost and delay MAE improved' if improved else 'the lifecycle model did not improve both primary MAE targets; no material-improvement claim is made'}.",
                          "Each ablation repeats the same training-only candidate selection protocol; the final holdout remains untouched. Trajectory and agency-prior effects are interpreted metric by metric and do not inherit the full model's overall improvement claim.", ""])
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2001); parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--local-only", action="store_true"); parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-parse", action="store_true"); parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--skip-ingestion", action="store_true", help="Reuse the existing canonical snapshot CSV")
    parser.add_argument("--train", action="store_true"); parser.add_argument("--training-start", type=int)
    parser.add_argument("--training-end", type=int); parser.add_argument("--test-end", type=int)
    args = parser.parse_args()
    reports = discover_archive_reports()
    if args.discover_only:
        print(json.dumps(archive_coverage(reports), indent=2)); return
    manifest = None if args.local_only else download_archive_reports(from_year=args.start_year, to_year=args.end_year, force=args.force_download)
    if args.skip_ingestion:
        import pandas as pd
        from backend.app.services.paimana_ingestion_service import OUTPUT_PATH
        snapshots = pd.read_csv(OUTPUT_PATH, low_memory=False)
    else:
        snapshots = build_monthly_history(manifest, force_parse=args.force_parse)
    training, identity = build_training_dataset()
    comparison = None
    if args.train:
        if args.training_start is not None or args.training_end is not None or args.test_end is not None:
            if None in {args.training_start, args.training_end, args.test_end}:
                raise SystemExit("--training-start, --training-end and --test-end must be supplied together")
            comparison = {"windows": [train_window(args.training_start, args.training_end, args.test_end, training, identity)]}
        else:
            comparison = train_required_windows(training, identity)
    write_report(comparison)
    print(json.dumps({"snapshots": len(snapshots), "training_rows": len(training), "identity_rows": len(identity), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
