from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.monthly_lifecycle import engineer_as_of_features, resolve_identities
from backend.app.ml.monthly_training import temporal_project_split
from backend.app.services import paimana_ingestion_service as ingestion
from backend.app.services.paimana_parsers import ParseContext, detect_parser, parse_report
from scripts.ingest_paimana_completed_reports import parse_completed_projects
from backend.app.services import monthly_prediction_service as monthly_prediction


def context(year="2019-20"):
    return ParseContext(year, pd.Timestamp("2019-06-30"), "official.pdf", "https://official.example/report")


def test_archive_discovery_parses_year_month_part_and_official_url(monkeypatch):
    html = """<table><tr><td>1</td><td>2024-25</td><td>April (Part 2)</td><td><a href='../ReportPage/ViewPdf?id=62&amp;path=Content\\ArchiveReport\\flash\\2024-25\\April-II.pdf'>PDF</a></td></tr></table>"""
    monkeypatch.setattr(ingestion, "_fetch", lambda _: json.dumps({"html": html}).encode())
    rows = ingestion.discover_archive_reports()
    assert rows[0]["financial_year"] == "2024-25"
    assert rows[0]["calendar_year"] == 2024
    assert rows[0]["report_month"] == "April"
    assert rows[0]["part_number"] == 2
    assert rows[0]["source_url"].startswith(ingestion.BASE_URL)


def test_known_archive_gaps_are_not_parser_failures():
    reports = []
    for year in ["2005-06", "2008-09"]:
        missing = "April" if year == "2005-06" else "January"
        for month in ingestion.MONTH_NAMES:
            if month != missing:
                reports.append({"financial_year": year, "report_month": month})
    coverage = ingestion.archive_coverage(reports)
    assert not coverage["unexpected_missing_months"]
    assert all(item["known_archive_gap"] for item in coverage["missing_months"])


def test_classic_parser_preserves_code_and_nulls():
    text = """Detail of ongoing Projects Costing Rs.150 Crore and above.
Date of Commissioning\nSI.No Project Date of Approval Original / Revised Cost Anticipated Cost Cumulative Expenditure
ATOMIC ENERGY
  1 KAKRAPAR ATOMIC POWER PROJECT - 3 AND 4 - 10/2009 11,459.00 16,580.00 13,711.00 12/2015 06/2020 54(O) 52/70
    [N02000010]NPCIL,GUJARAT
"""
    parser = detect_parser(text); result = parse_report(text, context())
    assert parser.version == "classic-code-v3"
    row = result.frame.iloc[0]
    assert row.project_id == "N02000010"
    assert row.approved_cost_cr == 11459
    assert pd.isna(row.physical_progress)


def test_legacy_parser_preserves_code_less_rows_for_identity_audit():
    text = """Sector-Wise analysis of projects
Name of the Project Date of Approval Cost Original [Anticipated] Cumulative Expenditure Date of Commissioning [Anticipated]
COAL
1. SAMPLE LEGACY MINE 6/1995 415.93 -257.69 115.22 3/2002 0 0/8
   [158.24] [3/2002]
"""
    result = parse_report(text, context("2001-02"))
    assert result.parser_version == "legacy-sector-v1"
    assert len(result.frame) == 1
    assert pd.isna(result.frame.iloc[0].project_id)
    assert result.frame.iloc[0].approved_cost_cr == pytest.approx(415.93)


def test_recent_parser_handles_spaced_project_code_and_real_progress():
    text = """Project List: Ongoing Projects as of 30th June 2024
STATE      POWER             1    SAMPLE PROJECT           1/2020           3/2024               500.00           250.00            50
                                      (Mar-25)             (550.00)
                                      {3/2025}             {575.00}
                                      (AGENCY )
                                      (N18000001 )
"""
    result = parse_report(text, context("2024-25"))
    assert result.parser_version == "recent-project-list-v3"
    assert result.frame.iloc[0].project_id == "N18000001"
    assert result.frame.iloc[0].physical_progress == 50


def test_part_reports_are_coalesced_by_code_and_snapshot_with_conflict_audit():
    columns = ingestion.CANONICAL_COLUMNS
    first = {column: None for column in columns}; second = {column: None for column in columns}
    first.update({"project_id": "N18000001", "project_name": "Project", "snapshot_date": "2024-04-30", "approved_cost_cr": 500, "source_report": "part1", "source_url": "one", "parser_version": "recent"})
    second.update({"project_id": "N18000001", "project_name": "Project", "snapshot_date": "2024-04-30", "approved_cost_cr": 510, "physical_progress": 40, "source_report": "part2", "source_url": "two", "parser_version": "recent"})
    merged, conflicts = ingestion._combine_part_rows(pd.DataFrame([first, second], columns=columns))
    assert len(merged) == 1
    assert merged.iloc[0].physical_progress == 40
    assert conflicts.field.tolist() == ["approved_cost_cr"]


def test_malformed_report_is_quarantined():
    result = parse_report("not a PAIMANA table", context())
    assert result.frame.empty
    assert result.parser_version == "unrecognized"
    assert result.warnings


def outcomes():
    return pd.DataFrame([
        {"project_id": "N02000010", "project_name": "Project A", "sector": "Power", "implementing_agency": "Agency A", "approved_cost_cr": 100,
         "planned_commissioning_date": "2018-12-31", "reported_completion_expenditure_cr": 130, "completion_date": "2020-12-31"},
        {"project_id": "N02000011", "project_name": "Ambiguous", "sector": "Power", "implementing_agency": "Agency A", "approved_cost_cr": 100,
         "planned_commissioning_date": "2019-12-31", "reported_completion_expenditure_cr": 110, "completion_date": "2021-12-31"},
        {"project_id": "N02000012", "project_name": "Ambiguous", "sector": "Power", "implementing_agency": "Agency B", "approved_cost_cr": 100,
         "planned_commissioning_date": "2019-12-31", "reported_completion_expenditure_cr": 120, "completion_date": "2022-12-31"},
    ])


def snapshots():
    return pd.DataFrame([
        {"project_id": "N02000010", "project_name": "Project A", "snapshot_date": "2019-01-31", "financial_year": "2018-19", "report_month": "January", "sector": "Power", "implementing_agency": "Agency A", "approved_cost_cr": 100, "revised_cost_cr": 105, "cumulative_expenditure_cr": 40, "planned_start_date": "2018-01-01", "planned_completion_date": "2018-12-31", "revised_completion_date": "2019-06-30", "physical_progress": 30, "parser_version": "fixture"},
        {"project_id": "N02000010", "project_name": "Project A", "snapshot_date": "2019-04-30", "financial_year": "2019-20", "report_month": "April", "sector": "Power", "implementing_agency": "Agency A", "approved_cost_cr": 100, "revised_cost_cr": 110, "cumulative_expenditure_cr": 60, "planned_start_date": "2018-01-01", "planned_completion_date": "2018-12-31", "revised_completion_date": "2019-12-31", "physical_progress": 50, "parser_version": "fixture"},
        {"project_id": None, "project_name": "Ambiguous", "snapshot_date": "2019-04-30", "financial_year": "2019-20", "report_month": "April", "sector": "Power", "implementing_agency": "Agency A", "approved_cost_cr": 100, "parser_version": "fixture"},
    ])


def test_identity_prefers_exact_code_and_rejects_ambiguous_name():
    resolved, audit = resolve_identities(snapshots(), outcomes())
    assert resolved.iloc[0].identity_method == "exact_official_project_id"
    assert bool(resolved.iloc[0].identity_verified)
    assert audit.iloc[-1].identity_method == "ambiguous_exact_name"
    assert not bool(audit.iloc[-1].identity_verified)


def test_as_of_velocity_ignores_future_snapshot():
    resolved, _ = resolve_identities(snapshots().iloc[:2], outcomes())
    first = engineer_as_of_features(resolved.iloc[:1], outcomes()).iloc[0]
    both = engineer_as_of_features(resolved, outcomes()).iloc[0]
    assert pd.isna(first.progress_velocity_3m)
    assert pd.isna(both.progress_velocity_3m)
    second = engineer_as_of_features(resolved, outcomes()).iloc[1]
    assert second.progress_velocity_3m > 0


def test_direct_lifecycle_feature_formulas_use_snapshot_values():
    resolved, _ = resolve_identities(snapshots().iloc[:1], outcomes())
    row = engineer_as_of_features(resolved, outcomes()).iloc[0]
    assert row.expenditure_ratio == pytest.approx(0.4)
    assert row.cost_escalation_percentage == pytest.approx(5.0)
    assert row.planned_duration_days == 364
    assert row.schedule_slippage_days == 181
    assert row.progress_deviation == pytest.approx(-70.0)


def test_future_completed_projects_cannot_influence_priors():
    resolved, _ = resolve_identities(snapshots().iloc[:1], outcomes())
    before = engineer_as_of_features(resolved, outcomes()).iloc[0]
    changed = outcomes().copy(); changed.loc[changed.completion_date.eq("2022-12-31"), "reported_completion_expenditure_cr"] = 99999
    after = engineer_as_of_features(resolved, changed).iloc[0]
    assert (pd.isna(before.sector_average_cost_overrun) and pd.isna(after.sector_average_cost_overrun)) or before.sector_average_cost_overrun == after.sector_average_cost_overrun


def test_project_balancing_weights_sum_to_one_per_project():
    resolved, _ = resolve_identities(snapshots().iloc[:2], outcomes())
    frame = engineer_as_of_features(resolved, outcomes())
    assert frame.groupby("canonical_project_id").sample_weight.sum().iloc[0] == pytest.approx(1.0)


def test_temporal_split_rejects_same_project_in_train_and_test():
    frame = pd.DataFrame({"canonical_project_id": ["A", "A"], "completion_year": [2015, 2016]})
    with pytest.raises(ValueError, match="Project-group leakage"):
        temporal_project_split(frame, 2001, 2015, 2021)


def test_completed_parser_retains_code_on_multiline_agency_block():
    text = """Month wise List of Completed Projects
April,2019
COAL
  1 KHADIA EXPANSION OPENCAST PROJECT(4 TO 10 MTPA,6MTPA 1,131.28 03/2018 796.85
    INCREMENTAL) (NORTHERN COAL FIELDS
    LIMITED) - [N06000091]
Details of Ongoing Projects
"""
    frame = parse_completed_projects(text, "official", "2019-20")
    assert frame.iloc[0].project_id == "N06000091"


def test_recent_completed_parser_uses_report_month_as_actual_completion():
    text = """Table:-3. Project List: Completed during March 2025
Sector Sl. No. Project Name Original Date of Commissioning Cumulative
PETROLEUM
  1 SAMPLE PIPELINE PROJECT                         196.26       07/2020       210.00
    (HPCL)
    (N16000378 )
    (ANDHRA PRADESH)
Table:-4. Project List: Added during March 2025
"""
    frame = parse_completed_projects(text, "official", "2024-25")
    assert frame.iloc[0].project_id == "N16000378"
    assert frame.iloc[0].completion_date == pd.Timestamp("2025-03-31")
    assert frame.iloc[0].implementing_agency == "HPCL"


def test_comparison_payload_is_strict_json_compatible(tmp_path, monkeypatch):
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps({"windows": [{"window": "2015_2021", "metric": None}]}, allow_nan=False))
    monkeypatch.setattr(monthly_prediction, "COMPARISON", report)
    payload = monthly_prediction.lifecycle_comparison()
    assert payload["available"] is True
    assert json.dumps(payload, allow_nan=False)


class _Model:
    def __init__(self, value): self.value = value
    def predict(self, frame): return np.array([self.value] * len(frame))


def test_monthly_inference_uses_latest_real_snapshot_and_trajectory(monkeypatch):
    frame = pd.DataFrame([
        {"project_id": "N18000001", "project_name": "Project", "snapshot_date": pd.Timestamp("2020-01-31"), "approved_cost_cr": 100, "progress_velocity_3m": np.nan},
        {"project_id": "N18000001", "project_name": "Project", "snapshot_date": pd.Timestamp("2020-04-30"), "approved_cost_cr": 100, "progress_velocity_3m": 5.0},
    ])
    bundle = {"metadata": {"features_used": ["approved_cost_cr", "progress_velocity_3m"], "model_version": "monthly-test"},
              "cost": _Model(12), "delay": _Model(90), "risk": _Model("MEDIUM"),
              "importance": {"cost": {"features": [{"feature": "progress_velocity_3m", "importance": .7}]}}}
    monkeypatch.setattr(monthly_prediction, "_inference_frame", lambda: frame)
    monkeypatch.setattr(monthly_prediction, "_bundle", lambda _: bundle)
    result = monthly_prediction.lifecycle_project_forecast("N18000001", "test")
    assert result["snapshot_date"] == "2020-04-30"
    assert result["history_snapshots"] == 2
    assert result["model_inputs"]["progress_velocity_3m"] == 5.0


def test_monthly_inference_keeps_missing_single_snapshot_trajectory_null(monkeypatch):
    frame = pd.DataFrame([{"project_id": "N18000001", "project_name": "Project", "snapshot_date": pd.Timestamp("2020-01-31"),
                           "approved_cost_cr": 100, "progress_velocity_3m": np.nan}])
    bundle = {"metadata": {"features_used": ["approved_cost_cr", "progress_velocity_3m"], "model_version": "monthly-test"},
              "cost": _Model(12), "delay": _Model(90), "risk": _Model("MEDIUM"),
              "importance": {"cost": {"features": []}}}
    monkeypatch.setattr(monthly_prediction, "_inference_frame", lambda: frame)
    monkeypatch.setattr(monthly_prediction, "_bundle", lambda _: bundle)
    result = monthly_prediction.lifecycle_project_forecast("N18000001", "test")
    assert result["history_snapshots"] == 1
    assert result["model_inputs"]["progress_velocity_3m"] is None
