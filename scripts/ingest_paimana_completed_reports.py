#!/usr/bin/env python3
"""Fetch March PAIMANA reports and extract official completed-project outcomes."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from backend.app.services.paimana_ingestion_service import _fetch, discover_archive_reports, extract_report_text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paimana_completed_archive"
OUTPUT = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"
MONTHS = {name.lower(): number for number, name in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}


def number(value: str) -> float:
    return float(value.replace(",", ""))


def parse_completed_projects(text: str, source_url: str, financial_year: str) -> pd.DataFrame:
    """Parse the completed-project table found in PAIMANA annual flash reports."""
    start = re.search(r"Month wise List of Completed Projects", text, re.I)
    if not start:
        return pd.DataFrame()
    tail = text[start.start():]
    end = re.search(r"(?:Detail(?:s)? of [Oo]ngoing|TABLE-\s*3|Project List:\s*Ongoing)", tail, re.I)
    section = tail[:end.start()] if end else tail
    rows, current_month, current_sector, pending = [], None, "Not reported", []
    month_re = re.compile(r"^\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(20\d{2})\s*$", re.I)
    numeric_re = re.compile(r"(?P<cost>[\d,]+(?:\.\d+)?)\s+(?P<planned>\d{2}/\d{4})\s+(?P<expenditure>[\d,]+(?:\.\d+)?)\s*$")
    sector_re = re.compile(r"^[A-Z][A-Z &/()\-]{2,}$")

    def flush() -> None:
        nonlocal pending
        if not pending or current_month is None:
            pending = []
            return
        joined = " ".join(x.strip() for x in pending if x.strip())
        match = numeric_re.search(joined)
        if not match:
            pending = []
            return
        code = re.search(r"\[([A-Z]?\d{8,9})\]", joined)
        agency = re.search(r"\(([^()]{3,160})\)\s*-?\s*\[[A-Z]?\d{8,9}\]", joined)
        name = joined[:match.start()].strip()
        name = re.sub(r"^\d+\s+", "", name)
        name = re.sub(r"\s+", " ", name)
        if len(name) >= 4:
            rows.append({"project_id": code.group(1) if code else None, "project_name": name, "sector": current_sector.title(), "implementing_agency": agency.group(1).strip() if agency else None, "approved_cost_cr": number(match.group("cost")), "planned_commissioning_date": pd.to_datetime(match.group("planned"), format="%m/%Y", errors="coerce"), "reported_completion_expenditure_cr": number(match.group("expenditure")), "completion_date": current_month, "financial_year": financial_year, "source_url": source_url})
        pending = []

    for raw in section.splitlines():
        line = raw.replace("\f", "").strip()
        if not line:
            continue
        month = month_re.match(line)
        if month:
            flush(); current_month = pd.Timestamp(year=int(month.group(2)), month=MONTHS[month.group(1).lower()], day=1) + pd.offsets.MonthEnd(0); continue
        if sector_re.match(line) and not any(word in line for word in ["TABLE", "PROJECT", "COST", "EXPENDITURE", "COMMISSIONING", "MONTH"]):
            flush(); current_sector = line; continue
        if re.match(r"^\d+\s+", line):
            flush(); pending = [line]; continue
        if pending:
            pending.append(line)
    flush()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-year", type=int, default=2001)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    reports = [r for r in discover_archive_reports() if r["label"] == "March" and args.from_year <= int(r["financial_year"][:4]) <= args.to_year]
    RAW.mkdir(parents=True, exist_ok=True)
    frames, manifest = [], []
    for report in reports:
        year = report["financial_year"][:4]
        pdf = RAW / f"march_{report['financial_year']}.pdf"
        if not pdf.exists() and not args.local_only:
            pdf.write_bytes(_fetch(report["url"]))
        if not pdf.exists():
            continue
        frame = parse_completed_projects(extract_report_text(pdf), report["url"], report["financial_year"])
        frames.append(frame)
        manifest.append({**report, "filename": pdf.name, "completed_records": int(len(frame))})
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        raise SystemExit("No completed-project records extracted.")
    # Reports are annual cumulative lists; preserve source rows but remove repeated project/month records.
    data["project_id"] = data["project_id"].fillna("")
    data["project_name_key"] = data.project_name.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    data = data.drop_duplicates(["project_id", "project_name_key", "completion_date"], keep="last").drop(columns="project_name_key")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(data)} official completed-project outcomes to {OUTPUT}")


if __name__ == "__main__":
    main()
