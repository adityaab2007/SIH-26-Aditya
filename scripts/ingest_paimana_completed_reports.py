#!/usr/bin/env python3
"""Fetch March PAIMANA reports and extract official completed-project outcomes."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from backend.app.services.paimana_ingestion_service import ARCHIVE_DIR, MANIFEST_PATH, _fetch, discover_archive_reports, extract_report_text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "paimana_completed_archive"
OUTPUT = ROOT / "data" / "processed" / "paimana_completed_outcomes.csv"
AUDIT = ROOT / "data" / "processed" / "paimana_completed_outcomes_audit.json"
MONTHS = {name.lower(): number for number, name in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}


def number(value: str) -> float:
    cleaned = value.replace(",", "").strip()
    return float(cleaned) if cleaned else float("nan")


def parse_completed_projects(text: str, source_url: str, financial_year: str) -> pd.DataFrame:
    """Parse the completed-project table found in PAIMANA annual flash reports."""
    recent_markers = list(re.finditer(r"Table:-3\.\s*Project List:\s*Completed during\s+(\w+)\s+(20\d{2})", text, re.I))
    recent_start = next((match for match in recent_markers if re.search(
        r"\nSector\s+Sl\.\s*No\.", text[match.end():match.end() + 500], re.I,
    )), None)
    if recent_start:
        month_name, year = recent_start.group(1), int(recent_start.group(2))
        completion_date = pd.Timestamp(year=year, month=MONTHS[month_name.lower()], day=1) + pd.offsets.MonthEnd(0)
        tail = text[recent_start.start():]
        end = re.search(r"Table:-4\.\s*Project List:\s*Added", tail, re.I)
        section = tail[:end.start()] if end else tail
        start_re = re.compile(
            r"^\s*(?P<seq>\d+)\s+(?P<name>.+?)\s{2,}(?P<cost>\d[\d,]*(?:\.\d+)?)\s+"
            r"(?P<planned>\d{1,2}/\d{4})\s+(?P<expenditure>\d[\d,]*(?:\.\d+)?)\s*$"
        )
        rows: list[dict] = []; current_sector = "Not reported"
        lines = section.splitlines(); starts: list[tuple[int, re.Match[str], str]] = []
        for index, raw in enumerate(lines):
            stripped = raw.strip()
            if re.fullmatch(r"[A-Z][A-Z &/()\-]{2,}", stripped) and not any(word in stripped for word in ("PROJECT", "COST", "DATE", "SECTOR", "TOTAL")):
                current_sector = stripped
            match = start_re.match(raw)
            if match:
                starts.append((index, match, current_sector))
        for position, (index, match, sector) in enumerate(starts):
            stop = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
            continuation = "\n".join(lines[index + 1:stop])
            code = re.search(r"\(([A-Z]?\d{8,9})\s*\)", continuation)
            agency_match = re.search(r"\(([^()\n]+)\)\s*\n\s*\([A-Z]?\d{8,9}\s*\)", continuation)
            agency = agency_match.group(1).strip() if agency_match else None
            rows.append({
                "project_id": code.group(1) if code else None,
                "project_name": re.sub(r"\s+", " ", match.group("name")).strip(),
                "sector": sector.title(), "implementing_agency": agency,
                "approved_cost_cr": number(match.group("cost")),
                "planned_commissioning_date": pd.to_datetime(match.group("planned"), format="%m/%Y", errors="coerce"),
                "reported_completion_expenditure_cr": number(match.group("expenditure")),
                "completion_date": completion_date, "financial_year": financial_year, "source_url": source_url,
            })
        return pd.DataFrame(rows)

    # PAIMANA's older annual reports use "Completed Projects Costing..." while
    # newer reports use "Month wise List of Completed Projects". Both contain
    # the same official completed-project table.
    start = re.search(r"(?:Month\s+wise\s+List\s+of\s+Completed\s+Project(?:s)?|Completed\s+Projects\s+Costing)", text, re.I)
    if not start:
        return pd.DataFrame()
    tail = text[start.start():]
    end = re.search(r"(?:Detail(?:s)? of [Oo]ngoing|TABLE-\s*3|Project List:\s*Ongoing)", tail, re.I)
    section = tail[:end.start()] if end else tail
    rows, current_month, current_sector, pending = [], None, "Not reported", []
    month_re = re.compile(r"^\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(20\d{2})\s*$", re.I)
    # The identifier/agency is usually printed on continuation lines after the
    # three numeric columns. Do not anchor this pattern to the joined block end.
    numeric_re = re.compile(r"(?P<cost>\d[\d,]*(?:\.\d+)?)\s+(?P<planned>\d{2}/\d{4})\s+(?P<expenditure>\d[\d,]*(?:\.\d+)?)\b")
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
        name = re.sub(r"^\d+\.?\s+", "", name)
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
        if sector_re.match(line) and "(" not in line and ")" not in line and len(line) < 60 and not any(word in line for word in ["TABLE", "PROJECT", "COST", "EXPENDITURE", "COMMISSIONING", "MONTH"]):
            flush(); current_sector = line; continue
        if re.match(r"^\d+\.?\s+", line):
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
    monthly_manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else []
    reports = [r for r in monthly_manifest if r.get("download_status") in {"downloaded", "cached"}
               and args.from_year <= int(r["financial_year"][:4]) <= args.to_year
               and (r.get("report_month") == "March" or r["financial_year"] == "2024-25")]
    if not reports:
        reports = [r for r in discover_archive_reports() if r["label"] == "March"
                   and args.from_year <= int(r["financial_year"][:4]) <= args.to_year]
    RAW.mkdir(parents=True, exist_ok=True)
    frames, manifest = [], []
    for report in reports:
        year = report["financial_year"][:4]
        filename = report.get("downloaded_filename")
        pdf = ARCHIVE_DIR / report["financial_year"] / filename if filename else RAW / f"march_{report['financial_year']}.pdf"
        if not pdf.exists() and not args.local_only:
            pdf.write_bytes(_fetch(report["url"]))
        if not pdf.exists():
            continue
        source_url = report.get("source_url", report.get("url", ""))
        frame = parse_completed_projects(extract_report_text(pdf), source_url, report["financial_year"])
        frames.append(frame)
        manifest.append({**report, "filename": pdf.name, "completed_records": int(len(frame)), "completed_records_with_official_id": int(frame.project_id.notna().sum()) if not frame.empty else 0})
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        raise SystemExit("No completed-project records extracted.")
    # Reports are annual cumulative lists. Official project code is the primary
    # identity: retain the latest published outcome for each code and audit the
    # historical textual variants instead of making that code ambiguous.
    data["project_id"] = data["project_id"].replace("", pd.NA)
    data["project_name_key"] = data.project_name.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    coded = data[data.project_id.notna()].copy()
    code_variant_counts = coded.groupby("project_id").size()
    coded = coded.sort_values(["financial_year", "completion_date"]).drop_duplicates("project_id", keep="last")
    legacy = data[data.project_id.isna()].drop_duplicates(["project_name_key", "completion_date"], keep="last")
    data = pd.concat([coded, legacy], ignore_index=True).drop(columns="project_name_key")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    AUDIT.write_text(json.dumps({
        "raw_cumulative_rows": int(sum(item["completed_records"] for item in manifest)),
        "canonical_outcomes": int(len(data)),
        "canonical_outcomes_with_official_id": int(data.project_id.notna().sum()),
        "unique_official_project_ids": int(data.project_id.nunique()),
        "official_ids_with_historical_variants": int(code_variant_counts.gt(1).sum()),
        "deduplication_policy": "Latest official cumulative record per exact project code; code-less rows use exact normalized name and completion date only.",
    }, indent=2))
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(data)} official completed-project outcomes to {OUTPUT}")


if __name__ == "__main__":
    main()
