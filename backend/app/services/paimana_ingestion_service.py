"""Discover, cache and normalize the complete official PAIMANA monthly archive.

Raw PDFs are immutable inputs. Parsing is resumable by source SHA-256 and parser
version. Missing official values remain null and ambiguous conflicts are audited.
"""
from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
from pathlib import Path
import json
import re
import shutil
import subprocess
from urllib.parse import parse_qs, quote, urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from backend.app.services.paimana_parsers import CANONICAL_COLUMNS, ParseContext, detect_parser, parse_report
from backend.app.services.paimana_parsers.recent import RecentProjectListParser

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = ROOT / "data" / "raw" / "paimana_archive"
MANIFEST_PATH = ARCHIVE_DIR / "manifest.json"
PARSE_CACHE = ARCHIVE_DIR / "parse_cache"
OUTPUT_PATH = ROOT / "data" / "processed" / "paimana_monthly_snapshots.csv"
LEGACY_OUTPUT_PATH = ROOT / "data" / "processed" / "project_monthly_history.csv"
AUDIT_PATH = ROOT / "data" / "processed" / "paimana_monthly_ingestion_audit.json"
CONFLICT_PATH = ROOT / "data" / "processed" / "paimana_monthly_conflicts.csv"
BASE_URL = "https://paimana-proj.mospi.gov.in"
ARCHIVE_PAGE = f"{BASE_URL}/ReportPage/ArchiveProjectMonitoring"
INDEX_URL = f"{BASE_URL}/ReportPage/ArchiveReport?fyear=&month=0&quater=0&reportType=F"
KNOWN_MISSING_MONTHS = {("2005-06", "April"), ("2008-09", "January")}
MONTH_NAMES = tuple("January February March April May June July August September October November December".split())
# Backward-compatible schema for the untouched five-feature baseline history.
OUTPUT_COLUMNS = [
    "project_id", "project_name", "sector", "ministry", "state", "implementing_agency",
    "original_cost", "revised_cost", "current_expenditure", "planned_start_date",
    "planned_completion_date", "revised_completion_date", "actual_completion_date", "month",
    "physical_progress_percentage", "financial_progress_percentage", "milestone_status", "delay_months",
    "source_report", "source_url",
]


def _fetch(url: str) -> bytes:
    curl = shutil.which("curl")
    if curl:
        completed = subprocess.run(
            [curl, "-LfsS", "--connect-timeout", "15", "--max-time", "180", "--retry", "2",
             "--user-agent", "InfraSight-SIH26103/2.0 (+official public PAIMANA archive ingestion)", url],
            check=True, capture_output=True,
        )
        return completed.stdout
    request = Request(url, headers={"User-Agent": "InfraSight-SIH26103/2.0 (+official public PAIMANA archive ingestion)"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def _report_month(financial_year: str, label: str) -> pd.Timestamp | None:
    match = re.search("|".join(MONTH_NAMES), label, re.I)
    if not match or not re.fullmatch(r"\d{4}-\d{2}", financial_year):
        return None
    month = datetime.strptime(match.group(0).title(), "%B").month
    start_year = int(financial_year[:4]); year = start_year + 1 if month <= 3 else start_year
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def _part_number(label: str) -> int | None:
    match = re.search(r"Part\s*[- ]?(\d+|I{1,3})", label, re.I)
    if not match:
        return None
    value = match.group(1).upper()
    return {"I": 1, "II": 2, "III": 3}.get(value, int(value) if value.isdigit() else None)


def _filename(report: dict) -> str:
    suffix = Path(report["archive_path"].replace("\\", "/")).suffix.lower() or ".pdf"
    part = f"_part{report['part_number']}" if report.get("part_number") else ""
    return f"{report['financial_year']}_{report['report_month'].lower()}{part}_{report['id']}{suffix}"


def discover_archive_reports() -> list[dict]:
    """Return every official monthly Flash Report listed by PAIMANA."""
    payload = json.loads(_fetch(INDEX_URL).decode("utf-8")); html = unescape(payload["html"])
    reports: list[dict] = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S):
        cells = [re.sub(r"<[^>]+>", "", value).strip() for value in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)]
        link = re.search(r"href=['\"]([^'\"]+)['\"]", row, flags=re.I)
        if len(cells) < 3 or not link:
            continue
        relative = link.group(1).replace("\\", "/"); parsed = urlparse(urljoin(BASE_URL, relative))
        if parsed.netloc != urlparse(BASE_URL).netloc or not parsed.path.endswith("/ReportPage/ViewPdf"):
            continue
        query = parse_qs(parsed.query); raw_path = query.get("path", [""])[0].replace("\\", "/")
        report_date = _report_month(cells[1], cells[2])
        if report_date is None:
            continue
        source_url = f"{BASE_URL}/ReportPage/ViewPdf?id={quote(query.get('id', [''])[0])}&path={quote(raw_path)}"
        report = {"financial_year": cells[1], "calendar_year": int(report_date.year),
                  "report_month": report_date.strftime("%B"), "report_label": cells[2], "label": cells[2],
                  "part_number": _part_number(cells[2]), "id": query.get("id", [""])[0],
                  "official_archive_path": raw_path, "archive_path": raw_path, "source_url": source_url, "url": source_url,
                  "discovery_status": "discovered", "download_status": "pending", "parsing_status": "pending",
                  "parser_version": None, "parsing_errors": []}
        report["downloaded_filename"] = _filename(report); report["filename"] = report["downloaded_filename"]
        reports.append(report)
    reports.sort(key=lambda item: (item["financial_year"], datetime.strptime(item["report_month"], "%B").month, item.get("part_number") or 0))
    return reports


def archive_coverage(reports: list[dict]) -> dict:
    found = {(r["financial_year"], r["report_month"]) for r in reports}; years = sorted({r["financial_year"] for r in reports})
    missing = sorted({(fy, month) for fy in years for month in MONTH_NAMES} - found)
    return {"financial_years": years, "financial_year_count": len(years), "reports_discovered": len(reports),
            "months_covered": len(found),
            "missing_months": [{"financial_year": fy, "month": month, "known_archive_gap": (fy, month) in KNOWN_MISSING_MONTHS} for fy, month in missing],
            "unexpected_missing_months": [{"financial_year": fy, "month": month} for fy, month in missing if (fy, month) not in KNOWN_MISSING_MONTHS]}


def _load_previous_manifest() -> dict[tuple[str, str, int | None], dict]:
    if not MANIFEST_PATH.exists():
        return {}
    return {(r["financial_year"], r.get("report_label", r.get("label", "")), r.get("part_number")): r for r in json.loads(MANIFEST_PATH.read_text())}


def download_archive_reports(financial_year: str | None = None, labels: set[str] | None = None, force: bool = False,
                             *, from_year: int = 2001, to_year: int = 2024) -> list[dict]:
    """Download all selected reports idempotently, retaining failures."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True); previous = _load_previous_manifest(); selected = []
    for report in discover_archive_reports():
        start = int(report["financial_year"][:4])
        if financial_year and report["financial_year"] != financial_year:
            continue
        if not from_year <= start <= to_year or (labels is not None and report["report_label"] not in labels):
            continue
        selected.append(report)
    now = datetime.now(timezone.utc).isoformat()
    def fetch_one(report: dict) -> dict:
        key = (report["financial_year"], report["report_label"], report.get("part_number")); old = previous.get(key, {})
        year_dir = ARCHIVE_DIR / report["financial_year"]; year_dir.mkdir(parents=True, exist_ok=True)
        target = year_dir / report["downloaded_filename"]
        legacy_target = ARCHIVE_DIR / old.get("filename", "") if old.get("filename") else None
        try:
            if not target.exists() and legacy_target and legacy_target.is_file():
                shutil.copy2(legacy_target, target)
            cached = target.exists() and target.read_bytes()[:4] == b"%PDF"
            if force or not cached:
                data = _fetch(report["source_url"])
                if not data.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
                if target.exists() and sha256(target.read_bytes()).hexdigest() != sha256(data).hexdigest():
                    target.replace(target.with_suffix(f".{sha256(target.read_bytes()).hexdigest()[:12]}.pdf"))
                target.write_bytes(data); state = "downloaded"
            else:
                state = "cached"
            digest = sha256(target.read_bytes()).hexdigest()
            report.update({"download_status": state, "status": "downloaded", "bytes": target.stat().st_size, "sha256": digest,
                           "download_timestamp_utc": old.get("download_timestamp_utc") if state == "cached" else now,
                           "downloaded_at_utc": old.get("downloaded_at_utc") if state == "cached" else now})
        except Exception as exc:
            report.update({"download_status": "failed", "status": "failed", "download_timestamp_utc": now,
                           "download_error": f"{type(exc).__name__}: {exc}", "error": f"{type(exc).__name__}: {exc}"})
        report["parsing_status"] = old.get("parsing_status", "pending"); report["parser_version"] = old.get("parser_version")
        report["parsing_errors"] = old.get("parsing_errors", [])
        return report

    # Bounded concurrency keeps the official host load modest while avoiding a
    # single slow historical PDF serializing the entire reproducible run.
    with ThreadPoolExecutor(max_workers=4) as pool:
        selected = list(pool.map(fetch_one, selected))
    MANIFEST_PATH.write_text(json.dumps(selected, indent=2)); return selected


def extract_report_text(pdf_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        completed = subprocess.run([pdftotext, "-layout", str(pdf_path), "-"], check=True, capture_output=True)
        return completed.stdout.decode("utf-8", errors="replace")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf or Poppler pdftotext to parse PAIMANA PDFs") from exc
    return "\f".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)


def parse_project_list(text: str, report_month: pd.Timestamp, source_report: str, source_url: str) -> pd.DataFrame:
    """Backward-compatible direct entrypoint for the recent project-list parser."""
    return RecentProjectListParser.parse(text, ParseContext("unknown", report_month, source_report, source_url)).frame


def _cache_path(report: dict, parser_version: str) -> Path:
    return PARSE_CACHE / f"{report['sha256']}_{parser_version}.csv"


def _combine_part_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, pd.DataFrame()
    conflicts: list[dict] = []; rows: list[dict] = []
    for (project_id, snapshot_date), group in frame.groupby(["project_id", "snapshot_date"], dropna=False, sort=False):
        if pd.isna(project_id):
            rows.extend(group.to_dict("records")); continue
        merged = group.iloc[0].to_dict()
        for column in CANONICAL_COLUMNS:
            values = [value for value in group[column].tolist() if pd.notna(value) and str(value).strip()]
            unique = list(dict.fromkeys(map(str, values)))
            if len(unique) > 1 and column not in {"source_report", "source_url", "source_page", "parser_version"}:
                conflicts.append({"project_id": project_id, "snapshot_date": snapshot_date, "field": column, "values": " | ".join(unique)})
            if values:
                merged[column] = values[-1]
        merged["source_report"] = " | ".join(dict.fromkeys(group.source_report.astype(str)))
        merged["source_url"] = " | ".join(dict.fromkeys(group.source_url.astype(str))); rows.append(merged)
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS), pd.DataFrame(conflicts)


def snapshot_quality(frame: pd.DataFrame) -> dict:
    """Report quality violations without silently dropping official rows."""
    if frame.empty:
        return {"rows": 0}
    approved = pd.to_numeric(frame.approved_cost_cr, errors="coerce")
    revised = pd.to_numeric(frame.revised_cost_cr, errors="coerce")
    expenditure = pd.to_numeric(frame.cumulative_expenditure_cr, errors="coerce")
    progress = pd.to_numeric(frame.physical_progress, errors="coerce")
    snapshot = pd.to_datetime(frame.snapshot_date, errors="coerce")
    planned = pd.to_datetime(frame.planned_completion_date, errors="coerce")
    start = pd.to_datetime(frame.planned_start_date, errors="coerce")
    return {"rows": int(len(frame)), "approved_cost_not_positive": int(approved.notna().mul(approved.le(0)).sum()),
            "revised_cost_negative": int(revised.notna().mul(revised.lt(0)).sum()),
            "expenditure_negative": int(expenditure.notna().mul(expenditure.lt(0)).sum()),
            "physical_progress_outside_0_100": int(progress.notna().mul(~progress.between(0, 100)).sum()),
            "invalid_snapshot_date": int(snapshot.isna().sum()),
            "missing_project_identity": int(frame.project_id.isna().mul(frame.project_name.astype("string").str.strip().eq("") | frame.project_name.isna()).sum()),
            "planned_completion_before_start": int((planned.notna() & start.notna() & planned.lt(start)).sum()),
            "duplicate_code_snapshot_rows": int(frame[frame.project_id.notna()].duplicated(["project_id", "snapshot_date"], keep=False).sum()),
            "policy": "Violations are retained and surfaced for audit; missing values are never replaced by zero."}


def build_monthly_history(manifest: list[dict] | None = None, output_path: Path = OUTPUT_PATH, *, force_parse: bool = False) -> pd.DataFrame:
    manifest = manifest if manifest is not None else (json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else [])
    PARSE_CACHE.mkdir(parents=True, exist_ok=True); frames = []; parse_counts: dict[str, int] = {}
    for report in manifest:
        if report.get("download_status", report.get("status")) not in {"downloaded", "cached"}:
            continue
        month = _report_month(report["financial_year"], report.get("report_label", report.get("label", "")))
        pdf = ARCHIVE_DIR / report["financial_year"] / report.get("downloaded_filename", report.get("filename", ""))
        if not pdf.exists():
            pdf = ARCHIVE_DIR / report.get("filename", "")
        try:
            text = extract_report_text(pdf); parser = detect_parser(text); parser_version = parser.version if parser else "unrecognized"
            cache = _cache_path(report, parser_version)
            if cache.exists() and not force_parse:
                frame = pd.read_csv(cache, dtype={"project_id": "string"}); warnings = []
            else:
                result = parse_report(text, ParseContext(report["financial_year"], month, report.get("downloaded_filename", pdf.name), report.get("source_url", report.get("url", ""))))
                frame, warnings = result.frame, result.warnings; frame.to_csv(cache, index=False)
            report.update({"parser_version": parser_version, "parsing_status": "parsed" if not frame.empty else "no_rows", "parsing_errors": warnings, "parsed_observations": int(len(frame))})
            parse_counts[parser_version] = parse_counts.get(parser_version, 0) + 1
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            report.update({"parsing_status": "failed", "parsing_errors": [f"{type(exc).__name__}: {exc}"], "parsed_observations": 0})
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)
    result, conflicts = _combine_part_rows(result)
    if not result.empty:
        result = result.sort_values(["project_id", "project_name", "snapshot_date"], na_position="last").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output_path, index=False)
    # The legacy compatibility output historically guaranteed an official code
    # on every row. Code-less historical observations remain in the canonical
    # snapshot output and identity audit, not this compatibility view.
    # This path belongs to the historical five-feature production baseline.
    # A lifecycle rebuild must not rewrite that controlled-comparison input.
    if not LEGACY_OUTPUT_PATH.exists():
        result[result.project_id.notna()].to_csv(LEGACY_OUTPUT_PATH, index=False)
    conflicts.to_csv(CONFLICT_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2)); coverage = archive_coverage(manifest)
    audit = {**coverage, "reports_downloaded_or_cached": sum(r.get("download_status") in {"downloaded", "cached"} for r in manifest),
             "reports_parsed": sum(r.get("parsing_status") == "parsed" for r in manifest),
             "reports_without_rows": sum(r.get("parsing_status") == "no_rows" for r in manifest),
             "reports_failed_parsing": sum(r.get("parsing_status") == "failed" for r in manifest),
             "parser_report_counts": parse_counts, "monthly_observations": int(len(result)),
             "unique_reported_project_codes": int(result.project_id.nunique()) if not result.empty else 0,
             "duplicate_conflicts": int(len(conflicts)), "data_quality": snapshot_quality(result),
             "observations_by_financial_year": result.financial_year.value_counts().sort_index().astype(int).to_dict() if not result.empty else {}}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2)); return result


def ingest_latest_archive() -> pd.DataFrame:
    """Backward-compatible entrypoint; now ingests the complete official archive."""
    return build_monthly_history(download_archive_reports())
