"""Download and normalize official PAIMANA Project Monitoring archive reports.

Raw PDFs are immutable inputs. The parser intentionally keeps unavailable report
fields null instead of synthesizing values.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from pathlib import Path
import json
import re
import shutil
import subprocess
from urllib.parse import parse_qs, quote, urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = ROOT / "data" / "raw" / "paimana_archive"
OUTPUT_PATH = ROOT / "data" / "processed" / "project_monthly_history.csv"
BASE_URL = "https://paimana-proj.mospi.gov.in"
ARCHIVE_PAGE = f"{BASE_URL}/ReportPage/ArchiveProjectMonitoring"
INDEX_URL = f"{BASE_URL}/ReportPage/ArchiveReport?fyear=N&month=0&quater=0&reportType=F"

OUTPUT_COLUMNS = [
    "project_id", "project_name", "sector", "ministry", "state", "implementing_agency",
    "original_cost", "revised_cost", "current_expenditure", "planned_start_date",
    "planned_completion_date", "revised_completion_date", "actual_completion_date", "month",
    "physical_progress_percentage", "financial_progress_percentage", "milestone_status", "delay_months",
    "source_report", "source_url",
]


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "InfraSight-SIH26103/1.0 (+public PAIMANA archive ingestion)"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def discover_archive_reports() -> list[dict]:
    payload = json.loads(_fetch(INDEX_URL).decode("utf-8"))
    html = unescape(payload["html"])
    reports = []
    for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.I | re.S):
        cells = [re.sub(r"<[^>]+>", "", value).strip() for value in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)]
        link = re.search(r"href=['\"]([^'\"]+)['\"]", row, flags=re.I)
        if len(cells) < 3 or not link:
            continue
        relative = link.group(1).replace("\\", "/")
        parsed = urlparse(urljoin(BASE_URL, relative))
        if parsed.netloc != urlparse(BASE_URL).netloc or not parsed.path.endswith("/ReportPage/ViewPdf"):
            continue
        query = parse_qs(parsed.query)
        raw_path = query.get("path", [""])[0]
        reports.append({
            "financial_year": cells[1], "label": cells[2], "id": query.get("id", [""])[0],
            "archive_path": raw_path, "url": f"{BASE_URL}/ReportPage/ViewPdf?id={quote(query.get('id', [''])[0])}&path={quote(raw_path)}",
        })
    return reports


def _report_month(financial_year: str, label: str) -> pd.Timestamp | None:
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)", label, re.I)
    if not match:
        return None
    month = datetime.strptime(match.group(1).title(), "%B").month
    start_year = int(financial_year[:4])
    year = start_year + 1 if month <= 3 else start_year
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def download_archive_reports(financial_year: str = "2024-25", labels: set[str] | None = None, force: bool = False) -> list[dict]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    selected = [r for r in discover_archive_reports() if r["financial_year"] == financial_year and (labels is None or r["label"] in labels)]
    manifest = []
    for report in selected:
        filename = Path(report["archive_path"].replace("\\", "/")).name
        target = ARCHIVE_DIR / filename
        try:
            if force or not target.exists():
                data = _fetch(report["url"])
                if not data.startswith(b"%PDF"):
                    raise ValueError(f"PAIMANA response for {report['label']} was not a PDF")
                target.write_bytes(data)
            record = {**report, "filename": filename, "status": "downloaded", "bytes": target.stat().st_size, "sha256": sha256(target.read_bytes()).hexdigest(), "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            record = {**report, "filename": filename, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}
        manifest.append(record)
    (ARCHIVE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def extract_report_text(pdf_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        completed = subprocess.run([pdftotext, "-layout", str(pdf_path), "-"], check=True, capture_output=True)
        return completed.stdout.decode("utf-8", errors="replace")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf or Poppler pdftotext to parse PAIMANA PDFs") from exc
    return "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)


def _number(value: str | None) -> float | None:
    if not value or value.strip() in {"-", "N.A."}:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _month_date(value: str | None) -> str | None:
    if not value or value.strip() in {"/", "-", "N.A."}:
        return None
    parsed = pd.to_datetime(value.strip(), format="%m/%Y", errors="coerce")
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def parse_project_list(text: str, report_month: pd.Timestamp, source_report: str, source_url: str) -> pd.DataFrame:
    """Parse the official June-2024-and-later table layout containing project codes."""
    marker = text.lower().find("project list: ongoing projects as of")
    if marker < 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    lines = text[marker:].splitlines()
    start_re = re.compile(r"^(?P<head>.*?)\s{2,}(?P<approval>\d{1,2}/\d{4})\s+(?P<completion>\d{1,2}/\d{4}|/)\s+(?P<cost>[\d,]+\.\d{2})\s+(?P<expenditure>[\d,]+\.\d{2})\s+(?P<progress>\d+(?:\.\d+)?)\s*$")
    starts = []
    for index, line in enumerate(lines):
        match = start_re.match(line)
        if not match:
            continue
        head = match.group("head")
        sequence = re.search(r"(?<!\S)(\d{1,4})\s+(.+)$", head)
        if sequence:
            starts.append((index, match, sequence))
    rows = []
    for position, (line_index, match, sequence) in enumerate(starts):
        block = lines[line_index + 1: starts[position + 1][0] if position + 1 < len(starts) else len(lines)]
        code = agency = None
        name_parts = [sequence.group(2).strip()]
        for line in block:
            code_match = re.fullmatch(r"\s*\(([A-Z]?\d{8})\s*\)\s*", line)
            if code_match:
                code = code_match.group(1)
                break
            agency_match = re.fullmatch(r"\s*\(([^{}]+?)\s*\)\s*", line)
            if agency_match and agency_match.group(1).strip() not in {"N.A.", "-"}:
                agency = agency_match.group(1).strip()
                continue
            project_fragment = line[20:58].strip() if len(line) > 20 else ""
            if project_fragment and not any(token in project_fragment for token in ["{", "(", "Table:", "of 302"]):
                name_parts.append(project_fragment)
        if not code:
            continue
        braces = re.findall(r"\{([^}]+)\}", "\n".join(block))
        parentheses = re.findall(r"\(([^)]+)\)", "\n".join(block))
        anticipated_completion = next((_month_date(x) for x in braces if _month_date(x)), None)
        anticipated_cost = next((_number(x) for x in braces if _number(x) is not None), None)
        revised_completion = next((_month_date(x) for x in parentheses if _month_date(x)), None)
        revised_cost = next((_number(x) for x in parentheses if _number(x) is not None), None)
        original_cost = _number(match.group("cost"))
        expenditure = _number(match.group("expenditure"))
        progress = _number(match.group("progress"))
        completion = _month_date(match.group("completion"))
        effective_cost = anticipated_cost or revised_cost or original_cost
        financial_progress = expenditure / effective_cost * 100 if expenditure is not None and effective_cost else None
        delay = None
        if completion and anticipated_completion:
            start, end = pd.Timestamp(completion), pd.Timestamp(anticipated_completion)
            delay = (end.year - start.year) * 12 + end.month - start.month
        rows.append({
            "project_id": code, "project_name": " ".join(name_parts).strip(), "sector": None, "ministry": None,
            "state": None, "implementing_agency": agency, "original_cost": original_cost, "revised_cost": revised_cost,
            "current_expenditure": expenditure, "planned_start_date": None, "planned_completion_date": completion,
            "revised_completion_date": revised_completion or anticipated_completion, "actual_completion_date": None,
            "month": report_month.strftime("%Y-%m-%d"), "physical_progress_percentage": progress,
            "financial_progress_percentage": None if financial_progress is None else round(financial_progress, 4),
            "milestone_status": None, "delay_months": delay, "source_report": source_report, "source_url": source_url,
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def build_monthly_history(manifest: list[dict] | None = None, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if manifest is None:
        manifest_path = ARCHIVE_DIR / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    frames = []
    for report in manifest:
        if report.get("status") == "failed":
            continue
        month = _report_month(report["financial_year"], report["label"])
        if month is None:
            continue
        pdf = ARCHIVE_DIR / report["filename"]
        frame = parse_project_list(extract_report_text(pdf), month, report["filename"], report["url"])
        if not frame.empty:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = result.drop_duplicates(["project_id", "month"], keep="last").sort_values(["project_id", "month"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def ingest_latest_archive() -> pd.DataFrame:
    labels = {"June", "October", "November", "December", "January", "February", "March"}
    manifest = download_archive_reports("2024-25", labels=labels)
    return build_monthly_history(manifest)
