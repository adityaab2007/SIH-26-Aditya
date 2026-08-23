#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.paimana_ingestion_service import (
    archive_coverage,
    build_monthly_history,
    discover_archive_reports,
    download_archive_reports,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true", help="Parse already downloaded immutable PDFs without network access")
    parser.add_argument("--discover-only", action="store_true", help="List official archive coverage without downloading PDFs")
    parser.add_argument("--from-year", type=int, default=2001)
    parser.add_argument("--to-year", type=int, default=2024)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-parse", action="store_true")
    args = parser.parse_args()
    if args.discover_only:
        print(__import__("json").dumps(archive_coverage(discover_archive_reports()), indent=2))
        raise SystemExit(0)
    manifest = None if args.local_only else download_archive_reports(
        from_year=args.from_year, to_year=args.to_year, force=args.force_download,
    )
    frame = build_monthly_history(manifest, force_parse=args.force_parse)
    print(f"Wrote {len(frame)} official PAIMANA monthly observations across {frame.project_id.nunique()} reported project codes")
