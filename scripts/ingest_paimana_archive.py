#!/usr/bin/env python3
import argparse
from backend.app.services.paimana_ingestion_service import build_monthly_history, ingest_latest_archive

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true", help="Parse already downloaded immutable PDFs without network access")
    args = parser.parse_args()
    frame = build_monthly_history() if args.local_only else ingest_latest_archive()
    print(f"Wrote {len(frame)} official PAIMANA monthly observations across {frame.project_id.nunique()} projects")
