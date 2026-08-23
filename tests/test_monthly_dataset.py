import gzip
import json
from pathlib import Path

import pandas as pd

from backend.app.ml.monthly_lifecycle import load_monthly_snapshots


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data" / "processed" / "paimana_monthly_snapshots.csv"


def test_committed_monthly_snapshots_are_official_and_longitudinal():
    frame = load_monthly_snapshots()
    audit = json.loads((ROOT / "data/processed/paimana_monthly_ingestion_audit.json").read_text())

    assert not frame.empty
    assert len(frame) == audit["monthly_observations"]
    assert frame["project_id"].nunique() > 1000
    assert frame["project_id"].duplicated().any()
    dates = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    assert dates.notna().all()
    assert dates.min().year == 2001
    assert dates.max().year == 2025
    assert frame["source_report"].notna().all()
    assert frame["source_url"].fillna("").str.startswith("https://paimana-proj.mospi.gov.in/").all()
    assert not frame.astype(str).apply(lambda column: column.str.contains("synthetic|placeholder|fake", case=False, regex=True).any()).any()

    completion = pd.to_datetime(frame["actual_completion_date"], errors="coerce")
    assert not ((dates >= completion) & completion.notna()).any()


def test_monthly_snapshot_loader_reads_gzip_directly(tmp_path):
    source = pd.DataFrame({"project_id": ["P1"], "snapshot_date": ["2024-01-31"]})
    compressed = tmp_path / "paimana_monthly_snapshots.csv.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        source.to_csv(handle, index=False)

    loaded = load_monthly_snapshots(compressed)
    assert loaded.to_dict("records") == source.to_dict("records")
