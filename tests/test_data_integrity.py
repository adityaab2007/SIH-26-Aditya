from pathlib import Path
import json
import pandas as pd

from backend.app.services.paimana_ingestion_service import OUTPUT_COLUMNS, parse_project_list

ROOT=Path(__file__).resolve().parents[1]

def test_curated_data_is_real_project_coded_subset():
    df=pd.read_csv(ROOT/'data/raw/paimana_projects_may_2026.csv',dtype={'project_code':str})
    assert len(df)==96
    assert df.project_code.is_unique
    assert not df.project_code.str.match(r'^70000[1-8]$').any(), 'App-local high-value IDs must not enter official-code dataset'
    assert df.source_url.str.contains('mospi.gov.in').all()
    assert '701263' in set(df.project_code)
    assert '706775' in set(df.project_code)


def test_history_has_multiple_real_snapshots():
    df=pd.read_csv(ROOT/'data/raw/paimana_high_value_history.csv',dtype={'project_code':str})
    assert len(df)==14
    assert df.groupby('project_code').size().max() >= 3
    assert set(df.project_code).issubset(set(pd.read_csv(ROOT/'data/raw/paimana_projects_may_2026.csv',dtype={'project_code':str}).project_code))


def test_official_archive_manifest_and_processed_schema():
    manifest=json.loads((ROOT/'data/raw/paimana_archive/manifest.json').read_text())
    downloaded=[item for item in manifest if item['status']=='downloaded']
    assert len(downloaded) >= 3
    assert all(len(item['sha256']) == 64 for item in downloaded)
    df=pd.read_csv(ROOT/'data/processed/project_monthly_history.csv',dtype={'project_id':str})
    assert set(OUTPUT_COLUMNS) == set(df.columns)
    assert len(df) > 1000
    assert df.project_id.str.match(r'^[A-Z]?\d{8}$').all()
    assert (df.groupby('project_id').size() > 1).any()
    assert df.actual_completion_date.isna().all(), 'Unavailable actual completion must remain null'


def test_archive_parser_keeps_unavailable_fields_null():
    text='''Project List: Ongoing Projects as of 30th June 2024
STATE      POWER             1    SAMPLE PROJECT           1/2020           3/2024               500.00           250.00            50
                                      (Mar-25)             (550.00)
                                      {3/2025}             {575.00}
                                      (AGENCY )
                                      (N18000001 )
'''
    parsed=parse_project_list(text,pd.Timestamp('2024-06-30'),'fixture.pdf','https://paimana-proj.mospi.gov.in/report')
    assert len(parsed)==1
    assert parsed.iloc[0].project_id=='N18000001'
    assert pd.isna(parsed.iloc[0].actual_completion_date)
    assert pd.isna(parsed.iloc[0].planned_start_date)
