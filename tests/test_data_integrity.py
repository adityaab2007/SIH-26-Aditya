from pathlib import Path
import pandas as pd

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
