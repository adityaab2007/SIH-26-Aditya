import pandas as pd
from backend.app.ml.experiments.exp82_paimana_bottlenecks import *

def test_discovery_normalization_and_detection():
    assert discover_text_columns(['project_id','latest_remarks','reason_for_delay'])==['latest_remarks','reason_for_delay']
    assert normalize_text(' Land-Acquisition!! ')=='land acquisition'
    assert category_active('land acquisition pending',CATEGORIES['land_acquisition'])==1
    assert category_active('no land acquisition issue',CATEGORIES['land_acquisition'])==0

def test_asof_persistence_and_first_seen():
    f=pd.DataFrame({'canonical_project_id':['p','p','p'],'snapshot_date':['2020-01-01','2020-02-01','2020-03-01'],'remarks':['','Land acquisition pending','Land acquisition still pending']})
    out,cols=add_bottleneck_features(f,text_columns=['remarks'])
    assert cols==['remarks']
    assert out.issue_land_acquisition_active.tolist()==[0,1,1]
    assert out.issue_land_acquisition_seen_before.tolist()==[0,1,1]
    assert out.issue_land_acquisition_months_active.tolist()==[0,1,2]
    assert out.repeated_bottleneck_flag.tolist()==[0,0,1]
    assert pd.isna(out.months_since_first_bottleneck.iloc[0]) and out.months_since_first_bottleneck.iloc[2]>0

def test_future_text_does_not_change_earlier_features():
    early=pd.DataFrame({'canonical_project_id':['p'],'snapshot_date':['2020-01-01'],'remarks':['no issue']})
    full=pd.concat([early,pd.DataFrame({'canonical_project_id':['p'],'snapshot_date':['2020-02-01'],'remarks':['court case pending']})],ignore_index=True)
    a,_=add_bottleneck_features(early,text_columns=['remarks']); b,_=add_bottleneck_features(full,text_columns=['remarks'])
    cols=[c for c in a.columns if c.startswith('issue_') or 'bottleneck' in c]
    assert a.iloc[0][cols].to_dict()==b.iloc[0][cols].to_dict()
