import pandas as pd
import pytest
from backend.app.ml.experiments.exp80_deterministic_linkage import *

def test_normalization_and_tolerances():
    assert normalize_code(' ab-12 / 34 ')=='AB1234'
    assert normalize_title('NH- Road  Project')=='national highway road project'
    assert normalize_state('Orissa')=='odisha'
    assert cost_compatible(100,100.5)
    assert date_compatible('2020-01-01','2020-01-20')

def test_unique_exact_code_and_ambiguity_rejection():
    s=pd.DataFrame([{'snapshot_id':'s1','project_id':'A-1','project_name':'Alpha','approved_cost_cr':100,'state':'Delhi','sector':'Road'}])
    o=pd.DataFrame([{'completion_id':'c1','project_id':'A1','project_name':'Alpha','approved_cost_cr':100,'state':'Delhi','sector':'Road'}])
    links,diag=resolve_links(s,o)
    assert len(links)==1 and links.iloc[0].tier==1 and diag.empty
    o2=pd.concat([o,o.assign(completion_id='c2')],ignore_index=True)
    links,diag=resolve_links(s,o2)
    assert links.empty and diag.iloc[0].status=='ambiguous'

def test_tier2_and_audit_metadata_reproducible():
    s=pd.DataFrame([{'snapshot_id':'s','project_id':'','project_name':'Alpha Project','approved_cost_cr':100,'state':'Orissa','sector':'Power'}])
    o=pd.DataFrame([{'completion_id':'c','project_id':'','project_name':'alpha project','approved_cost_cr':100.02,'state':'Odisha','sector':'Power'}])
    a,d=resolve_links(s,o); b,_=resolve_links(s,o)
    assert a.to_dict('records')==b.to_dict('records')
    assert a.iloc[0].rule=='title_state_sector_cost' and a.iloc[0].ambiguity_count==1

def test_no_outcome_leakage_guard():
    assert_no_outcome_leakage(['project_id','state','approved_cost_cr'])
    with pytest.raises(ValueError): assert_no_outcome_leakage(['project_id','actual_delay_days'])
