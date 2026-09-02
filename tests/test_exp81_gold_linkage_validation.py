import pandas as pd
import pytest
from backend.app.ml.experiments.exp81_gold_linkage_validation import *

def _gold(labels,rule='r1'):
    return pd.DataFrame({'snapshot_id':[f's{i}' for i in range(len(labels))],'candidate_completion_id':[f'c{i}' for i in range(len(labels))],'candidate_rule':[rule]*len(labels),'label':labels,'reviewer':['human']*len(labels),'review_timestamp':['2026-09-01']*len(labels)})

def test_precision_and_threshold():
    m=rule_metrics(_gold(['MATCH']*20),min_support=20,precision_threshold=.98).iloc[0]
    assert m.precision==1 and m.status=='APPROVED FOR AUTO-LINKING'
    assert rule_metrics(_gold(['MATCH']*19+['NON_MATCH']),20,.98).iloc[0].status=='REJECTED'

def test_insufficient_evidence_and_no_unaudited_approval():
    assert rule_metrics(_gold(['MATCH']*5),20,.98).iloc[0].status=='INSUFFICIENT EVIDENCE'
    bad=_gold(['MATCH']); bad['reviewer']=''
    with pytest.raises(ValueError): approved_rules(bad,min_support=1)

def test_candidate_export_is_unreviewed_not_gold():
    out=generate_audit_candidates(pd.DataFrame([{'snapshot_id':'s','candidate_completion_id':'c','candidate_rule':'r'}]))
    assert out.iloc[0].label=='UNSURE' and out.iloc[0].reviewer==''

def test_synthetic_fixture_cannot_be_real_gold(tmp_path):
    p=tmp_path/'gold.csv'; x=_gold(['MATCH']); x['synthetic_test_data']=True; x.to_csv(p,index=False)
    with pytest.raises(ValueError): load_real_gold(p)
