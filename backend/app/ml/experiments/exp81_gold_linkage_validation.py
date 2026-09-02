from __future__ import annotations
from pathlib import Path
import pandas as pd

EXPERIMENT_ID='exp_81'; EXPERIMENT_SEQUENCE=81
ALLOWED_LABELS={'MATCH','NON_MATCH','UNSURE'}
REQUIRED=['snapshot_id','candidate_completion_id','candidate_rule','label','reviewer','review_timestamp']

def validate_gold_labels(frame: pd.DataFrame) -> pd.DataFrame:
    x=frame.copy()
    missing=[c for c in REQUIRED if c not in x]
    if missing: raise ValueError('Missing gold columns: '+','.join(missing))
    x['label']=x.label.astype(str).str.strip().str.upper()
    bad=set(x.label)-ALLOWED_LABELS
    if bad: raise ValueError('Invalid labels: '+','.join(sorted(bad)))
    if x.reviewer.fillna('').astype(str).str.strip().eq('').any(): raise ValueError('Human reviewer is required; generated matches are not gold labels.')
    if pd.to_datetime(x.review_timestamp,errors='coerce').isna().any(): raise ValueError('Valid review_timestamp required.')
    return x

def rule_metrics(gold: pd.DataFrame, min_support=20, precision_threshold=.98) -> pd.DataFrame:
    x=validate_gold_labels(gold)
    rows=[]
    for rule,g in x.groupby('candidate_rule',sort=True):
        decided=g[g.label.ne('UNSURE')]
        tp=int((decided.label=='MATCH').sum()); fp=int((decided.label=='NON_MATCH').sum())
        support=tp+fp; precision=tp/support if support else None
        if support<min_support: status='INSUFFICIENT EVIDENCE'
        elif precision is not None and precision>=precision_threshold: status='APPROVED FOR AUTO-LINKING'
        else: status='REJECTED'
        rows.append({'rule':rule,'true_positives':tp,'false_positives':fp,'audited_support':support,'precision':precision,'recall':None,'status':status})
    return pd.DataFrame(rows)

def approved_rules(gold,**kwargs):
    m=rule_metrics(gold,**kwargs)
    return set(m.loc[m.status.eq('APPROVED FOR AUTO-LINKING'),'rule'])

def generate_audit_candidates(pairs: pd.DataFrame) -> pd.DataFrame:
    cols=['snapshot_id','snapshot_project_code','snapshot_title','snapshot_state','snapshot_sector','snapshot_agency','snapshot_approved_cost','snapshot_planned_commissioning_date','candidate_completion_id','candidate_project_code','candidate_title','candidate_state','candidate_sector','candidate_agency','candidate_approved_cost','candidate_planned_commissioning_date','candidate_rule']
    out=pairs.reindex(columns=cols).copy()
    out['label']='UNSURE'; out['reviewer']=''; out['review_timestamp']=''; out['review_note']=''
    return out

def load_real_gold(path: str|Path) -> pd.DataFrame:
    path=Path(path)
    if not path.exists(): raise FileNotFoundError('No real manually audited linkage dataset exists at '+str(path))
    frame=pd.read_csv(path)
    if 'synthetic_test_data' in frame.columns and frame.synthetic_test_data.fillna(False).astype(bool).any(): raise ValueError('Synthetic fixture cannot be used as human gold evidence.')
    return validate_gold_labels(frame)
