from __future__ import annotations
import re
import pandas as pd

EXPERIMENT_ID='exp_82'; EXPERIMENT_SEQUENCE=82
TEXT_COLUMN_PATTERNS=('remark','reason','bottleneck','issue','status','narrative','comment','observation')
CATEGORIES={
'land_acquisition':[r'\bland acquisition\b',r'\bland.*acquir'],
'forest_clearance':[r'\bforest clearance\b',r'\bforest.*clear'],
'environmental_clearance':[r'\benvironment(?:al)? clearance\b',r'\benvironment.*clear'],
'funding':[r'\bfund(?:ing|s)? constraint',r'\bshortage of funds\b',r'\bfunds? not available\b'],
'litigation':[r'\blitigation\b',r'\bcourt case\b',r'\bstay order\b'],
'contractor':[r'\bcontractor\b.*(?:delay|issue|problem|slow|default)',r'\bpoor contractor performance\b'],
'utility_shifting':[r'\butility shifting\b',r'\bshifting of utilities\b'],
'railway_crossing':[r'\brail(?:way)? crossing\b',r'\brailway approval\b'],
'dpr':[r'\bdpr\b',r'\bdetailed project report\b'],
'approval_sanction':[r'\bapproval delay\b',r'\bsanction delay\b',r'\bawaiting approval\b'],
'tender_procurement':[r'\btender\b',r'\bprocurement\b'],
'law_order':[r'\blaw and order\b'],
'public_obstruction':[r'\blocal obstruction\b',r'\bpublic protest\b'],
'design':[r'\bdesign (?:issue|change|delay)\b'],
'material_shortage':[r'\bmaterial shortage\b'],
'manpower_shortage':[r'\bmanpower shortage\b',r'\blabou?r shortage\b'],
'geological_site':[r'\bgeolog(?:y|ical)\b',r'\bsite condition\b']}
NEGATION=re.compile(r'\b(?:no|not|without|resolved|cleared)\b.{0,25}$')

def discover_text_columns(columns):
    return [c for c in columns if any(p in str(c).lower() for p in TEXT_COLUMN_PATTERNS)]
def normalize_text(v):
    if v is None or pd.isna(v): return ''
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9]+',' ',str(v).lower())).strip()
def category_active(text,patterns):
    t=normalize_text(text)
    for pattern in patterns:
        m=re.search(pattern,t)
        if m and not NEGATION.search(t[max(0,m.start()-30):m.start()]): return 1
    return 0

def add_bottleneck_features(frame,project_col='canonical_project_id',date_col='snapshot_date',text_columns=None):
    x=frame.copy(); x[date_col]=pd.to_datetime(x[date_col],errors='coerce')
    cols=text_columns or discover_text_columns(x.columns)
    if not cols: raise ValueError('No PAIMANA narrative/delay-reason columns found in actual snapshot schema.')
    x['_issue_text']=x[cols].fillna('').astype(str).agg(' | '.join,axis=1).map(normalize_text)
    x=x.sort_values([project_col,date_col],kind='stable').copy()
    for cat,patterns in CATEGORIES.items():
        active='issue_'+cat+'_active'; seen='issue_'+cat+'_seen_before'; months='issue_'+cat+'_months_active'; since='issue_'+cat+'_months_since_first_seen'
        x[active]=x['_issue_text'].map(lambda t: category_active(t,patterns))
        x[seen]=x.groupby(project_col)[active].cummax()
        x[months]=x.groupby(project_col)[active].cumsum()
        first=x[date_col].where(x[active].eq(1)).groupby(x[project_col]).transform('min')
        x[since]=((x[date_col]-first).dt.days/30.4375).where(first.notna())
    active_cols=[c for c in x if c.endswith('_active')]
    seen_cols=[c for c in x if c.endswith('_seen_before')]
    x['issue_text_present']=x['_issue_text'].ne('').astype(int)
    x['issue_count_active']=x[active_cols].sum(axis=1)
    x['issue_count_seen']=x[seen_cols].sum(axis=1)
    x['number_of_distinct_bottleneck_categories']=x['issue_count_seen']
    x['repeated_bottleneck_flag']=(x[[c for c in x if c.endswith('_months_active')]].max(axis=1)>=2).astype(int)
    first_any=x[date_col].where(x.issue_count_active.gt(0)).groupby(x[project_col]).transform('min')
    x['months_since_first_bottleneck']=((x[date_col]-first_any).dt.days/30.4375).where(first_any.notna())
    return x.drop(columns=['_issue_text']),cols

def assert_asof_only(frame,project_col='canonical_project_id',date_col='snapshot_date'):
    ordered=frame.sort_values([project_col,date_col])
    if ordered.groupby(project_col)[date_col].apply(lambda s: not s.is_monotonic_increasing).any(): raise ValueError('Temporal ordering failure')
    return True
