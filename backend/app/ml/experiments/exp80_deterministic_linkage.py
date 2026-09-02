from __future__ import annotations
import re, unicodedata
from collections import Counter
import numpy as np
import pandas as pd

EXPERIMENT_ID='exp_80'; EXPERIMENT_SEQUENCE=80
OUTCOME_FIELDS={'actual_cost_overrun_percentage','actual_delay_days','completion_date','reported_completion_expenditure_cr'}

def _text(v):
    if v is None or pd.isna(v): return ''
    s=unicodedata.normalize('NFKC',str(v)).lower().strip()
    s=re.sub(r'[^a-z0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()

def normalize_code(v): return re.sub(r'[^a-z0-9]','',_text(v)).upper()
def normalize_title(v):
    s=_text(v)
    repl={' rd ':' road ',' nh ':' national highway ',' rly ':' railway ',' stn ':' station '}
    s=' '+s+' '
    for a,b in repl.items(): s=s.replace(a,b)
    return re.sub(r'\s+',' ',s).strip()
def normalize_state(v):
    s=_text(v); aliases={'orissa':'odisha','uttaranchal':'uttarakhand','nct of delhi':'delhi'}
    return aliases.get(s,s)
def normalize_sector(v): return _text(v).replace('&','and')
def normalize_agency(v):
    s=_text(v); aliases={'nhai':'national highways authority of india','ircon':'ircon international limited'}
    return aliases.get(s,s)
def parse_cost(v): return pd.to_numeric(pd.Series([v]),errors='coerce').iloc[0]
def parse_date(v): return pd.to_datetime(v,errors='coerce')
def cost_compatible(a,b,rel_tol=.01,abs_tol=.05):
    a,b=parse_cost(a),parse_cost(b)
    return bool(pd.notna(a) and pd.notna(b) and abs(float(a)-float(b)) <= max(abs_tol,rel_tol*max(abs(float(a)),abs(float(b)),1.0)))
def date_compatible(a,b,tolerance_days=31):
    a,b=parse_date(a),parse_date(b)
    return bool(pd.notna(a) and pd.notna(b) and abs((a-b).days)<=tolerance_days)

def _prep(df):
    x=df.copy()
    x['_code']=x.get('project_id',x.get('project_code',pd.Series(index=x.index,dtype=object))).map(normalize_code)
    x['_title']=x.get('project_name',x.get('title',pd.Series(index=x.index,dtype=object))).map(normalize_title)
    x['_state']=x.get('state',pd.Series('',index=x.index)).map(normalize_state)
    x['_sector']=x.get('sector',pd.Series('',index=x.index)).map(normalize_sector)
    x['_agency']=x.get('implementing_agency',x.get('agency',pd.Series('',index=x.index))).map(normalize_agency)
    return x

def resolve_links(snapshots,outcomes):
    s,o=_prep(snapshots),_prep(outcomes); accepted=[]; diagnostics=[]
    for si,row in s.iterrows():
        candidates=[]
        if row['_code']:
            c=o[o._code.eq(row['_code'])]
            if len(c): candidates=[(1,i,'exact_project_code') for i in c.index]
        if not candidates and row['_title']:
            c=o[o._title.eq(row['_title']) & o._state.eq(row['_state']) & o._sector.eq(row['_sector'])]
            c=c[[cost_compatible(row.get('approved_cost_cr'),r.get('approved_cost_cr')) for _,r in c.iterrows()]] if len(c) else c
            if len(c): candidates=[(2,i,'title_state_sector_cost') for i in c.index]
        if not candidates and row['_title']:
            c=o[o._title.eq(row['_title']) & o._state.eq(row['_state']) & o._sector.eq(row['_sector']) & o._agency.eq(row['_agency'])]
            ok=[]
            for i,r in c.iterrows():
                if cost_compatible(row.get('approved_cost_cr'),r.get('approved_cost_cr')) and date_compatible(row.get('planned_completion_date',row.get('planned_commissioning_date')),r.get('planned_commissioning_date',r.get('planned_completion_date'))): ok.append(i)
            if ok: candidates=[(3,i,'agency_state_sector_cost_date_title') for i in ok]
        uniq={i:(tier,rule) for tier,i,rule in candidates}
        if len(uniq)==1:
            oi,(tier,rule)=next(iter(uniq.items())); orow=o.loc[oi]
            accepted.append({'snapshot_id':row.get('snapshot_id',si),'completion_id':orow.get('completion_id',orow.get('project_id',oi)),'rule':rule,'tier':tier,'matched_fields':['project_code'] if tier==1 else ['project_title','state','sector','approved_cost']+(['agency','planned_commissioning_date'] if tier==3 else []),'confidence_category':'VERY_HIGH' if tier==1 else 'HIGH','ambiguity_count':1})
        else:
            diagnostics.append({'snapshot_id':row.get('snapshot_id',si),'status':'ambiguous' if len(uniq)>1 else 'unmatched','ambiguity_count':len(uniq)})
    return pd.DataFrame(accepted),pd.DataFrame(diagnostics)

def linkage_diagnostics(original_links,new_links,diagnostics):
    return {'original_matched_projects':int(original_links.completion_id.nunique()) if len(original_links) else 0,'original_matched_snapshots':int(len(original_links)),'newly_matched_projects':int(len(set(new_links.completion_id)-set(original_links.completion_id))) if len(new_links) else 0,'newly_matched_snapshots':int(max(0,len(new_links)-len(original_links))),'total_matched_projects_after_expansion':int(new_links.completion_id.nunique()) if len(new_links) else 0,'total_matched_snapshots_after_expansion':int(len(new_links)),'ambiguous_snapshots':int((diagnostics.status=='ambiguous').sum()) if len(diagnostics) else 0,'unmatched_snapshots':int((diagnostics.status=='unmatched').sum()) if len(diagnostics) else 0,'matches_by_rule':dict(Counter(new_links.rule)) if len(new_links) else {},'matches_by_tier':{str(k):int(v) for k,v in Counter(new_links.tier).items()} if len(new_links) else {},'uniqueness_rate':float((new_links.ambiguity_count.eq(1)).mean()) if len(new_links) else 0.0}

def assert_no_outcome_leakage(columns):
    bad=OUTCOME_FIELDS.intersection(columns)
    if bad: raise ValueError('Outcome leakage in identity matching: '+','.join(sorted(bad)))
