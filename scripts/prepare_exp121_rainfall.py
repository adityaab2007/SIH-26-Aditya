"""Normalize the official OGD/IMD subdivision rainfall CSV for Exp121.

Expected source layout is the published annual wide resource with SUBDIVISION,
YEAR and JAN..DEC rainfall columns. No future climatological normal is invented;
normal_mm is left missing unless the reviewed input provides explicit monthly
normal columns named JAN_NORMAL..DEC_NORMAL.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external" / "exp121_rainfall"
MANIFEST = DEST / "source_manifest.json"
MONTHS = [("JAN",1),("FEB",2),("MAR",3),("APR",4),("MAY",5),("JUN",6),("JUL",7),("AUG",8),("SEP",9),("OCT",10),("NOV",11),("DEC",12)]


def _sha(path: Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def prepare(source: Path, output: Path) -> dict:
    manifest=json.loads(MANIFEST.read_text())
    if not str(manifest.get("source_institution", "")).startswith("India Meteorological Department"):
        raise ValueError("Exp121 refuses non-IMD provenance")
    raw=pd.read_csv(source); raw.columns=[str(c).strip().upper() for c in raw.columns]
    required={"SUBDIVISION","YEAR",*(m for m,_ in MONTHS)}
    missing=required-set(raw.columns)
    if missing: raise ValueError(f"official rainfall input missing columns: {sorted(missing)}")
    records=[]
    for _,row in raw.iterrows():
        year=pd.to_numeric(pd.Series([row.YEAR]),errors="coerce").iloc[0]
        if pd.isna(year): continue
        for name,month in MONTHS:
            rainfall=pd.to_numeric(pd.Series([row.get(name)]),errors="coerce").iloc[0]
            if pd.isna(rainfall): continue
            normal_col=f"{name}_NORMAL"; normal=pd.to_numeric(pd.Series([row.get(normal_col)]),errors="coerce").iloc[0] if normal_col in raw.columns else float("nan")
            records.append({"period":pd.Timestamp(year=int(year),month=month,day=1)+pd.offsets.MonthEnd(0),"geography_type":"subdivision","geography_name":str(row.SUBDIVISION).strip(),"rainfall_mm":float(rainfall),"normal_mm":float(normal) if pd.notna(normal) else None})
    out=pd.DataFrame(records).sort_values(["geography_name","period"])
    if out.duplicated(["period","geography_type","geography_name"]).any(): raise ValueError("duplicate subdivision/month observations")
    output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(output,index=False,date_format="%Y-%m-%d")
    evidence={"source_file":str(source),"source_sha256":_sha(source),"output_file":str(output),"output_sha256":_sha(output),"rows":int(len(out)),"coverage_start":str(out.period.min().date()) if len(out) else None,"coverage_end":str(out.period.max().date()) if len(out) else None,"geographies":int(out.geography_name.nunique()) if len(out) else 0,"normal_values_supplied_by_source":bool(out.normal_mm.notna().any()) if len(out) else False}
    (output.parent/"prepared_data_manifest.json").write_text(json.dumps(evidence,indent=2)+"\n"); return evidence


def main():
    p=argparse.ArgumentParser(); p.add_argument("--official-ogd-csv",required=True); p.add_argument("--output",default=str(DEST/"rainfall_monthly.csv")); a=p.parse_args(); print(json.dumps(prepare(Path(a.official_ogd_csv),Path(a.output)),indent=2))

if __name__=="__main__": main()
