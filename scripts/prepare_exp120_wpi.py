"""Prepare the compact Exp120 WPI table from reviewed official OEA exports.

The Office of the Economic Adviser publishes monthly .xls index files. Because
sheet layouts can change, this utility deliberately does not guess a row/series
mapping. A researcher exports the reviewed official rows to the documented
long-form CSV schema, then this script validates provenance, uniqueness and
base-year compatibility before producing wpi_monthly.csv.

Input CSV columns:
  date, series, index_value, base_year, source_url, source_sha256
Allowed series:
  all_commodities, manufactured_products, fuel_power, steel, cement
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external" / "exp120_wpi"
MANIFEST = DEST / "source_manifest.json"
ALLOWED = {"all_commodities", "manufactured_products", "fuel_power", "steel", "cement"}


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(source: Path, output: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("source_institution") != "Office of the Economic Adviser, DPIIT, Government of India":
        raise ValueError("Exp120 refuses non-OEA provenance")
    raw = pd.read_csv(source)
    required = {"date", "series", "index_value", "base_year", "source_url", "source_sha256"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"input missing columns: {sorted(missing)}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    raw["series"] = raw["series"].astype(str).str.strip()
    raw["index_value"] = pd.to_numeric(raw["index_value"], errors="coerce")
    if raw[["date", "series", "index_value"]].isna().any().any():
        raise ValueError("invalid date/series/index value in reviewed OEA extract")
    if not set(raw.series).issubset(ALLOWED):
        raise ValueError(f"unapproved series names: {sorted(set(raw.series) - ALLOWED)}")
    if raw.duplicated(["date", "series"]).any():
        raise ValueError("duplicate month/series observations")
    if not raw.source_url.astype(str).str.startswith("https://eaindustry.nic.in/").all():
        raise ValueError("every observation must retain an official eaindustry.nic.in source URL")
    # Never silently concatenate index levels from incompatible base years.
    base_counts = raw.groupby("series")["base_year"].nunique()
    if (base_counts > 1).any():
        mixed = base_counts[base_counts > 1].index.tolist()
        raise ValueError(
            "mixed base years require an explicit, separately reviewed official linking transformation before this step: "
            + ", ".join(mixed)
        )
    wide = raw.pivot(index="date", columns="series", values="index_value").reset_index()
    for col in ALLOWED:
        if col not in wide:
            wide[col] = pd.NA
    wide = wide[["date", "all_commodities", "manufactured_products", "fuel_power", "steel", "cement"]].sort_values("date")
    output.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output, index=False, date_format="%Y-%m-%d")
    evidence = {
        "input_file": str(source),
        "input_sha256": _file_sha256(source),
        "output_file": str(output),
        "output_sha256": _file_sha256(output),
        "rows": int(len(wide)),
        "coverage_start": str(wide.date.min().date()) if len(wide) else None,
        "coverage_end": str(wide.date.max().date()) if len(wide) else None,
        "series": sorted(raw.series.unique().tolist()),
        "base_year_by_series": {str(k): str(v.iloc[0]) for k, v in raw.groupby("series")["base_year"]},
        "source_files": raw[["source_url", "source_sha256"]].drop_duplicates().to_dict("records"),
    }
    (output.parent / "prepared_data_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reviewed-long-csv", required=True)
    p.add_argument("--output", default=str(DEST / "wpi_monthly.csv"))
    args = p.parse_args()
    print(json.dumps(prepare(Path(args.reviewed_long_csv), Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
