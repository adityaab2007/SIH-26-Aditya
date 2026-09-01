"""Prepare Exp122's compact official Union Budget evidence tables.

This utility intentionally does not scrape or guess historical ministry rows.
Researchers export reviewed rows from official India Budget Expenditure Profile /
Demands for Grants documents into the documented CSV schemas, and this script
validates provenance, uniqueness, publication dates, mappings and checksums.

Budget CSV columns:
  fiscal_year_start,published_date,official_budget_ministry,capital_be_cr,
  total_be_cr,source_url,source_sha256

Mapping CSV columns:
  raw_ministry,official_budget_ministry,match_method,match_confidence,
  effective_from,effective_to,source_url
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external" / "exp122_union_budget"
MANIFEST = DEST / "source_manifest.json"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(budget_csv: Path, mapping_csv: Path, dest: Path = DEST) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("source_institution") != "Ministry of Finance, Government of India — Union Budget":
        raise ValueError("Exp122 refuses non-Union-Budget provenance")

    budget = pd.read_csv(budget_csv)
    mapping = pd.read_csv(mapping_csv)
    b_req = {"fiscal_year_start","published_date","official_budget_ministry","capital_be_cr","total_be_cr","source_url","source_sha256"}
    m_req = {"raw_ministry","official_budget_ministry","match_method","match_confidence","effective_from","effective_to","source_url"}
    if not b_req.issubset(budget.columns): raise ValueError(f"budget input missing columns: {sorted(b_req-set(budget.columns))}")
    if not m_req.issubset(mapping.columns): raise ValueError(f"mapping input missing columns: {sorted(m_req-set(mapping.columns))}")

    budget["fiscal_year_start"] = pd.to_numeric(budget.fiscal_year_start, errors="coerce").astype("Int64")
    budget["published_date"] = pd.to_datetime(budget.published_date, errors="coerce")
    budget["capital_be_cr"] = pd.to_numeric(budget.capital_be_cr, errors="coerce")
    budget["total_be_cr"] = pd.to_numeric(budget.total_be_cr, errors="coerce")
    budget["official_budget_ministry"] = budget.official_budget_ministry.astype(str).str.strip()
    if budget[["fiscal_year_start","published_date","official_budget_ministry","capital_be_cr","total_be_cr"]].isna().any().any():
        raise ValueError("invalid/missing reviewed Union Budget values")
    if budget.duplicated(["fiscal_year_start","official_budget_ministry"]).any():
        raise ValueError("duplicate fiscal-year/ministry rows")
    if not budget.source_url.astype(str).str.startswith(("https://www.indiabudget.gov.in/","https://indiabudget.gov.in/")).all():
        raise ValueError("every budget row must retain an official indiabudget.gov.in source URL")

    mapping["match_confidence"] = pd.to_numeric(mapping.match_confidence, errors="coerce")
    mapping["effective_from"] = pd.to_datetime(mapping.effective_from, errors="coerce")
    mapping["effective_to"] = pd.to_datetime(mapping.effective_to, errors="coerce")
    if mapping[["raw_ministry","official_budget_ministry","match_method","match_confidence","effective_from","effective_to"]].isna().any().any():
        raise ValueError("invalid/missing ministry mapping values")
    if mapping.match_method.astype(str).str.contains("fuzzy", case=False, na=False).any():
        raise ValueError("fuzzy ministry mapping is forbidden")
    if (mapping.match_confidence < 0.95).any():
        raise ValueError("reviewed mapping confidence must be >= 0.95")
    if not mapping.source_url.astype(str).str.startswith(("https://www.indiabudget.gov.in/","https://indiabudget.gov.in/")).all():
        raise ValueError("every mapping row must retain an official indiabudget.gov.in source URL")

    dest.mkdir(parents=True, exist_ok=True)
    budget_out = dest / "union_budget_ministry_year.csv"
    map_out = dest / "ministry_mapping.csv"
    budget.sort_values(["fiscal_year_start","official_budget_ministry"]).to_csv(budget_out, index=False, date_format="%Y-%m-%d")
    mapping.sort_values(["raw_ministry","effective_from"]).to_csv(map_out, index=False, date_format="%Y-%m-%d")
    evidence = {
        "budget_input_sha256": _sha(budget_csv),
        "mapping_input_sha256": _sha(mapping_csv),
        "budget_output_sha256": _sha(budget_out),
        "mapping_output_sha256": _sha(map_out),
        "budget_rows": int(len(budget)),
        "mapping_rows": int(len(mapping)),
        "coverage_start_fy": int(budget.fiscal_year_start.min()),
        "coverage_end_fy": int(budget.fiscal_year_start.max()),
        "source_files": budget[["source_url","source_sha256"]].drop_duplicates().to_dict("records"),
    }
    (dest / "prepared_data_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reviewed-budget-csv", required=True)
    p.add_argument("--reviewed-mapping-csv", required=True)
    args = p.parse_args()
    print(json.dumps(prepare(Path(args.reviewed_budget_csv), Path(args.reviewed_mapping_csv)), indent=2))


if __name__ == "__main__": main()
