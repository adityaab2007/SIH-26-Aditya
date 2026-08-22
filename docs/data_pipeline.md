# PAIMANA Archive Data Pipeline

## Official source

The ingestion service reads the MoSPI PAIMANA Project Monitoring archive at `https://paimana-proj.mospi.gov.in/ReportPage/ArchiveProjectMonitoring`. The live archive exposes monthly Flash Reports from financial year 2001–02 through 2024–25.

## Extraction

`backend/app/services/paimana_ingestion_service.py`:

1. retrieves the archive's official report index;
2. accepts only PDF links on the PAIMANA host and `/ReportPage/ViewPdf` path;
3. saves original PDFs unchanged under `data/raw/paimana_archive/`;
4. records source URL, byte size, UTC retrieval time and SHA-256 in `manifest.json`;
5. records individual broken report links without discarding successful downloads;
6. extracts layout-preserving text using Poppler `pdftotext`, with `pypdf` as a fallback;
7. parses report tables that expose official project codes; and
8. writes `data/processed/project_monthly_history.csv`.

Run a live refresh with `python scripts/ingest_paimana_archive.py`. Use `--local-only` to reproduce the processed CSV from checked-in raw reports without network access.

## Normalized schema

The processed dataset contains project ID/name, sector, ministry, state, implementing agency, original/revised cost, expenditure, planned/revised/actual dates, snapshot month, physical/financial progress, milestone status, delay months and source provenance.

Fields absent from a report remain null. In the currently parsed archive layouts, planned start date, actual completion date, ministry, milestone status and some state/sector cells are not reliably exposed at project-row level. They are deliberately not inferred or fabricated. Financial progress is the transparent ratio of cumulative expenditure to the latest available anticipated/revised/original cost.

## Current checked-in extraction

The reproducible snapshot includes six original 2024–25 reports. Three supported project-code table layouts currently yield 4,692 observations across 1,844 project codes, with 1,490 projects observed in more than one month. The official February 2025 link returned HTTP 500 during retrieval and is retained as a failed manifest entry.

## Forecast-training boundary

Official ongoing-project reports provide longitudinal monitoring inputs but do not consistently publish final actual cost and actual completion for each project. Consequently they cannot safely supply supervised final-outcome labels without a governed completed-project export. The operational ingestion is real; the bundled temporal model remains explicitly trained on deterministic synthetic completion trajectories solely to demonstrate the leakage-safe training/API/UI contract. Replacing those trajectories with an authorized completed-project PAIMANA/OCMS table requires no architecture change.
