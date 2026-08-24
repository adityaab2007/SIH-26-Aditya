# PAIMANA Archive Data Pipeline

## Official source

The ingestion service reads the MoSPI PAIMANA Project Monitoring archive at `https://paimana-proj.mospi.gov.in/ReportPage/ArchiveProjectMonitoring`. The live archive exposes monthly Flash Reports from financial year 2001–02 through 2024–25.

## Extraction

`backend/app/services/paimana_ingestion_service.py` now:

1. retrieves all 291 entries in the live archive index across 2001–02 through 2024–25;
2. accepts only PDF links on the PAIMANA host and `/ReportPage/ViewPdf` path;
3. saves original PDFs unchanged under `data/raw/paimana_archive/`;
4. records source URL, byte size, UTC retrieval time and SHA-256 in `manifest.json`;
5. records individual broken report links without discarding successful downloads;
6. extracts layout-preserving text using Poppler `pdftotext`, with `pypdf` as a fallback;
7. selects a parser from detected document structure (`legacy-sector-v1`, `classic-code-v3`, `redesigned-code-v1`, or `recent-project-list-v3`);
8. caches parsed rows by source SHA-256 plus parser version;
9. combines multi-part reports by exact project ID and logs conflicts; and
10. writes `data/processed/paimana_monthly_snapshots.csv`.

Run a live refresh with `python scripts/build_monthly_ml_pipeline.py`. Use `--local-only` to reproduce processed outputs from cached immutable PDFs. Discovery alone is available through `--discover-only`.

## Data refresh versus model retraining

The archive refresh is an occasional, explicit operation: it may discover/cache the official PAIMANA PDFs, parse them, resolve identities and rebuild `data/processed/paimana_monthly_snapshots.csv`. The normal website flow does not run that operation. `POST /api/models/retrain` loads the already-committed official processed snapshot dataset and trains the selected monthly lifecycle window. It never downloads or parses the 291-report archive. If the processed dataset is unavailable, the lifecycle catalog reports an explicit unavailable state instead of showing legacy completed-project counts as lifecycle data.

## Normalized schema

The processed dataset contains project ID/name, sector, ministry, state, implementing agency, original/revised cost, expenditure, planned/revised/actual dates, snapshot month, physical/financial progress, milestone status, delay months and source provenance.

Fields absent from a report remain null. In the currently parsed archive layouts, planned start date, actual completion date, ministry, milestone status and some state/sector cells are not reliably exposed at project-row level. They are deliberately not inferred or fabricated. Financial progress is the transparent ratio of cumulative expenditure to the latest available anticipated/revised/original cost.

## Archive coverage

The official index contains 291 report entries representing 286 financial-year/month combinations. April 2005–06 and January 2008–09 are known catalog gaps. Multi-part 2024–25 reports are distinct source files for the same snapshot month. Download and parser outcomes are recorded per report; no failure is silently converted into a missing month.

## Forecast-training boundary

Monthly observations enter supervised training only after exact official-code linkage or unique exact-name plus exact approved-cost verification against the official completed-project outcome archive. Ambiguous/unverified rows remain useful for trajectory inspection but are excluded from training. Production monthly training never reads `data/project_history.csv`.

Each training row means “official project state at snapshot T → eventual official outcome.” Features use only that snapshot and earlier snapshots. Historical priors use only projects completed before T. Snapshots are sampled quarterly and weighted so each project contributes approximately equal total weight. When an explicit planned start is absent, lifecycle-stage ratios use the separately preserved official approval date as a documented proxy; the source is recorded as `official_approval_date_proxy` and is never presented as an observed construction start.

## Completed-project historical simulation

`scripts/ingest_paimana_completed_reports.py` extracts annual cumulative completed-project tables from historical March reports and each monthly completion list in the newer 2024–25 layout into `data/processed/paimana_completed_outcomes.csv`. Repeated cumulative entries are collapsed by exact official project code, retaining the latest official record; code-less legacy rows require exact audited linkage. The real simulation pipeline uses approved cost, sector, implementing agency when published, and original commissioning month as inputs. Reported cumulative expenditure at completion and the reported completion month are held out as targets, never features.

This source supports the 2001–2015 and 2015–2021 training windows. Evaluation is limited to completion years for which the archive has recorded outcomes; unrecorded future years are surfaced as forecast-only rather than given fabricated actual values.
