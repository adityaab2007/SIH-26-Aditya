# Data Provenance

## Included source-aligned files

### `data/raw/paimana_projects_may_2026.csv`

A curated subset transcribed from the **official PAIMANA public dashboard** surfaced by MoSPI. Every included row has a surfaced project code and retains a `source_url`.

The build process originally identified several high-value records where official values were visible but the project code was not surfaced in the retrieved table. Those rows were removed rather than assigning invented PAIMANA identifiers.

### `data/raw/paimana_high_value_history.csv`

Official monthly values from PAIMANA/Flash Report surfaces for a smaller set of projects with stable project codes. It powers the historical replay UI.

## What is not claimed

The repository does **not** currently contain a clean 20-year monthly OCMS training table. The problem statement indicates that the underlying historical repository exists, but converting old OCMS/Flash Reports into a stable longitudinal machine-learning table is a separate ETL milestone.

## Data quality

Operational project records can contain:

- missing revised cost;
- missing revised completion date;
- missing physical progress;
- expenditure greater than listed revised cost;
- revised dates earlier than original dates.

InfraSight creates explicit data-quality flags instead of silently changing these values. This supports auditability and lets future models use missingness itself as a reliability signal where appropriate.
