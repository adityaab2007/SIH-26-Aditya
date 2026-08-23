# Lifecycle feature audit

## Legacy five-feature baseline

The existing completed-outcome model remains unchanged and identifiable as the controlled five-feature baseline.

The monthly lifecycle model uses `paimana_monthly_snapshots.csv` and links labels by exact official project ID. Where early reports lack IDs, only a unique exact normalized name plus exact approved-cost match can be verified. Fuzzy matching is never production truth; ambiguous rows are quarantined in `paimana_identity_audit.csv`.

## Audited lifecycle fields

| Feature | Actual monthly observations? | Real matching snapshot? | Valid timestamps? | Decision |
|---|---:|---:|---:|---|
| `progress_velocity_3m` | Audited per window | Exact trajectory only | Uses snapshots through T | Keep only when temporally distributed |
| `progress_velocity_6m` | Audited per window | Exact trajectory only | Uses snapshots through T | Keep only when temporally distributed |
| `progress_acceleration` | Audited per window | Exact trajectory only | Difference of as-of slopes | Keep only when temporally distributed |
| `cost_growth_velocity_3m/6m` | Audited per window | Exact trajectory only | Uses actual date deltas | Keep only when temporally distributed |
| `milestone_delay_count` | No | No | No milestone timeline | Remove from training |
| `monthly_expenditure_growth` | No | No | Legacy demonstration only | Remove from training |

Missing trajectory features remain null when history is insufficient. They are never backfilled from later snapshots.

## Preserved five-feature baseline contract

The legacy completed-outcome pipeline intentionally remains fixed to:

- `approved_cost_cr`
- `sector_average_delay`
- `sector_average_cost_overrun`
- `sector`
- `project_size_category`

Agency fields remain excluded from that controlled baseline even though newer outcome extraction improves their coverage.

## Monthly lifecycle contract

The monthly model runs the richer audit independently inside each training window. The generated `models/monthly_lifecycle/*/feature_quality_report.json` files contain per-feature availability by year and parser, project coverage, leakage notes, and the exact keep/remove decision. In the completed evaluations, direct expenditure/schedule/duration features, revised-cost state, cost trajectories, sector priors, and agency priors survive; physical-progress trajectories and ministry are rejected because their temporal coverage is insufficient. No missing lifecycle field is replaced with zero or future information.
