# Lifecycle feature audit

## Production data boundary

The production time-window models read `data/processed/paimana_completed_outcomes.csv` through `backend/app/ml/real_time_windows.py`. They never read `data/project_history.csv`, which is a legacy deterministic demonstration dataset.

The completed-outcome archive contains official approved cost, planned commissioning date, reported completion expenditure, completion date, and sector information. The checked-in monthly monitoring archive has no safe exact project-ID match to these completed outcomes. A fuzzy name join is not accepted because similarly named infrastructure packages may be different projects.

## Audited lifecycle fields

| Feature | Actual monthly observations? | Real matching snapshot? | Valid timestamps? | Decision |
|---|---:|---:|---:|---|
| `progress_trend_6m` | No | No | No matching timestamps | Remove from training |
| `progress_trend_12m` | No | No | No matching timestamps | Remove from training |
| `progress_acceleration` | No | No | No matching timestamps | Remove from training |
| `progress_velocity` | No | No | No matching timestamps | Remove from training |
| `milestone_delay_count` | No | No | No milestone timeline | Remove from training |
| `monthly_expenditure_growth` | No | No | Legacy demonstration only | Remove from training |

The production feature builder may calculate these fields when a future input contains multiple real snapshots with valid dates. They are not members of the current production training feature contract and missing values remain missing.

## Final production feature contract

The automated audit retains only features that are observed, variable, and available in at least 5% of the selected training window. For the official 2001–2015 window this is:

- `approved_cost_cr`
- `sector_average_delay`
- `sector_average_cost_overrun`
- `sector`
- `project_size_category`

Agency fields, ministry, revised cost, expenditure behaviour, and duration behaviour are excluded because the completed-project source does not publish enough valid values. Historical sector values for a training row use only projects completed before that row's prediction-period date. No missing lifecycle field is replaced with zero, a sector average, or synthetic history.
