# SIH26103 Longitudinal Forecasting Demo

## Goal
Predict future cost escalation and schedule delays using project progress history.

## Data flow

Historical monthly snapshots
-> Feature engineering
-> Temporal training split
-> Cost and delay models
-> Risk prediction + explanation

## Required model targets

- future_cost_escalation_pct
- future_schedule_extension_days

## Demo flow

Select project -> analyse current status -> predict cost/time overrun -> show risk factors.
