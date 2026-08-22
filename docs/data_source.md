# Longitudinal demo data source

`data/project_history.csv` is a deterministic **synthetic demonstration** timeline. It preserves project IDs, names, sectors, ministries, approved costs and planned dates from the included PAIMANA May 2026 seed, then creates coherent monthly progress, expenditure, revision and milestone trajectories. It is not presented as official historical project reporting.

The forecasting pipeline accepts the same schema when an authorised PAIMANA/OCMS monthly export is available. Replace `data/project_history.csv` with that governed export and run `python -m backend.app.ml.train`; no model or API redesign is required.

The synthetic data exists solely to demonstrate the SIH26103 longitudinal feature, time-split training, prediction, and explainability flow. It must not be used for operational or policy decisions.
