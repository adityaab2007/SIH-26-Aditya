# Architecture

InfraSight is organized by **domain responsibility**, following the understandable file separation used in the user's ATS while avoiding large page/controller monoliths.

## Request path

```text
Browser page
  → frontend/src/services/api.js
  → FastAPI route
  → domain service
  → engineered real project row
  → selected trained artifact
  → explanation / peer / history service
  → JSON
  → feature component
```

## Backend boundaries

- `routes/`: HTTP shape only; minimal business logic.
- `services/data_service.py`: project and history repositories.
- `services/model_service.py`: artifact registry and prediction primitives.
- `services/prediction_service.py`: one-project risk/priority orchestration.
- `services/explanation_service.py`: local SHAP contributions.
- `services/benchmark_service.py`: peer selection and medians.
- `services/history_service.py`: longitudinal replay.
- `services/portfolio_service.py`: vectorized portfolio-level scoring.
- `services/assistant_service.py`: deterministic grounded portfolio queries.
- `ml/`: data engineering and model training, kept out of API modules.

## Frontend boundaries

- `pages/`: route-level orchestration.
- `features/`: domain visualizations/interactions.
- `components/`: reusable generic UI.
- `services/api.js`: all network calls.
- `utils/`: formatting only.
- `styles/`: base, layout, components and page-level CSS split.

This means a teammate looking for scenario logic, history replay, model metrics, project table behavior, or SHAP logic has a predictable file location.
