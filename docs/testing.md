# Testing

The prototype is tested at four layers.

## 1. Data integrity

`tests/test_data_integrity.py` verifies:

- expected real curated row count;
- official-code uniqueness;
- no app-local high-value IDs leak into the official-code dataset;
- MoSPI source domains;
- historical codes map back to current project master rows.

It also verifies the immutable official archive manifest, normalized monthly-history schema, multiple observations per project, official project-code shape, and null preservation for unavailable fields.

## 2. Model artifacts

`tests/test_models.py` verifies temporal cost/delay artifacts, time-split metrics, future-label formulas, and cutoff backtest arithmetic.

## 3. API integration

`tests/test_api.py` covers health, exact forecast contract, SHAP factors, validation APIs, history and model metrics.

## 4. Browser smoke test

`tests/browser_smoke.py` exercises:

- Dashboard;
- Rajasthan Refinery project page;
- Scenario Explorer;
- Time Machine;
- Project Forecast;
- Model Performance; and
- Prediction Accuracy.

It asserts key visible values and captures screenshots. The build container has an administrator policy blocking Chromium direct loopback navigation, so the test supports `INFRASIGHT_BROWSER_PROXY=1`, which serves the **exact live localhost responses** into Playwright. Normal developer machines use direct localhost navigation.
