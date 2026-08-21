# Testing

The prototype is tested at four layers.

## 1. Data integrity

`tests/test_data_integrity.py` verifies:

- expected real curated row count;
- official-code uniqueness;
- no app-local high-value IDs leak into the official-code dataset;
- MoSPI source domains;
- historical codes map back to current project master rows.

## 2. Model artifacts

`tests/test_models.py` verifies all **16** artifacts and validates metric ranges.

## 3. API integration

`tests/test_api.py` covers health, portfolio, model metrics, scenario disclaimer, and a real Rajasthan Refinery prediction.

## 4. Browser smoke test

`tests/browser_smoke.py` exercises:

- Dashboard;
- Rajasthan Refinery project page;
- Scenario Explorer;
- Time Machine;
- Model Lab.

It asserts key visible values and captures screenshots. The build container has an administrator policy blocking Chromium direct loopback navigation, so the test supports `INFRASIGHT_BROWSER_PROXY=1`, which serves the **exact live localhost responses** into Playwright. Normal developer machines use direct localhost navigation.
