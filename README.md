# InfraSight AI — SIH26103

**Predictive infrastructure project intelligence for MoSPI / PAIMANA.**

InfraSight AI is an end-to-end prototype for **Smart India Hackathon 2026 problem statement SIH26103**, “Use case on web-based integrated project-monitoring platform”. The system turns public PAIMANA project records into a modular decision-support workspace for cost/schedule overrun intelligence, project prioritisation, explainability, peer benchmarking, historical replay, data-quality review and scenario sensitivity analysis.

> **Data integrity:** the showcased project rows are real PAIMANA public records. The repository deliberately excludes high-value records when an official project code was not surfaced; it does not invent PAIMANA identifiers.

## What is implemented

- **Portfolio command dashboard** over the real curated PAIMANA subset.
- **Project explorer** with sector/search filters and source links.
- **Schedule risk baseline** trained with Logistic Regression, Random Forest, XGBoost and CatBoost.
- **Cost risk baseline** trained with the same four classifier families.
- **Schedule extension regression** with Linear Regression, Random Forest, XGBoost and CatBoost.
- **Cost escalation regression** with the same four regressor families.
- **Automatic model selection** from cross-validated results rather than assuming one algorithm wins.
- **Project-level SHAP explanations** for the selected tree classifiers.
- **Priority / intervention queue** combining model risk signals with financial exposure.
- **Peer benchmarking** against similar projects in the same sector.
- **Historical Time Machine** using real monthly PAIMANA/Flash Report snapshots.
- **Scenario Explorer** for sensitivity testing; outputs are explicitly not presented as causal guarantees.
- **Data Quality Observatory** that surfaces missing/contradictory operational fields rather than silently repairing them.
- **Grounded analytics assistant** that answers from computed local portfolio analytics without generating risk numbers through an LLM.
- **FastAPI API + modular browser UI**, with feature/page files separated in the style of the companion ATS project.

## Real dataset currently included

The reproducible dataset seed builds:

- **96** PAIMANA May 2026 project rows with surfaced official project codes.
- **14** official historical snapshots across selected high-value projects.
- **11** represented sectors.
- roughly **₹5.67 lakh crore** of original approved project cost in the included rows.

Primary source surfaces:

- PAIMANA Public Dashboard: `https://ipm.mospi.gov.in/Home/PublicDashboard`
- PAIMANA high-value project surface: `https://ipm.mospi.gov.in/Home/GetHighlyValue`
- PAIMANA / MoSPI monthly Flash Reports, including March 2026.

See [`docs/data-provenance.md`](docs/data-provenance.md) for the exact provenance and limitations.

## Current model results

These are **real cross-validated metrics from the included dataset**, not placeholder numbers.

### Classification baselines

| Task | Best model | Rows | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| Observed schedule overrun > 90 days | XGBoost | 70 | **0.8952** | **0.9531** |
| Observed cost overrun > 5% | XGBoost | 56 | **0.8511** | 0.7000 |

### Regression baselines

| Task | Best model | Rows | MAE | R² |
|---|---|---:|---:|---:|
| Observed schedule extension (days) | Random Forest | 70 | **269.91 days** | **0.5496** |
| Observed cost escalation (%) | Random Forest | 56 | **23.02 percentage points** | **0.4302** |

The linear schedule model performs poorly on this small heterogeneous subset, which is intentionally visible in Model Lab rather than hidden. The evaluation run for all 16 model artifacts is recorded in [`models/training_output.txt`](models/training_output.txt); fresh runs regenerate machine-readable metrics/registry files locally.

### Important forecasting distinction

The currently included trained artifacts are **baseline overrun-intelligence models over the real May 2026 snapshot**. They are useful for proving the complete data → feature → model → explanation → API → UI pipeline, but they are **not represented as a fully forward-validated six-month forecasting model**.

The production SIH step is to ingest the larger OCMS + PAIMANA monthly archive and train on labels of the form:

```text
project state at month T
        ↓
will deadline shift >90 days by T+6?
will cost increase >5% by T+6?
how many days / % will it move?
```

The repository already contains [`backend/app/ml/forward_labels.py`](backend/app/ml/forward_labels.py) for generating those leakage-safe future labels once the archive is expanded.

## Architecture

```text
SIH-26/
├── backend/
│   └── app/
│       ├── core/                  # configuration
│       ├── ml/                    # features, training, forward labels
│       ├── routes/                # thin API routes by feature
│       └── services/              # data, prediction, SHAP, peers, history...
├── frontend/
│   └── src/
│       ├── components/            # shared UI components
│       ├── features/              # feature-specific view modules
│       ├── pages/                 # one top-level file per screen
│       ├── services/              # API client
│       ├── styles/                # base/layout/components/page styles
│       └── utils/                 # formatting helpers
├── data/
│   ├── raw/                       # generated source-aligned real records
│   └── processed/                 # generated engineered model dataset
├── models/                        # generated binaries/metrics + retained training record
├── scripts/                       # seed/train/run entrypoints
├── tests/                         # data, API, model and browser tests
└── docs/                          # architecture/methodology/provenance
```

The organization mirrors the feature/page/service separation used in `fyndbridge-ats`, but intentionally avoids putting an entire feature into one huge page/controller file.

## Run locally

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

./scripts/run_local.sh          # rebuilds real-data CSVs and trains missing model artifacts automatically
```

Open:

```text
http://127.0.0.1:8000
```

The frontend has no CDN/runtime dependency; FastAPI serves the API and the modular SPA together. On a fresh clone, first launch rebuilds the real PAIMANA dataset from the checked-in source-aligned seed and trains the selected inference artifacts before starting the server.

## Run tests

```bash
pytest
```

For the browser smoke test:

```bash
pip install -r requirements-dev.txt
playwright install chromium
python tests/browser_smoke.py
```

The development container used for this build applies a policy that blocks Chromium from directly navigating loopback URLs. In that environment the test was run with the exact localhost responses proxied into Chromium. This passed across **Dashboard → real Rajasthan Refinery project → Scenario Explorer → Time Machine → Model Lab** with zero browser console/page errors.

## Real test case: Rajasthan Refinery (`701263`)

From the official PAIMANA row:

- Original cost: **₹43,129 Cr**
- Revised cost: **₹79,459 Cr**
- Cumulative expenditure: **₹69,997 Cr**
- Original completion: **31 Oct 2022**
- Revised completion: **30 Jun 2026**
- Physical progress: **92%**
- Observed cost escalation: **84.2%**
- Observed schedule extension: **1,338 days / ~3.7 years**

The selected baseline models currently return:

- schedule risk signal: **99.34%**
- cost risk signal: **92.60%**
- review-priority score: **96.5 / 100 — Critical**

These model signals are explicitly labelled as baseline overrun intelligence, while the observed cost/schedule values are separately displayed as facts.

## SIH26103 mapping

| Problem-statement outcome | InfraSight module |
|---|---|
| Cost Overrun Prediction Model | Cost classifier + regression pipeline |
| Time Overrun Prediction Model | Schedule classifier + regression pipeline |
| Project Risk Scoring Framework | Portfolio review-priority engine |
| Early Warning Alert System | Early Warnings queue |
| Benchmarking & Comparative Analytics | Sector peer benchmarking |
| Cost Escalation Driver Analysis | Local SHAP driver view |
| AI-powered Monitoring Dashboard | Dashboard + project intelligence pages |
| LLM-enabled Project Intelligence Assistant | Grounded analytics interface now; optional LLM adapter can be added after core analytics validation |
| Documentation & reproducibility | This repository + docs + tests |

## Why the code avoids overclaiming

- SHAP is described as **feature contribution**, not causality.
- Scenario output is described as **model sensitivity**, not a guaranteed intervention effect.
- Missing PAIMANA values are surfaced as data-quality signals.
- Current baseline models are not called forward-validated future forecasts.
- No risk number is generated by an LLM.
- Every showcased project retains its official source URL.

## Next SIH milestone

The highest-value next step is expanding the longitudinal archive. Once sufficient monthly records are parsed, retrain with grouped/time-based validation so a project’s future snapshots can never leak into its own training history. That enables the full “freeze month T → predict T+6 → reveal actual outcome” Time Machine envisioned by SIH26103.
