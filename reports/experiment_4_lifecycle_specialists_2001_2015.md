# Experiment 4 — Lifecycle-Specific Forecasting

Run: 2001–2015 training, 2016–2025 future holdout. The holdout contains 24,186
snapshot rows from 1,549 projects for the global baseline and the same 24,186
rows for routed specialist evaluation, including explicit global fallback rows.

| Approach | Cost MAE (pp) | Delay MAE (days) |
|---|---:|---:|
| Global baseline | 39.998 | 535.479 |
| Lifecycle-aware global | 39.867 | 535.188 |
| Routed specialists | 37.347 | 530.613 |

Routed specialists reduced cost MAE by **6.63%** and delay MAE by **0.91%**
relative to the global baseline. Positive values indicate lower error.

| Stage | Cost global | Cost specialist | Cost Δ% | Delay global | Delay specialist | Delay Δ% | Cost algorithm | Delay algorithm |
|---|---:|---:|---:|---:|---:|---:|---|---|
| early (0–25%) | 42.212 | 43.659 | −3.428 | 467.708 | 487.898 | −4.317 | Extra Trees | Extra Trees |
| early-mid (25–50%) | 39.676 | 36.090 | 9.038 | 470.842 | 472.785 | −0.413 | XGBoost | LightGBM |
| late-mid (50–75%) | 37.330 | 33.901 | 9.186 | 501.095 | 495.690 | 1.079 | XGBoost | Extra Trees |
| late (75%+) | 37.546 | 34.790 | 7.340 | 530.381 | 524.282 | 1.150 | XGBoost | Extra Trees |

The experiment remains experimental: early-stage models regress on both
targets, while later stages show the clearest benefit. Specialist `.pkl`
artifacts and the full JSON provenance/evaluation bundle are generated under
`models/lifecycle_specialists/2001_2015/` by the explicit retraining endpoint
and are intentionally gitignored.
