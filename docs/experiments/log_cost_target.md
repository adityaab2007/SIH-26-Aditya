# Experiment 1: log-transformed cost target

## Purpose

Extreme cost overruns create a highly skewed regression target. This isolated
experiment tests whether compressing the cost-overrun target improves future
holdout accuracy. It does not change the production model.

## Controlled comparison

Both versions use the same:

- 303 official PAIMANA training projects completed from 2001 through 2017
- 156 official future-holdout projects completed from 2018 through 2024
- temporal split and historical-prior policy
- CatBoost MAE depth-4 algorithm and hyperparameters
- five production features
- original-percentage evaluation scale
- nested temporal sector-correction methodology

The delay and four-class risk models are not trained, modified, or saved by the
experiment. Their reported values are copied from the baseline evaluation.

## Target transformation

For non-negative overruns, the requested transformation is used directly:

```text
z = log1p(cost_overrun_percentage)
cost_overrun_percentage = expm1(z)
```

The real training set also contains 187 cost underruns, including 183 values at
or below -1%. Plain `log1p(y)` is undefined for those observations. Dropping or
clipping them would violate the requirement to use exactly the same data.
Therefore, the experiment uses the invertible signed extension:

```text
z = sign(y) * log1p(abs(y))
y = sign(z) * expm1(abs(z))
```

This equals ordinary `log1p` for every non-negative overrun while preserving all
baseline projects and all negative values.

## Result

| Metric | Baseline | Log-target experiment | Change |
|---|---:|---:|---:|
| Cost MAE | 32.886 pp | 32.868 pp | 0.018 pp better |
| Cost RMSE | 60.183 pp | 60.566 pp | 0.383 pp worse |
| Cost MAPE | 257.604% | 256.868% | 0.736 pp better |
| Delay MAE | 1026.711 days | 1026.711 days | unchanged |

The MAE improvement is **0.05%**. This is measurable but operationally
negligible, while RMSE became slightly worse. The experiment therefore does not
provide enough evidence to replace the production cost model.

## Isolation and reproducibility

- Model artifact: `models/experiments/log_cost_target/cost_model.pkl`
- Metadata: `models/experiments/log_cost_target/metadata.json`
- Baseline report: `reports/experiments/v1_baseline.json`
- Experiment report: `reports/experiments/v2_log_cost_target.json`
- Evolution report: `reports/experiments/model_evolution.json`
- Target audit: `reports/experiments/cost_target_analysis_before.json`

Run with:

```bash
source .venv/bin/activate
python scripts/run_log_cost_target_experiment.py
```

The runner records the production cost-model SHA-256 before and after training
and fails the comparison contract if the experiment features differ from the
baseline metadata.
