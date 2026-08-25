# Experiment 4 Improved

Evaluation uses the official 2001–2015 training window and untouched
2016–2025 holdout: 24,186 snapshots from 1,549 projects. `test-1` remains the
frozen v1 reference.

## Baseline and frozen v1

| Model | Cost MAE (pp) | Delay MAE (days) |
|---|---:|---:|
| Main/global reference | 39.998 | 535.479 |
| Experiment 4 v1 routed specialists | 37.347 | 530.613 |
| v2A fairness-only routed specialists | 42.517 | 541.230 |
| Improved independent specialists | 41.495 | 552.944 |
| Improved hybrid router | 41.546 | 553.799 |

The fairness-only rerun removed the artificial 100-estimator cap and
renormalized weights within every filtered stage. It is therefore the valid
capacity/weight comparison, even though it regressed against the frozen v1
artifact on this holdout.

## Improved early and partial pooling

The improved independent early specialist reached cost MAE **52.565 pp** and
delay MAE **392.594 days** on its feature-model comparison rows. The temporal
partial-pooling selector chose alpha **0.25** for cost and **0.50** for delay;
its final hybrid early errors were **54.801 pp** and **430.096 days**. These
figures do not beat the frozen v1 early reference (42.212 pp, 467.708 days)
on cost, and the improved architecture is not recommended for promotion.

## Stage comparison

| Stage | Improved specialist cost | Improved specialist delay | Cost Δ vs improved global | Delay Δ vs improved global |
|---|---:|---:|---:|---:|
| 0–25% | 52.565 | 392.594 | +3.609% | −3.954% |
| 25–50% | 45.337 | 495.130 | +0.211% | −11.813% |
| 50–75% | 39.331 | 501.700 | +0.914% | +0.392% |
| 75%+ | 38.294 | 555.792 | +1.195% | −1.101% |

Positive cost/delay deltas mean lower error. Negative values are regressions.

## Early checkpoints

Checkpoint diagnostics use real official snapshots only; no interpolation was
added.

| Checkpoint | Rows | Projects | Cost MAE | Delay MAE | Cost R² | Delay R² |
|---|---:|---:|---:|---:|---:|---:|
| 0–10% | 66 | 50 | 61.572 | 324.816 | −0.4046 | 0.1181 |
| 10–20% | 221 | 163 | 50.582 | 371.587 | −0.4107 | 0.0537 |
| 20–25% | 173 | 166 | 50.632 | 456.205 | −0.1322 | 0.0556 |

## Selected algorithms and features

Improved independent specialists selected: early LightGBM/Extra Trees,
early-mid Extra Trees/LightGBM, late-mid XGBoost/LightGBM, and late
XGBoost/XGBoost for cost/delay respectively. The global and specialist
definitions use LightGBM 240, XGBoost 240, and Extra Trees 260 estimators;
the v1-only 100-tree cap is absent.

Added as-of-safe features include expenditure-time gap and burn rate,
approved-cost-per-planned-day, current cost revision indicators, slippage
onset and per-day slippage, snapshot count, first/previous snapshot changes,
short-history velocities, previous slippage change, and practical sector/size,
sector/duration, and agency/sector interactions. Physical progress remains
missing where unavailable; no synthetic progress was fabricated.

Tree importance is labelled `tree_feature_importance`, not SHAP. The persisted
stage JSON files contain cost and delay feature rankings; residual correction
drivers are in the improved comparison artifact.

## Leakage safeguards

- lifecycle boundaries remain exactly 0–25%, 25–50%, 50–75%, and 75%+;
- stage weights are recomputed as 1 / retained snapshots per project;
- previous/first/history features use only records at or before the current
  official snapshot;
- final targets and completion outcomes are excluded from model features;
- residual targets use expanding temporal out-of-fold global predictions;
- alpha selection uses an internal temporal validation fold only;
- 2016–2025 is used once for final evaluation, never for selection;
- actual outcomes remain hidden in the pre-reveal simulation API.

## Recommendation

**NO MEANINGFUL IMPROVEMENT / REGRESSION.** Keep the production/global model
as the default. Keep Experiment 4 v1 and v2 behind explicit experimental
comparison modes; do not promote the improved early specialist or hybrid.
The strongest evidence is that the fairness correction is scientifically
necessary but worsens this controlled holdout, while the improved independent
and hybrid variants also fail the required early-cost threshold.
