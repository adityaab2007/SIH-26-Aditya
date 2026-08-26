# Experiment 12 — Trajectory-enhanced cost forecasting

## Final scope

Experiment 12 is now a **cost-only challenger**.

The original cost+delay variant was evaluated on two independent future windows. It produced repeatable cost-overrun MAE improvements, but it did not produce a defensible delay improvement. The active adapter therefore changes only cost prediction and retains the freshly trained production delay model unchanged.

## Measured evidence that motivated the split

| Training window | Target | Production MAE | Exp 12 v2 MAE | Change |
| --- | --- | ---: | ---: | ---: |
| 2001–2019 | Cost | 30.255 pp | 29.224 pp | **3.4077% better** |
| 2001–2019 | Delay | 545.877 days | 545.877 days | 0.00% |
| 2001–2021 | Cost | 27.309 pp | 26.872 pp | **1.6002% better** |
| 2001–2021 | Delay | 534.691 days | 537.169 days | **0.4634% worse** |

The project-level cost bootstrap evidence was also positive on both windows:

- 2001–2019: 95% improvement interval **2.5721% to 4.3718%**, probability candidate better **1.000**, 949 projects.
- 2001–2021: 95% improvement interval **0.3920% to 2.9472%**, probability candidate better **0.997**, 721 projects.

The delay evidence did not justify promotion. For 2001–2021 its confidence interval crossed zero, so the delay variant is explicitly recorded as rejected rather than being bundled with the successful cost change.

## Final prediction contract

- **Cost overrun:** Experiment 12 trajectory-enhanced challenger.
- **Delay:** existing freshly trained production delay model.
- **Risk:** production behavior is unchanged.

Experiment 12 does not train, save, serve, or compare an experimental delay model in its final `v3_cost_only` implementation.

## Changed dimension

Only the **cost feature representation** changes.

The following controls remain fixed against the production cost model:

- same identity-verified supervised quarterly snapshots;
- same training and future-holdout project boundary;
- same final cost-overrun target;
- same project-balanced weighting policy;
- same production-selected cost regressor family;
- same LightGBM/XGBoost/ExtraTrees constructors and deterministic production seed;
- no automatic production promotion.

## Trajectory representation

The cost challenger can select leakage-safe information from official monthly history, including:

- trailing-12-month history depth;
- 3/6/12-month cost growth rates and acceleration;
- cost revision frequency and magnitude;
- months since a cost revision;
- cost volatility and worsening persistence;
- 3/6/12-month expenditure velocity and acceleration;
- expenditure normalized by approved cost;
- spend-versus-expected-progress gap where real expected-progress data is available;
- schedule/slippage trajectory signals when they improve **cost** prediction on the internal historical validation block.

Using a schedule-derived signal in the cost model does not make this a delay experiment. The experiment target is cost; feature sources may include any leakage-safe official information that helps predict that target.

Candidate features must have at least 10% training-window availability and more than one observed value.

## Internal feature selection

Experiment 12 does not force every new trajectory feature into the cost model.

For each requested training window it compares cost feature groups on the final two completion years **inside the training period** and selects the lowest-MAE group before fitting on the full training window. The future holdout is never used to choose the feature group.

The candidate groups are:

1. production features only;
2. cost trajectory features;
3. cost + expenditure trajectory features;
4. all usable trajectory signals.

This is why the two verified windows legitimately selected different cost representations: `cost_plus_spend` for 2001–2019 and `all_trajectory` for 2001–2021.

## Leakage rule

Every Experiment 12 feature uses only the current official PAIMANA snapshot and earlier snapshots for the same canonical project. Later reports and final outcomes are never consulted while constructing trajectory features.

Tests verify that appending an extreme future report cannot change earlier Experiment 12 feature values.

## Retrain & Compare integration

`backend/app/ml/experiments/adapter_exp12.py` registers:

- `EXPERIMENT_ID = exp_12`
- `EXPERIMENT_SEQUENCE = 12`
- `scope = cost`
- active implementation: `v3_cost_only`

The generic Retrain & Compare endpoint still retrains production first on the frozen evidence contract. Experiment 12 then fits only its cost challenger. For individual held-out projects the browser receives production cost/delay predictions plus the Experiment 12 cost prediction before reveal. Delay remains the production prediction.

## Artifact isolation

Experiment evidence is written below:

`models/monthly_lifecycle/experiments/exp_12/<window>/<run_id>/`

The final cost-only implementation writes `cost_model.pkl`; it deliberately does not write an experimental delay model.

Registry evidence records `delay_policy: production_retained` and `delay_experiment_status: rejected_after_two_window_audit` so the target-level decision remains auditable.

Experiment 12 remains `PENDING` and `promotion_allowed: false`; merging this PR does not automatically replace production.
