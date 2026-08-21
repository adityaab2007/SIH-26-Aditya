# Model outputs

InfraSight trains **16 model artifacts locally**: Logistic/Linear Regression, Random Forest, XGBoost and CatBoost across schedule classification, cost classification, schedule regression and cost regression.

Generated artifacts are intentionally not committed:

- `*.joblib`
- `metrics.json`
- `registry.json`
- `global_feature_importance.json`

A fresh clone recreates all of them from the real PAIMANA seed data when `./scripts/run_local.sh` starts, or explicitly with:

```bash
python scripts/seed_official_data.py
python scripts/train_models.py
```

`training_output.txt` is retained as a human-readable record of the evaluation run used for the current README figures.
