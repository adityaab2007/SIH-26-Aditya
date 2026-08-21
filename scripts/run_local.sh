#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# Rebuild the included real PAIMANA subset on a fresh clone.
if [[ ! -f data/raw/paimana_projects_may_2026.csv || ! -f data/raw/paimana_high_value_history.csv ]]; then
  echo "[InfraSight] Real PAIMANA seed files not found. Rebuilding the curated official dataset..."
  python scripts/seed_official_data.py
fi

# Keep the repository source-first: trained binaries are reproducible and ignored by git.
REQUIRED=(
  models/schedule_classifier_xgboost.joblib
  models/cost_classifier_xgboost.joblib
  models/schedule_regressor_random_forest.joblib
  models/cost_regressor_random_forest.joblib
)
missing=0
for artifact in "${REQUIRED[@]}"; do
  if [[ ! -f "$artifact" ]]; then missing=1; break; fi
done
if [[ "$missing" -eq 1 ]]; then
  echo "[InfraSight] Trained artifacts not found. Training all model families from the real PAIMANA subset..."
  python scripts/train_models.py
fi

exec uvicorn backend.app.main:app --host 127.0.0.1 --port "${PORT:-8000}" --reload
