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

# Keep the repository source-first: temporal demo data and binaries are reproducible.
if [[ ! -f data/project_history.csv ]]; then
  echo "[InfraSight] Temporal demonstration history not found. Generating it..."
  python scripts/generate_project_history.py
fi

if [[ ! -f data/processed/project_monthly_history.csv && -f data/raw/paimana_archive/manifest.json ]]; then
  echo "[InfraSight] Normalizing checked-in official PAIMANA archive reports..."
  python scripts/ingest_paimana_archive.py --local-only
fi

REQUIRED=(
  models/cost_model.pkl
  models/delay_model.pkl
  models/model_metrics.json
)
missing=0
for artifact in "${REQUIRED[@]}"; do
  if [[ ! -f "$artifact" ]]; then missing=1; break; fi
done
if [[ "$missing" -eq 1 ]]; then
  echo "[InfraSight] Temporal forecast artifacts not found. Training model candidates..."
  python scripts/train_models.py
fi

exec uvicorn backend.app.main:app --host 127.0.0.1 --port "${PORT:-8000}" --reload
