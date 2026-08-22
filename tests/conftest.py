"""Make a fresh clone testable without committing generated model/data binaries."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'paimana_projects_may_2026.csv'
PROCESSED = ROOT / 'data' / 'processed' / 'model_dataset.csv'
HISTORY = ROOT / 'data' / 'project_history.csv'
MODEL_ARTIFACTS = [
    ROOT / 'models' / 'cost_model.pkl',
    ROOT / 'models' / 'delay_model.pkl',
    ROOT / 'models' / 'registry.json',
    ROOT / 'models' / 'global_feature_importance.json',
]

if not RAW.exists():
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'seed_official_data.py')], cwd=ROOT, check=True)
if not PROCESSED.exists():
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'build_model_dataset.py')], cwd=ROOT, check=True)
if not HISTORY.exists():
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'generate_project_history.py')], cwd=ROOT, check=True)
if any(not artifact.exists() for artifact in MODEL_ARTIFACTS):
    subprocess.run([sys.executable, '-m', 'backend.app.ml.train'], cwd=ROOT, check=True)
