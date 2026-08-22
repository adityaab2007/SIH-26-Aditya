"""Make a fresh clone testable without committing generated model/data binaries."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'paimana_projects_may_2026.csv'
MODEL = ROOT / 'models' / 'cost_model.pkl'

if not RAW.exists():
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'seed_official_data.py')], cwd=ROOT, check=True)
if not MODEL.exists():
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'generate_project_history.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'backend.app.ml.train'], cwd=ROOT, check=True)
