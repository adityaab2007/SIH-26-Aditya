from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"
APP_NAME = "InfraSight AI"
APP_VERSION = "0.1.0"
