.PHONY: data train test run browser-test

data:
	PYTHONPATH=. python scripts/seed_official_data.py
	PYTHONPATH=. python scripts/generate_project_history.py

train:
	PYTHONPATH=. python scripts/train_models.py

test:
	PYTHONPATH=. pytest

run:
	PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

browser-test:
	PYTHONPATH=. python tests/browser_smoke.py
