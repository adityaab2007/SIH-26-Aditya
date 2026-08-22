from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import APP_NAME, APP_VERSION, FRONTEND_DIR
from backend.app.routes import assistant, data_quality, history, models, portfolio, projects, scenario, model_validation

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.include_router(portfolio.router)
app.include_router(projects.router)
app.include_router(models.router)
app.include_router(history.router)
app.include_router(scenario.router)
app.include_router(assistant.router)
app.include_router(data_quality.router)
app.include_router(model_validation.router)

@app.get("/api/health")
def health():
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}

app.mount("/src", StaticFiles(directory=FRONTEND_DIR / "src"), name="src")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

@app.get("/{full_path:path}")
def spa(full_path: str):
    if full_path.startswith("api/"):
        return {"detail": "Not found"}
    return FileResponse(FRONTEND_DIR / "index.html")
