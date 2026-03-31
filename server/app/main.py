from fastapi import FastAPI

from app.core.logging import setup_logging
from app.api.routes_health import router as health_router

setup_logging()

app = FastAPI(
    title="Research Paper Assistant",
    version="0.1.0",
)


# Routes
app.include_router(health_router)


@app.get("/")
def root():
    return {
        "message": "Research Paper Assistant API",
    }