"""FastAPI entrypoint for local EmoGame evaluation APIs."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes.evaluation import router as evaluation_router


app = FastAPI(
    title="EmoGame API",
    version="0.1.0",
    description="Local APIs for skin evidence, evaluation, and sales action reports.",
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(evaluation_router, prefix="/api")
