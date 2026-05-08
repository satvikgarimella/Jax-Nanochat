from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

DEFAULT_MODEL_API_URL = "https://surjahead--nanochat-nanochat-api.modal.run"

MODEL_API_URL = os.getenv("NANOCHAT_MODEL_API_URL", DEFAULT_MODEL_API_URL).rstrip("/")

app = FastAPI(title="NanoChat Vercel Health", version="0.1.0")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    upstream: str


@app.get("/api/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", upstream=MODEL_API_URL)
