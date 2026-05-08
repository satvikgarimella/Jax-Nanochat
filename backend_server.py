from __future__ import annotations

import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat_pipeline import finalize_model_response, prepare_model_payload

DEFAULT_MODEL_API_URL = "https://surjahead--nanochat-nanochat-api.modal.run"
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def _parse_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


MODEL_API_URL = os.getenv("NANOCHAT_MODEL_API_URL", DEFAULT_MODEL_API_URL).rstrip("/")

app = FastAPI(title="NanoChat Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    upstream: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    max_tokens: int = Field(default=150, ge=1, le=512)


class ChatResponse(BaseModel):
    response: str


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", upstream=MODEL_API_URL)

@app.post("/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        direct_response, body, mode = await prepare_model_payload(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if direct_response is not None:
        print(f"Mode=math for: {payload.message!r}")
        return ChatResponse(response=direct_response)

    print(f"Mode={mode} for: {payload.message!r}")
    canonical_answer = body.pop("_canonical_answer", None)

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            upstream = await client.post(MODEL_API_URL, json=body)
            upstream.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {detail[:400]}") from exc
    except httpx.HTTPError as exc:
        message = str(exc) or exc.__class__.__name__
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {message}") from exc

    data = upstream.json()
    response_text = data.get("response")
    if not isinstance(response_text, str):
        raise HTTPException(status_code=502, detail="Upstream response missing string field 'response'")

    final_response = finalize_model_response(payload.message, response_text, canonical_answer)
    return ChatResponse(response=final_response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
