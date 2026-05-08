from __future__ import annotations

import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from api.chat_pipeline import finalize_model_response, prepare_model_payload
except ImportError:
    from .chat_pipeline import finalize_model_response, prepare_model_payload

DEFAULT_MODEL_API_URL = "https://surjahead--nanochat-nanochat-api.modal.run"

MODEL_API_URL = os.getenv("NANOCHAT_MODEL_API_URL", DEFAULT_MODEL_API_URL).rstrip("/")

app = FastAPI(title="NanoChat Vercel Chat", version="0.1.0")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    max_tokens: int = Field(default=100, ge=1, le=512)


class ChatResponse(BaseModel):
    response: str


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        direct_response, body, _mode = await prepare_model_payload(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if direct_response is not None:
        return ChatResponse(response=direct_response)
    canonical_answer = body.pop("_canonical_answer", None)

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            upstream = await client.post(MODEL_API_URL, json=body)
            upstream.raise_for_status()
            data = upstream.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Upstream error: {detail[:400]}") from exc
    except httpx.HTTPError as exc:
        message = str(exc) or exc.__class__.__name__
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {message}") from exc

    response_text = data.get("response")
    if not isinstance(response_text, str):
        raise HTTPException(status_code=502, detail="Upstream response missing string field 'response'")

    final_response = finalize_model_response(payload.message, response_text, canonical_answer)
    return ChatResponse(response=final_response)
