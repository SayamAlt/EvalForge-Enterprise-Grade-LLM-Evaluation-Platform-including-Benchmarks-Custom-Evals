"""
Model registry endpoints — Sprint 3.

Routes:
    GET    /models              List models (optional ?provider= filter, ?available_only=true)
    POST   /models              Register a model configuration
    GET    /models/{model_id}   Get model details (pricing, capabilities)
    DELETE /models/{model_id}   Unregister a model
    POST   /models/ping-all     Ping all models in parallel, update availability
    POST   /models/{model_id}/ping  Test connectivity to the provider
"""
import asyncio
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from evalforge.providers.registry import MODEL_REGISTRY, get_provider
from pathlib import Path
from typing import Literal
import yaml
from evalforge.providers.base import GenerationRequest

router = APIRouter()

class ModelInfo(BaseModel):
    model_id: str = Field(..., description="Catalog key — pass this to evaluation endpoints", examples=["gpt-4o", "llama-3.3-70b-groq"])
    provider: str = Field(..., examples=["openai", "anthropic", "moonshot"])
    provider_model_id: str = Field(..., description="Actual model ID sent to the provider API", examples=["gpt-4o", "llama-3.3-70b-versatile"])
    type: str = Field(default="standard", examples=["standard", "reasoning"])
    context_window: int | None = Field(default=None)
    tools: bool = Field(default=False)
    pricing_input: float | None = Field(default=None)   # $ per million tokens
    pricing_output: float | None = Field(default=None)
    available: bool | None = Field(default=None, description="None=unchecked, True=ok, False=failed")

class PingResult(BaseModel):
    model_id: str = Field(..., examples=["gpt-3.5-turbo", "gemma2-9b", "deepseek-chat"])
    provider: str = Field(..., examples=["deepseek", "anthropic", "moonshot"])
    latency_ms: float = Field(..., ge=0)
    status: str = Field(..., examples=["ok", "error"])
    error: str | None = Field(default=None)

def load_catalog() -> dict[str, ModelInfo]:
    path = Path(__file__).parents[4] / "configs" / "model_config.yaml"
    raw = yaml.safe_load(path.read_text())
    return {
        key: ModelInfo(
            model_id=key,                      # catalog key — what users pass to eval endpoints
            provider_model_id=data["model_id"],  # actual ID sent to provider API
            provider=data["provider"],
            type=data.get("type", "standard"),
            context_window=data.get("context_window"),
            tools=data.get("tools", False),
            pricing_input=data["pricing"]["input_per_1m"],
            pricing_output=data["pricing"]["output_per_1m"],
        )
        for key, data in raw["models"].items()
    }

# Loaded once at import time, lives in memory for the session
MODEL_CATALOG: dict[str, ModelInfo] = load_catalog()

@router.get("", response_model=list[ModelInfo])
async def list_models(
    provider: str | None = Query(default=None),
    available_only: bool = Query(default=False, description="Only return models confirmed working via ping"),
) -> list[ModelInfo]:
    """List all registered models. Optionally filter by provider name or availability."""
    models = list(MODEL_CATALOG.values())
    if provider:
        models = [m for m in models if m.provider == provider]
    if available_only:
        models = [m for m in models if m.available is True]
    return models

@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(404, detail=f"Model '{model_id}' not found")
    return MODEL_CATALOG[model_id]

@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str) -> None:
    if model_id not in MODEL_CATALOG:
        raise HTTPException(404, detail=f"Model '{model_id}' not found")
    MODEL_CATALOG.pop(model_id)

@router.post("", response_model=ModelInfo, status_code=201)
async def register_model(body: ModelInfo) -> ModelInfo:
    if body.provider not in MODEL_REGISTRY:
        raise HTTPException(400, detail=f"Unknown provider: {body.provider}")
    MODEL_CATALOG[body.model_id] = body
    return body

@router.post("/ping-all", response_model=list[PingResult])
async def ping_all_models() -> list[PingResult]:
    """Ping all registered models in parallel. Updates available status on each entry."""
    async def _ping(model_id: str, entry: ModelInfo) -> PingResult:
        try:
            provider = get_provider(entry.provider, entry.provider_model_id)
            resp = await provider.generate(
                GenerationRequest(prompt="ping", max_tokens=16, temperature=0.0)
            )
            entry.available = True
            return PingResult(model_id=model_id, provider=entry.provider,
                              latency_ms=resp.latency_ms, status="ok")
        except Exception as e:
            entry.available = False
            return PingResult(model_id=model_id, provider=entry.provider,
                              latency_ms=0, status="error", error=str(e))

    results = await asyncio.gather(*[_ping(mid, e) for mid, e in MODEL_CATALOG.items()])
    return list(results)


@router.post("/{model_id}/ping", response_model=PingResult)
async def ping_model(model_id: str) -> PingResult:
    entry = MODEL_CATALOG.get(model_id)
    if not entry:
        raise HTTPException(404, detail=f"Model '{model_id}' not found.")
    try:
        provider = get_provider(entry.provider, entry.provider_model_id)
        response = await provider.generate(
            GenerationRequest(prompt="ping", max_tokens=16, temperature=0.0)
        )
        entry.available = True
        return PingResult(
            model_id=model_id,
            provider=entry.provider,
            latency_ms=response.latency_ms,
            status="ok",
        )
    except Exception as e:
        entry.available = False
        return PingResult(
            model_id=model_id,
            provider=entry.provider,
            latency_ms=0,
            status="error",
            error=str(e),
        )