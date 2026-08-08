"""
Integration tests for Model Registry endpoints (Sprint 3).

Routes covered:
    GET    /api/v1/models                   list_models
    GET    /api/v1/models?provider=<name>   list_models (filtered)
    GET    /api/v1/models/{model_id}        get_model
    POST   /api/v1/models                   register_model
    DELETE /api/v1/models/{model_id}        delete_model
    POST   /api/v1/models/{model_id}/ping   ping_model
"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

BASE = "/api/v1/models"

# GET models
async def test_list_models_returns_list(client: AsyncClient):
    r = await client.get(BASE)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0

async def test_list_models_schema(client: AsyncClient):
    r = await client.get(BASE)
    model = r.json()[0]
    assert "model_id" in model
    assert "provider" in model
    assert "type" in model
    assert "context_window" in model
    assert "tools" in model
    assert "pricing_input" in model
    assert "pricing_output" in model

# Get models by provider name
async def test_list_models_filter_by_provider(client: AsyncClient):
    r = await client.get(BASE, params={"provider": "openai"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0
    assert all(m["provider"] == "openai" for m in data)

async def test_list_models_filter_unknown_provider_returns_empty(client: AsyncClient):
    r = await client.get(BASE, params={"provider": "nonexistent_provider"})
    assert r.status_code == 200
    assert r.json() == []

async def test_list_models_filter_anthropic(client: AsyncClient):
    r = await client.get(BASE, params={"provider": "anthropic"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0
    assert all(m["provider"] == "anthropic" for m in data)

# Get models by model name
async def test_get_model_known(client: AsyncClient):
    r = await client.get(f"{BASE}/gpt-4o-mini")
    assert r.status_code == 200
    data = r.json()
    assert data["model_id"] == "gpt-4o-mini"
    assert data["provider"] == "openai"
    assert data["pricing_input"] == pytest.approx(0.15)
    assert data["pricing_output"] == pytest.approx(0.60)

async def test_get_model_not_found(client: AsyncClient):
    r = await client.get(f"{BASE}/does-not-exist-xyz")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()

async def test_get_model_context_window_present(client: AsyncClient):
    r = await client.get(f"{BASE}/gpt-4o")
    assert r.status_code == 200
    assert r.json()["context_window"] == 128000

# Register a model
async def test_register_model_success(client: AsyncClient):
    payload = {
        "model_id": "test-model-xyz",
        "provider": "openai",
        "type": "standard",
        "context_window": 8192,
        "tools": False,
        "pricing_input": 1.00,
        "pricing_output": 2.00,
    }
    r = await client.post(BASE, json=payload)
    assert r.status_code == 201
    data = r.json()
    assert data["model_id"] == "test-model-xyz"
    assert data["provider"] == "openai"

async def test_register_model_appears_in_list(client: AsyncClient):
    payload = {
        "model_id": "test-list-check",
        "provider": "groq",
        "type": "standard",
        "context_window": 4096,
        "tools": False,
        "pricing_input": 0.05,
        "pricing_output": 0.10,
    }
    await client.post(BASE, json=payload)
    r = await client.get(BASE, params={"provider": "groq"})
    ids = [m["model_id"] for m in r.json()]
    assert "test-list-check" in ids

async def test_register_model_unknown_provider_returns_400(client: AsyncClient):
    payload = {
        "model_id": "bad-provider-model",
        "provider": "fakeprovider",
        "type": "standard",
        "context_window": None,
        "tools": False,
        "pricing_input": None,
        "pricing_output": None,
    }
    r = await client.post(BASE, json=payload)
    assert r.status_code == 400
    assert "unknown provider" in r.json()["detail"].lower()

# Delete a model
async def test_delete_model_success(client: AsyncClient):
    # Register first so delete has something to remove
    payload = {
        "model_id": "to-be-deleted",
        "provider": "mistral",
        "type": "standard",
        "context_window": 32768,
        "tools": False,
        "pricing_input": 0.10,
        "pricing_output": 0.30,
    }
    await client.post(BASE, json=payload)
    r = await client.delete(f"{BASE}/to-be-deleted")
    assert r.status_code == 204

async def test_delete_model_gone_afterwards(client: AsyncClient):
    payload = {
        "model_id": "delete-and-verify",
        "provider": "anthropic",
        "type": "standard",
        "context_window": 200000,
        "tools": True,
        "pricing_input": 3.00,
        "pricing_output": 15.00,
    }
    await client.post(BASE, json=payload)
    await client.delete(f"{BASE}/delete-and-verify")
    r = await client.get(f"{BASE}/delete-and-verify")
    assert r.status_code == 404

async def test_delete_model_not_found(client: AsyncClient):
    r = await client.delete(f"{BASE}/never-existed")
    assert r.status_code == 404

# Ping a model
async def test_ping_model_success(client: AsyncClient):
    from evalforge.providers.base import GenerationResponse
    mock_response = GenerationResponse(
        model="gpt-4o-mini",
        text="ok",
        input_tokens=3,
        output_tokens=1,
        latency_ms=123.4,
        finish_reason="stop",
        raw={},
    )
    with patch(
        "app.api.v1.endpoints.models.get_provider"
    ) as mock_get:
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value=mock_response)
        mock_get.return_value = mock_provider

        r = await client.post(f"{BASE}/gpt-4o-mini/ping")

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_id"] == "gpt-4o-mini"
    assert data["provider"] == "openai"
    assert data["latency_ms"] == pytest.approx(123.4)
    assert data["error"] is None

async def test_ping_model_provider_error_returns_error_status(client: AsyncClient):
    with patch(
        "app.api.v1.endpoints.models.get_provider"
    ) as mock_get:
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=Exception("connection refused"))
        mock_get.return_value = mock_provider

        r = await client.post(f"{BASE}/gpt-4o-mini/ping")

    assert r.status_code == 200          # endpoint returns 200 with error field, not 5xx
    data = r.json()
    assert data["status"] == "error"
    assert "connection refused" in data["error"]

async def test_ping_model_not_in_catalog_returns_404(client: AsyncClient):
    r = await client.post(f"{BASE}/nonexistent-model-abc/ping")
    assert r.status_code == 404