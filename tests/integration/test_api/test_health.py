""" Integration test: health endpoint. """
import pytest

@pytest.mark.asyncio
async def test_health_returns_ok(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "EvalForge"
    assert data["uptime_seconds"] >= 0

@pytest.mark.asyncio
async def test_unimplemented_endpoints_return_501(client):
    for path in ["/api/v1/datasets", "/api/v1/models", "/api/v1/experiments"]:
        response = await client.get(path)
        assert response.status_code == 501, f"{path} should be 501"