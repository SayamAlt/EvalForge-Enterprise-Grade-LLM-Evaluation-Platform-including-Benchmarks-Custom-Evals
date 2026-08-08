"""
Integration tests for the Dataset API.

These tests exercise the full request → service → DB path using a real
(test) database and in-memory HTTP client. No HuggingFace network calls.

CONCEPT: Integration tests verify wiring
─────────────────────────────────────────
Unit tests verify loader logic in isolation.
Integration tests verify that routes, service, and DB all connect correctly:
  - Route accepts the right HTTP shape
  - Service writes and reads from the test DB
  - HTTP status codes are correct (201, 204, 404, 422)
"""

import io
import uuid

import pytest

CSV_CONTENT = b"question,answer\nWhat is 2+2?,4\nCapital of France?,Paris\n"
JSONL_CONTENT = b'{"question": "Q1", "answer": "A1"}\n{"question": "Q2", "answer": "A2"}\n'


@pytest.mark.asyncio
async def test_upload_csv_returns_201(client):
    response = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Test CSV", "task_type": "qa"},
        files={"file": ("test.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Test CSV"
    assert body["source_type"] == "csv"
    assert body["status"] == "ready"
    assert body["num_samples"] == 2


@pytest.mark.asyncio
async def test_upload_jsonl_returns_201(client):
    response = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Test JSONL", "task_type": "qa"},
        files={"file": ("test.jsonl", io.BytesIO(JSONL_CONTENT), "application/jsonl")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["source_type"] == "jsonl"
    assert body["num_samples"] == 2


@pytest.mark.asyncio
async def test_upload_unsupported_format_returns_500(client):
    response = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Bad", "task_type": "qa"},
        files={"file": ("data.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_list_datasets_empty(client):
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0


@pytest.mark.asyncio
async def test_list_datasets_after_upload(client):
    client.post(
        "/api/v1/datasets/upload",
        data={"name": "DS1", "task_type": "qa"},
        files={"file": ("t.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
    )
    response = client.get("/api/v1/datasets")
    assert response.json()["total"] == 1


@pytest.mark.asyncio
async def test_get_dataset_not_found(client):
    response = client.get(f"/api/v1/datasets/{uuid.uuid4()}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_dataset_by_id(client):
    upload = client.post(
        "/api/v1/datasets/upload",
        data={"name": "My DS", "task_type": "generation"},
        files={"file": ("t.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
    )
    dataset_id = upload.json()["id"]
    response = client.get(f"/api/v1/datasets/{dataset_id}")
    assert response.status_code == 200
    assert response.json()["id"] == dataset_id


@pytest.mark.asyncio
async def test_preview_dataset(client):
    upload = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Preview DS", "task_type": "qa"},
        files={"file": ("t.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
    )
    dataset_id = upload.json()["id"]
    response = client.get(f"/api/v1/datasets/{dataset_id}/preview?n=2")
    assert response.status_code == 200
    body = response.json()
    assert body["preview_count"] == 2
    assert body["samples"][0]["input"] == "What is 2+2?"
    assert body["samples"][0]["reference"] == "4"


@pytest.mark.asyncio
async def test_delete_dataset(client):
    upload = client.post(
        "/api/v1/datasets/upload",
        data={"name": "Delete Me", "task_type": "qa"},
        files={"file": ("t.csv", io.BytesIO(CSV_CONTENT), "text/csv")},
    )
    dataset_id = upload.json()["id"]

    assert client.delete(f"/api/v1/datasets/{dataset_id}").status_code == 204
    assert client.get(f"/api/v1/datasets/{dataset_id}").status_code == 404


@pytest.mark.asyncio
async def test_register_hf_missing_required_field_returns_422(client):
    """Missing hf_dataset_id → Pydantic validation error → 422."""
    response = client.post(
        "/api/v1/datasets/huggingface",
        json={"name": "MMLU", "task_type": "multiple_choice"},
    )
    assert response.status_code == 422
