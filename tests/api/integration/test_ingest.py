"""Integration tests for POST /ingest and GET /ingest/{job_id}."""

import pytest

pytestmark = pytest.mark.integration

import time


def test_trigger_ingest_returns_202(api_client, mock_rag_app, monkeypatch):
    monkeypatch.setattr(mock_rag_app, "ingest", lambda **kwargs: None)
    response = api_client.post("/ingest", json={})
    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert "message" in body


def test_get_ingest_status_known_job(api_client, mock_job_store):
    job = mock_job_store.create()
    response = api_client.get(f"/ingest/{job.job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job.job_id
    assert body["status"] == "queued"


def test_get_ingest_status_not_found(api_client):
    response = api_client.get("/ingest/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_trigger_ingest_conflict_when_running(api_client, mock_rag_app, mock_job_store, monkeypatch):
    """Should return 409 if a job is already running."""
    from src.api.tasks import JobStatus

    monkeypatch.setattr(mock_rag_app, "ingest", lambda **kwargs: None)

    # Create a running job manually
    job = mock_job_store.create()
    job.status = JobStatus.RUNNING
    mock_job_store.update(job)

    response = api_client.post("/ingest", json={})
    assert response.status_code == 409


def test_ingest_request_validates_chunk_size(api_client):
    response = api_client.post("/ingest", json={"chunk_size": 0})
    assert response.status_code == 422


def test_ingest_request_validates_chunk_overlap(api_client):
    response = api_client.post("/ingest", json={"chunk_overlap": -1})
    assert response.status_code == 422
