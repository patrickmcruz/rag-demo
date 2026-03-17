"""Integration tests for GET /health and GET /health/ready."""

import pytest

pytestmark = pytest.mark.integration


def test_liveness_returns_200(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "timestamp" in body


def test_liveness_version_matches_package(api_client):
    from src import __version__

    response = api_client.get("/health")
    assert response.json()["version"] == __version__


def test_readiness_503_when_vectorstore_missing(api_client):
    """Default tmp_path fixture does not create vectorstore dir."""
    response = api_client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["vectorstore_exists"] is False
    assert body["chain_ready"] is False


def test_readiness_200_when_vectorstore_exists(api_client, mock_rag_app):
    vs_dir = mock_rag_app.config.vectorstore_dir
    vs_dir.mkdir(parents=True, exist_ok=True)

    response = api_client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["vectorstore_exists"] is True
