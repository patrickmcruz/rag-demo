"""Integration tests for GET /info."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_info_returns_200(api_client):
    response = api_client.get("/info")
    assert response.status_code == 200


def test_info_reflects_mock_config(api_client, mock_rag_app):
    response = api_client.get("/info")
    body = response.json()
    cfg = mock_rag_app.config

    assert body["model"] == cfg.model
    assert body["embedding_model"] == cfg.embedding_model
    assert body["chunk_size"] == cfg.chunk_size
    assert body["chunk_overlap"] == cfg.chunk_overlap
    assert body["top_k_documents"] == cfg.top_k_documents
    assert body["use_gpu"] == cfg.use_gpu


def test_info_vectorstore_false_when_dir_missing(api_client):
    response = api_client.get("/info")
    body = response.json()
    assert body["vectorstore_exists"] is False
    assert body["data_dir_exists"] is False


def test_info_vectorstore_true_when_dir_exists(api_client, mock_rag_app):
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    mock_rag_app.config.data_dir = Path.cwd() / "data"

    response = api_client.get("/info")
    body = response.json()
    assert body["vectorstore_exists"] is True
    assert body["data_dir_exists"] is True


def test_info_file_counts_present(api_client, mock_rag_app):
    mock_rag_app.config.data_dir = Path.cwd() / "data"

    response = api_client.get("/info")
    body = response.json()
    counts = body["data_file_counts"]
    assert "txt" in counts
    assert "pdf" in counts
    assert "md" in counts
