"""Integration tests for POST /query."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.core.query_service import RAGResponse

pytestmark = pytest.mark.integration


def _make_dummy_response(question: str = "test question") -> RAGResponse:
    return RAGResponse(
        answer="mocked answer",
        sources=[
            Document(
                page_content="source content here",
                metadata={"source": "file.pdf", "page": 1},
            )
        ],
        query=question,
        response_time=0.1,
        model_name="llama3",
    )


def test_query_503_when_vectorstore_missing(api_client):
    """Vectorstore dir does not exist -> 503."""
    response = api_client.post("/query", json={"question": "hello"})
    assert response.status_code == 503
    assert "vectorstore" in response.json()["detail"].lower()


def test_query_happy_path(api_client, mock_rag_app, monkeypatch):
    """With vectorstore present and mocked query, endpoint returns 200."""
    mock_rag_app.config.vectorstore_dir = Path.cwd()

    dummy = _make_dummy_response()
    monkeypatch.setattr(mock_rag_app, "query", lambda *a, **kw: dummy)

    response = api_client.post(
        "/query",
        json={"question": "What is in the document?", "return_sources": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "mocked answer"
    assert body["model_name"] == "llama3"
    assert isinstance(body["sources"], list)
    assert body["sources"][0]["source"] == "file.pdf"
    assert body["sources"][0]["page"] == 1


def test_query_empty_sources_when_return_sources_false(api_client, mock_rag_app, monkeypatch):
    mock_rag_app.config.vectorstore_dir = Path.cwd()

    dummy = RAGResponse(
        answer="answer",
        sources=[],
        query="q",
        response_time=0.05,
        model_name="llama3",
    )
    monkeypatch.setattr(mock_rag_app, "query", lambda *a, **kw: dummy)

    response = api_client.post(
        "/query",
        json={"question": "test", "return_sources": False},
    )
    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_query_rejects_empty_question(api_client, mock_rag_app):
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    response = api_client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_rejects_oversized_question(api_client, mock_rag_app):
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    response = api_client.post("/query", json={"question": "x" * 2001})
    assert response.status_code == 422


def test_query_rejects_invalid_language(api_client, mock_rag_app):
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    response = api_client.post("/query", json={"question": "ok", "language": "es"})
    assert response.status_code == 422


def test_query_response_time_is_float(api_client, mock_rag_app, monkeypatch):
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    dummy = _make_dummy_response()
    monkeypatch.setattr(mock_rag_app, "query", lambda *a, **kw: dummy)

    response = api_client.post("/query", json={"question": "hello"})
    assert isinstance(response.json()["response_time"], float)
