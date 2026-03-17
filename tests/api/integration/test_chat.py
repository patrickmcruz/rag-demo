"""Integration tests for POST /chat/stream (SSE)."""

import json
from pathlib import Path

import pytest

from tests.helpers import DummyChain

pytestmark = pytest.mark.integration


def _parse_sse(content: str) -> list[dict]:
    """Parse raw SSE response body into a list of event payloads."""
    events = []
    for block in content.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[6:]))
    return events


def test_chat_503_when_vectorstore_missing(api_client):
    """Vectorstore dir does not exist -> 503 before any streaming starts."""
    response = api_client.post("/chat/stream", json={"question": "hello"})
    assert response.status_code == 503
    assert "vectorstore" in response.json()["detail"].lower()


def test_chat_streams_sse_with_correct_content_type(api_client, mock_rag_app, monkeypatch):
    """With vectorstore present, response is text/event-stream."""
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    monkeypatch.setattr(mock_rag_app, "get_chain", lambda: DummyChain())

    response = api_client.post("/chat/stream", json={"question": "hello"})

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = _parse_sse(response.text)
    assert len(events) >= 1


def test_chat_last_event_is_done(api_client, mock_rag_app, monkeypatch):
    """The final SSE event must carry done=true and an empty token."""
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    monkeypatch.setattr(mock_rag_app, "get_chain", lambda: DummyChain())

    response = api_client.post("/chat/stream", json={"question": "test"})

    events = _parse_sse(response.text)
    assert events, "Expected at least one SSE event"
    last = events[-1]
    assert last["done"] is True
    assert last["token"] == ""


def test_chat_intermediate_events_not_done(api_client, mock_rag_app, monkeypatch):
    """All SSE events except the last must have done=false."""
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    monkeypatch.setattr(mock_rag_app, "get_chain", lambda: DummyChain())

    response = api_client.post("/chat/stream", json={"question": "hi"})

    events = _parse_sse(response.text)
    assert len(events) >= 2, "Expected token events + final done event"
    for event in events[:-1]:
        assert event["done"] is False


def test_chat_rejects_empty_question(api_client, mock_rag_app):
    """Empty question string fails Pydantic validation with 422."""
    mock_rag_app.config.vectorstore_dir = Path.cwd()
    response = api_client.post("/chat/stream", json={"question": ""})
    assert response.status_code == 422
