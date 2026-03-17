"""Lightweight integration test — wiring from RAGApplication to RAGResponse.

Uses DummyChain to avoid any real LLM, ChromaDB, or embedding calls.
"""

import pytest

from src.core.rag_service import RAGApplication
from tests.helpers import DummyChain


@pytest.mark.integration
def test_light_integration(app_config, monkeypatch):
    app = RAGApplication(app_config)

    dummy_chain = DummyChain()
    monkeypatch.setattr(app, "build_chain", lambda: dummy_chain)

    response = app.query("hello")

    assert response.answer == "answer:hello"
    assert dummy_chain.calls == ["hello"]
