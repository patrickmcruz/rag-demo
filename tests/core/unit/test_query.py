"""Unit tests for RAGQuery and RAGResponse."""

import pytest

from src.core.query_service import RAGQuery, RAGResponse
from tests.helpers import DummyChain


def test_rag_response_format_sources_empty():
    resp = RAGResponse(
        answer="a",
        sources=[],
        query="q",
        response_time=1.0,
        model_name="m",
    )
    assert "Nenhuma fonte" in resp.format_sources()


def test_query_happy_path():
    chain = DummyChain()
    rag = RAGQuery(chain, model_name="llama")
    resp = rag.query("hi", return_sources=True, verbose=True)
    assert resp.answer == "answer:hi"
    assert chain.calls == ["hi"]


def test_batch_query():
    chain = DummyChain()
    rag = RAGQuery(chain, model_name="llama")
    resps = rag.batch_query(["a", "b"])
    assert len(resps) == 2
    assert chain.calls == ["a", "b"]


def test_query_stats_and_clear():
    chain = DummyChain()
    rag = RAGQuery(chain, model_name="llama")
    rag.query("x")
    stats = rag.get_stats()
    assert stats["total_queries"] == 1
    rag.clear_history()
    assert rag.get_stats()["total_queries"] == 0


def test_query_empty_raises():
    chain = DummyChain()
    rag = RAGQuery(chain, model_name="llama")
    with pytest.raises(ValueError):
        rag.query("")
