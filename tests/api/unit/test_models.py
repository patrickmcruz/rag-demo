"""Unit tests for Pydantic request/response schema validation."""

import pytest
from pydantic import ValidationError

from src.api.models import ChatRequest, IngestRequest, QueryRequest


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


def test_query_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_rejects_oversized_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 2001)


def test_query_request_rejects_invalid_language():
    with pytest.raises(ValidationError):
        QueryRequest(question="ok", language="es")


def test_query_request_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(question="ok", top_k=0)


def test_query_request_rejects_top_k_over_limit():
    with pytest.raises(ValidationError):
        QueryRequest(question="ok", top_k=51)


def test_query_request_rejects_negative_temperature():
    with pytest.raises(ValidationError):
        QueryRequest(question="ok", temperature=-0.1)


def test_query_request_defaults():
    r = QueryRequest(question="hello")
    assert r.return_sources is True
    assert r.language == "pt"
    assert r.top_k is None
    assert r.temperature is None


def test_query_request_accepts_both_languages():
    assert QueryRequest(question="ok", language="pt").language == "pt"
    assert QueryRequest(question="ok", language="en").language == "en"


# ---------------------------------------------------------------------------
# IngestRequest
# ---------------------------------------------------------------------------


def test_ingest_request_all_fields_optional():
    r = IngestRequest()
    assert r.data_dir is None
    assert r.chunk_size is None
    assert r.chunk_overlap is None
    assert r.file_types is None
    assert r.embedding_model is None


def test_ingest_request_rejects_zero_chunk_size():
    with pytest.raises(ValidationError):
        IngestRequest(chunk_size=0)


def test_ingest_request_rejects_negative_chunk_overlap():
    with pytest.raises(ValidationError):
        IngestRequest(chunk_overlap=-1)


def test_ingest_request_rejects_oversized_chunk():
    with pytest.raises(ValidationError):
        IngestRequest(chunk_size=10001)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------


def test_chat_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        ChatRequest(question="")


def test_chat_request_defaults_to_portuguese():
    r = ChatRequest(question="test")
    assert r.language == "pt"
