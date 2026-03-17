"""Unit tests for RAGApplication."""

import pytest

from src.app import RAGApplication
from tests.helpers import DummyFactory, DummyIngestionService, DummyIngestor, DummyQuery


def test_apply_overrides_resets_chain_and_ingestor(app_config, tmp_path):
    app = RAGApplication(app_config)

    app.apply_overrides(
        data_dir=tmp_path / "newdata",
        vectorstore_dir=tmp_path / "newvs",
        model="mistral",
        embedding_model="bge",
        chunk_size=200,
        chunk_overlap=20,
        top_k_documents=5,
        temperature=0.7,
    )

    assert app.config.data_dir == tmp_path / "newdata"
    assert app.config.vectorstore_dir == tmp_path / "newvs"
    assert app.config.model == "mistral"
    assert app.config.embedding_model == "bge"
    assert app.config.chunk_size == 200
    assert app.config.chunk_overlap == 20
    assert app.config.top_k_documents == 5
    assert app.config.temperature == 0.7
    assert app._rag_chain is None
    assert app._ingestion_service is None


def test_ingest_delegates(app_config, monkeypatch):
    app = RAGApplication(app_config)
    dummy_ingestor = DummyIngestor()
    dummy_service = DummyIngestionService(dummy_ingestor)
    monkeypatch.setattr(app, "_build_ingestion_service", lambda: dummy_service)

    result = app.ingest(data_dir="A", persist_dir="B", file_types=["pdf"])

    assert result == "vectorstore"
    assert dummy_service.calls == [("A", "B", ("pdf",))]
    assert dummy_ingestor.calls == [("A", "B", ("pdf",))]


def test_chain_build_and_cache(app_config, monkeypatch):
    app = RAGApplication(app_config)
    factory = DummyFactory()
    monkeypatch.setattr("src.app.RAGChainFactory", lambda **kwargs: factory)

    chain1 = app.get_chain()
    chain2 = app.get_chain()

    assert factory.created
    assert chain1 == "chain"
    assert chain2 == "chain"
    app.reset_chain()
    assert app._rag_chain is None


def test_query_uses_ragquery(app_config, monkeypatch):
    app = RAGApplication(app_config)
    monkeypatch.setattr(app, "get_chain", lambda: "chain")
    dummy_query = DummyQuery()
    monkeypatch.setattr("src.app.RAGQuery", lambda *args, **kwargs: dummy_query)

    res = app.query("hi", return_sources=False, verbose=True)

    assert res == {"answer": "ok"}
    assert dummy_query.questions == [("hi", False, True)]
