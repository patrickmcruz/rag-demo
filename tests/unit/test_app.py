from pathlib import Path

import pytest

from src.app import RAGApplication
from src.config import AppConfig


class DummyIngestor:
    def __init__(self):
        self.calls = []

    def ingest(self, data_dir, persist_dir, file_types=None):
        self.calls.append((data_dir, persist_dir, tuple(file_types) if file_types else None))
        return "vectorstore"


class DummyIngestionService:
    def __init__(self, ingestor):
        self.ingestor = ingestor
        self.calls = []

    def run(self, data_dir, persist_dir, file_types=None):
        self.calls.append((data_dir, persist_dir, tuple(file_types) if file_types else None))
        return self.ingestor.ingest(data_dir, persist_dir, file_types)


class DummyFactory:
    def __init__(self):
        self.created = False

    def create(self):
        self.created = True
        return "chain"


class DummyQuery:
    def __init__(self):
        self.questions = []

    def query(self, question, return_sources=True, verbose=False):
        self.questions.append((question, return_sources, verbose))
        return {"answer": "ok"}


def make_config(tmp_path):
    return AppConfig(
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vs",
        model="llama3",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=10,
        top_k_documents=3,
        temperature=0.0,
        log_level="INFO",
    )


def test_apply_overrides_resets_chain_and_ingestor(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    app = RAGApplication(cfg)

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


def test_ingest_delegates(monkeypatch, tmp_path):
    cfg = make_config(tmp_path)
    app = RAGApplication(cfg)
    dummy_ingestor = DummyIngestor()
    dummy_service = DummyIngestionService(dummy_ingestor)
    monkeypatch.setattr(app, "_build_ingestion_service", lambda: dummy_service)

    result = app.ingest(data_dir="A", persist_dir="B", file_types=["pdf"])

    assert result == "vectorstore"
    assert dummy_service.calls == [("A", "B", ("pdf",))]
    assert dummy_ingestor.calls == [("A", "B", ("pdf",))]


def test_chain_build_and_cache(monkeypatch, tmp_path):
    cfg = make_config(tmp_path)
    app = RAGApplication(cfg)
    factory = DummyFactory()
    monkeypatch.setattr("src.app.RAGChainFactory", lambda **kwargs: factory)

    chain1 = app.get_chain()
    chain2 = app.get_chain()

    assert factory.created
    assert chain1 == "chain"
    assert chain2 == "chain"
    app.reset_chain()
    assert app._rag_chain is None


def test_query_uses_ragquery(monkeypatch, tmp_path):
    cfg = make_config(tmp_path)
    app = RAGApplication(cfg)
    monkeypatch.setattr(app, "get_chain", lambda: "chain")
    dummy_query = DummyQuery()
    monkeypatch.setattr("src.app.RAGQuery", lambda *args, **kwargs: dummy_query)

    res = app.query("hi", return_sources=False, verbose=True)

    assert res == {"answer": "ok"}
    assert dummy_query.questions == [("hi", False, True)]
