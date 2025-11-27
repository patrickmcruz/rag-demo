"""Lightweight integration test without external LLM/network.

Uses dummy components to verify wiring from RAGApplication through query.
"""

from pathlib import Path

from src.app import RAGApplication
from src.config import AppConfig


class DummyChain:
    def __init__(self):
        self.calls = []
        self.first = {"context": self}

    def invoke(self, question):
        self.calls.append(question)
        return f"answer:{question}"

    def get_relevant_documents(self, question):
        return []


def test_light_integration(monkeypatch, tmp_path):
    cfg = AppConfig(
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vs",
        model="llama3",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=10,
        chunk_overlap=1,
        top_k_documents=1,
        temperature=0.0,
        log_level="INFO",
    )
    app = RAGApplication(cfg)

    dummy_chain = DummyChain()
    monkeypatch.setattr(app, "build_chain", lambda: dummy_chain)

    response = app.query("hello")

    assert response.answer == "answer:hello"
    assert dummy_chain.calls == ["hello"]
