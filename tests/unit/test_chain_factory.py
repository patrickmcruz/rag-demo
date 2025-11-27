import pytest

from src.chain import RAGChainFactory, RAGChainBuilder


class DummyBuilder(RAGChainBuilder):
    def __init__(self):
        self.build_called_with = None

    def build(self, language: str = "pt"):
        self.build_called_with = language
        return "chain"


def test_factory_uses_builder(monkeypatch, tmp_path):
    dummy_builder = DummyBuilder()
    def fake_builder(vectorstore_path, model_name, embedding_model, temperature, top_k):
        assert vectorstore_path == str(tmp_path)
        assert model_name == "llama"
        assert embedding_model == "embed"
        assert temperature == 0.3
        assert top_k == 5
        return dummy_builder

    monkeypatch.setattr("src.chain.RAGChainBuilder", fake_builder)

    factory = RAGChainFactory(
        vectorstore_path=str(tmp_path),
        model_name="llama",
        embedding_model="embed",
        temperature=0.3,
        top_k=5,
        language="en",
    )
    chain = factory.create()

    assert chain == "chain"
    assert dummy_builder.build_called_with == "en"
