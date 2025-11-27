from src.query import RAGQuery, RAGResponse, InteractiveQueryCLI


class FakeChain:
    def __init__(self):
        self.calls = []
        self.first = {"context": self}

    def invoke(self, question):
        self.calls.append(question)
        return f"answer:{question}"

    def get_relevant_documents(self, question):
        return []


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
    chain = FakeChain()
    rag = RAGQuery(chain, model_name="llama")
    resp = rag.query("hi", return_sources=True, verbose=True)
    assert resp.answer == "answer:hi"
    assert chain.calls == ["hi"]


def test_batch_query():
    chain = FakeChain()
    rag = RAGQuery(chain, model_name="llama")
    resps = rag.batch_query(["a", "b"])
    assert len(resps) == 2
    assert chain.calls == ["a", "b"]


def test_query_stats_and_clear():
    chain = FakeChain()
    rag = RAGQuery(chain, model_name="llama")
    rag.query("x")
    stats = rag.get_stats()
    assert stats["total_queries"] == 1
    rag.clear_history()
    assert rag.get_stats()["total_queries"] == 0


def test_query_empty_raises():
    chain = FakeChain()
    rag = RAGQuery(chain, model_name="llama")
    try:
        rag.query("")
    except ValueError:
        assert True
    else:
        assert False


def test_interactive_cli_commands(monkeypatch, capsys):
    # Simulate: stats -> clear -> sair
    inputs = iter(["stats", "clear", "sair"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    rag = RAGQuery(FakeChain(), model_name="llama")
    cli = InteractiveQueryCLI(rag)
    cli.run()
    out = capsys.readouterr().out
    assert "[RAG] Query Interface" in out
    assert "Histórico limpo" in out or "Total de consultas" in out
