"""Shared test doubles for the RAG Demo test suite.

All test files should import stubs from here instead of defining local
classes.  A single definition prevents drift when interfaces change.

Naming convention: Dummy* for stubs that record calls, no side-effects.
"""


class DummyIngestor:
    """Stub for DocumentIngestor — records calls, returns a fixed value."""

    def __init__(self):
        self.calls = []

    def ingest(self, data_dir, persist_dir, file_types=None):
        self.calls.append((data_dir, persist_dir, tuple(file_types or [])))
        return "vectorstore"


class DummyIngestionService:
    """Stub for IngestionService — delegates to an internal DummyIngestor."""

    def __init__(self, ingestor=None):
        self.ingestor = ingestor or DummyIngestor()
        self.calls = []

    def run(self, data_dir, persist_dir, file_types=None):
        self.calls.append((data_dir, persist_dir, tuple(file_types or [])))
        return self.ingestor.ingest(data_dir, persist_dir, file_types)


class DummyChain:
    """Stub for a LangChain RAG chain.

    Supports both sync (invoke) and async (astream) usage so it can be used
    in unit tests and in SSE endpoint integration tests.
    """

    def __init__(self):
        self.calls = []
        self.first = {"context": self}  # mirrors real chain structure for _get_sources

    def invoke(self, question):
        self.calls.append(question)
        return f"answer:{question}"

    def get_relevant_documents(self, question):
        return []

    async def astream(self, question):
        """Yields two tokens then stops — enough to verify SSE plumbing."""
        for token in [f"answer:{question}", ""]:
            yield token


class DummyFactory:
    """Stub for RAGChainFactory — records whether create() was called."""

    def __init__(self):
        self.created = False

    def create(self):
        self.created = True
        return "chain"


class DummyQuery:
    """Stub for RAGQuery — records questions, returns a fixed answer dict."""

    def __init__(self):
        self.questions = []

    def query(self, question, return_sources=True, verbose=False):
        self.questions.append((question, return_sources, verbose))
        return {"answer": "ok"}
