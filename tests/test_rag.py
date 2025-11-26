"""Tests for RAG system components."""

from pathlib import Path
import shutil
import time
import pytest

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_VECTORSTORE_DIR = Path(__file__).parent / "test_vectorstore"


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory with sample documents."""
    TEST_DATA_DIR.mkdir(exist_ok=True)

    sample_txt = TEST_DATA_DIR / "sample.txt"
    sample_txt.write_text(
        "Python é uma linguagem de programação de alto nível. "
        "É amplamente utilizada em ciência de dados, machine learning e desenvolvimento web. "
        "Python foi criado por Guido van Rossum e lançado em 1991."
    )

    yield TEST_DATA_DIR

    # Cleanup - with retry for Windows
    if TEST_DATA_DIR.exists():
        try:
            shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
        except Exception:
            pass


@pytest.fixture(scope="session")
def test_vectorstore_dir():
    """Create test vectorstore directory."""
    TEST_VECTORSTORE_DIR.mkdir(exist_ok=True)
    yield TEST_VECTORSTORE_DIR

    # Cleanup - with retry for Windows file locks
    if TEST_VECTORSTORE_DIR.exists():
        try:
            time.sleep(0.5)  # Give time for file handles to close
            shutil.rmtree(TEST_VECTORSTORE_DIR, ignore_errors=True)
        except Exception:
            pass


class TestDocumentIngestor:
    """Tests for document ingestion."""

    def test_import_ingestor(self):
        """Test that DocumentIngestor can be imported."""
        from src.ingest import DocumentIngestor

        assert DocumentIngestor is not None

    def test_ingestor_initialization(self):
        """Test DocumentIngestor initialization."""
        from src.ingest import DocumentIngestor

        ingestor = DocumentIngestor(
            embedding_model="all-MiniLM-L6-v2", chunk_size=100, chunk_overlap=10
        )

        assert ingestor.embedding_model == "all-MiniLM-L6-v2"
        assert ingestor.chunk_size == 100
        assert ingestor.chunk_overlap == 10

    def test_load_documents(self, test_data_dir):
        """Test document loading."""
        from src.ingest import DocumentIngestor

        ingestor = DocumentIngestor()
        docs = ingestor.load_documents(str(test_data_dir), file_types=["txt"])

        assert len(docs) > 0
        assert all(hasattr(doc, "page_content") for doc in docs)

    def test_load_documents_invalid_dir(self):
        """Test loading from non-existent directory."""
        from src.ingest import DocumentIngestor

        ingestor = DocumentIngestor()

        with pytest.raises(ValueError, match="does not exist"):
            ingestor.load_documents("/non/existent/path")

    def test_split_documents(self, test_data_dir):
        """Test document splitting."""
        from src.ingest import DocumentIngestor

        ingestor = DocumentIngestor(chunk_size=50, chunk_overlap=10)
        docs = ingestor.load_documents(str(test_data_dir), file_types=["txt"])
        splits = ingestor.split_documents(docs)

        assert len(splits) >= len(docs)
        assert all(len(split.page_content) <= 60 for split in splits)

    @pytest.mark.slow
    def test_create_vectorstore(self, test_data_dir, test_vectorstore_dir):
        """Test vectorstore creation (slow test - requires embedding model)."""
        from src.ingest import DocumentIngestor

        ingestor = DocumentIngestor()
        docs = ingestor.load_documents(str(test_data_dir), file_types=["txt"])
        splits = ingestor.split_documents(docs)

        vectorstore = ingestor.create_vectorstore(
            splits, str(test_vectorstore_dir / "test_vs")
        )

        assert vectorstore is not None
        assert (test_vectorstore_dir / "test_vs").exists()


class TestRAGChain:
    """Tests for RAG chain."""

    def test_import_chain_builder(self):
        """Test that RAGChainBuilder can be imported."""
        from src.chain import RAGChainBuilder

        assert RAGChainBuilder is not None

    def test_import_create_rag_chain(self):
        """Test that create_rag_chain can be imported."""
        from src.chain import create_rag_chain

        assert create_rag_chain is not None

    @pytest.mark.slow
    def test_chain_builder_invalid_path(self):
        """Test chain builder with invalid vectorstore path."""
        from src.chain import RAGChainBuilder

        with pytest.raises(ValueError, match="Vector store not found"):
            RAGChainBuilder(
                vectorstore_path="/non/existent/vectorstore", model_name="llama3"
            )


class TestRAGQuery:
    """Tests for RAG query interface."""

    def test_import_rag_query(self):
        """Test that RAGQuery can be imported."""
        from src.query import RAGQuery

        assert RAGQuery is not None

    def test_import_rag_response(self):
        """Test that RAGResponse can be imported."""
        from src.query import RAGResponse

        assert RAGResponse is not None

    def test_rag_response_creation(self):
        """Test RAGResponse dataclass creation."""
        from src.query import RAGResponse

        response = RAGResponse(
            answer="Test answer",
            sources=[],
            query="Test question",
            response_time=1.5,
            model_name="llama3",
        )

        assert response.answer == "Test answer"
        assert response.query == "Test question"
        assert response.response_time == 1.5
        assert response.model_name == "llama3"

    def test_rag_response_format_sources_empty(self):
        """Test source formatting with no sources."""
        from src.query import RAGResponse

        response = RAGResponse(
            answer="Test",
            sources=[],
            query="Test",
            response_time=1.0,
            model_name="llama3",
        )

        formatted = response.format_sources()
        assert "Nenhuma fonte" in formatted


class TestIntegration:
    """Integration tests (require full setup)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_pipeline(self, test_data_dir, test_vectorstore_dir):
        """Test complete RAG pipeline from ingestion to query."""
        from src.ingest import ingest_documents
        from src.chain import create_rag_chain

        # Step 1: Ingest documents
        vectorstore_path = str(test_vectorstore_dir / "e2e_test")
        vectorstore = ingest_documents(
            data_dir=str(test_data_dir),
            persist_dir=vectorstore_path,
            file_types=["txt"],
        )

        assert vectorstore is not None

        # Step 2: Create RAG chain (would require Ollama running)
        # chain = create_rag_chain(vectorstore_path)
        # response = chain.invoke("O que é Python?")
        # assert response is not None


def test_project_structure():
    """Test that project structure is correct."""
    project_root = Path(__file__).parent.parent

    # Check required directories
    assert (project_root / "src").exists()
    assert (project_root / "data").exists()
    assert (project_root / "vectorstore").exists()
    assert (project_root / "tests").exists()

    # Check required files
    assert (project_root / "requirements.txt").exists()
    assert (project_root / ".env.example").exists()
    assert (project_root / "src" / "__init__.py").exists()


def test_requirements_file():
    """Test that requirements.txt contains essential packages."""
    project_root = Path(__file__).parent.parent
    requirements = (project_root / "requirements.txt").read_text()

    essential_packages = [
        "langchain",
        "chromadb",
        "sentence-transformers",
        "pytest",
    ]

    for package in essential_packages:
        assert package in requirements.lower(), f"Missing {package} in requirements.txt"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
