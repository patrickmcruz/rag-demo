"""Application orchestrator for the RAG demo."""

import logging
from pathlib import Path
from typing import Optional, List

from src.config import AppConfig
from src.ingest import DocumentIngestor, IngestionService
from src.chain import RAGChainFactory
from src.query import RAGQuery, InteractiveQueryCLI

logger = logging.getLogger(__name__)


class RAGApplication:
    """High-level application object orchestrating ingestion and querying."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._ingestion_service: IngestionService | None = None
        self._rag_chain = None

    def apply_overrides(
        self,
        data_dir: Path | None = None,
        vectorstore_dir: Path | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k_documents: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """Apply runtime overrides and reset components when needed."""
        if data_dir is not None:
            self.config.data_dir = data_dir
        if vectorstore_dir is not None:
            self.config.vectorstore_dir = vectorstore_dir
            self.reset_chain()
        if model is not None:
            self.config.model = model
            self.reset_chain()
        if embedding_model is not None:
            self.config.embedding_model = embedding_model
            self.reset_chain()
            self._ingestion_service = None
        if chunk_size is not None:
            self.config.chunk_size = chunk_size
            self._ingestion_service = None
        if chunk_overlap is not None:
            self.config.chunk_overlap = chunk_overlap
            self._ingestion_service = None
        if top_k_documents is not None:
            self.config.top_k_documents = top_k_documents
            self.reset_chain()
        if temperature is not None:
            self.config.temperature = temperature
            self.reset_chain()

    def _build_ingestion_service(self) -> IngestionService:
        return IngestionService(
            DocumentIngestor(
                embedding_model=self.config.embedding_model,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        )

    def _get_ingestion_service(self) -> IngestionService:
        if self._ingestion_service is None:
            self._ingestion_service = self._build_ingestion_service()
        return self._ingestion_service

    def ingest(
        self,
        data_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
        file_types: Optional[List[str]] = None,
    ):
        """Run ingestion pipeline with optional overrides."""
        data_path = str(Path(data_dir or self.config.data_dir))
        persist_path = str(Path(persist_dir or self.config.vectorstore_dir))
        logger.info("Running ingestion through RAGApplication")
        return self._get_ingestion_service().run(
            data_dir=data_path,
            persist_dir=persist_path,
            file_types=file_types,
        )

    def build_chain(self):
        """Build and cache the RAG chain."""
        factory = RAGChainFactory(
            vectorstore_path=str(self.config.vectorstore_dir),
            model_name=self.config.model,
            embedding_model=self.config.embedding_model,
            temperature=self.config.temperature,
            top_k=self.config.top_k_documents,
            language="pt",
        )
        self._rag_chain = factory.create()
        return self._rag_chain

    def get_chain(self):
        """Return existing chain or build a new one."""
        if self._rag_chain is None:
            return self.build_chain()
        return self._rag_chain

    def reset_chain(self):
        """Drop cached chain to force rebuild."""
        self._rag_chain = None

    def query(self, question: str, return_sources: bool = True, verbose: bool = False):
        """Perform a single query using the configured chain."""
        chain = self.get_chain()
        return RAGQuery(chain, model_name=self.config.model).query(
            question, return_sources=return_sources, verbose=verbose
        )

    def interactive_cli(self):
        """Launch interactive CLI."""
        chain = self.get_chain()
        cli = InteractiveQueryCLI(RAGQuery(chain, model_name=self.config.model))
        cli.run()
