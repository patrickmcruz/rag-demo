"""Document ingestion module for RAG pipeline.

This module handles loading, splitting, embedding, and indexing documents
into a vector store for retrieval-augmented generation.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handles document ingestion and indexing for RAG."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Initialize the document ingestor.

        Args:
            embedding_model: Name of the HuggingFace embedding model
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding: Optional[HuggingFaceEmbeddings] = None

    def _get_embedding(self) -> HuggingFaceEmbeddings:
        """Lazy load embedding model."""
        if self.embedding is None:
            logger.info(f"Loading embedding model: {self.embedding_model}")
            self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return self.embedding

    def load_documents(
        self, data_dir: str, file_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Load documents from directory with support for multiple file types.

        Args:
            data_dir: Directory containing documents
            file_types: List of file extensions to load (e.g., ['txt', 'pdf', 'md'])

        Returns:
            List of loaded documents

        Raises:
            ValueError: If data directory doesn't exist or is empty
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        if file_types is None:
            file_types = ["txt", "pdf", "md"]

        all_docs = []

        for file_type in file_types:
            try:
                glob_pattern = f"**/*.{file_type}"
                logger.info(f"Loading {file_type} files from {data_dir}")

                loader = DirectoryLoader(
                    data_dir,
                    glob=glob_pattern,
                    loader_cls=self._get_loader_for_type(file_type),
                    show_progress=True,
                )
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} {file_type} documents")
            except Exception as e:
                logger.warning(f"Error loading {file_type} files: {e}")

        if not all_docs:
            raise ValueError(f"No documents found in {data_dir}")

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    def _get_loader_for_type(self, file_type: str):
        """Get appropriate loader class for file type."""
        loaders = {
            "txt": TextLoader,
            "pdf": PyPDFLoader,
            "md": UnstructuredMarkdownLoader,
        }
        return loaders.get(file_type, TextLoader)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} chunks")
        return splits  # type: ignore[no-any-return]

    def create_vectorstore(self, documents: List[Document], persist_dir: str) -> Chroma:
        """Create and persist vector store from documents.

        Args:
            documents: List of document chunks
            persist_dir: Directory to persist the vector store

        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store with {len(documents)} documents")

        # Ensure persist directory exists
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        embedding = self._get_embedding()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_dir,
        )

        logger.info(f"[OK] Vector store created and persisted to {persist_dir}")
        return vectorstore

    def ingest(
        self,
        data_dir: str,
        persist_dir: str,
        file_types: Optional[List[str]] = None,
    ) -> Chroma:
        """Complete ingestion pipeline: load, split, embed, and index.

        Args:
            data_dir: Directory containing source documents
            persist_dir: Directory to persist the vector store
            file_types: List of file extensions to load

        Returns:
            Chroma vector store instance
        """
        try:
            # Load documents
            docs = self.load_documents(data_dir, file_types)

            # Split into chunks
            splits = self.split_documents(docs)

            # Create and persist vector store
            vectorstore = self.create_vectorstore(splits, persist_dir)

            logger.info(
                f"[SUCCESS] Successfully ingested {len(docs)} documents "
                f"({len(splits)} chunks) into {persist_dir}"
            )
            return vectorstore

        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            raise


def ingest_documents(
    data_dir: str,
    persist_dir: str,
    file_types: Optional[List[str]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Chroma:
    """Convenience function for document ingestion.

    Args:
        data_dir: Directory containing source documents
        persist_dir: Directory to persist the vector store
        file_types: List of file extensions to load (default: ['txt', 'pdf', 'md'])
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between consecutive chunks
        embedding_model: Embedding model to use

    Returns:
        Chroma vector store instance
    """
    ingestor = DocumentIngestor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    return ingestor.ingest(data_dir, persist_dir, file_types)
