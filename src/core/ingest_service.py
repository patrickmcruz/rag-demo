"""Document ingestion module for RAG pipeline."""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.chain_factory import get_device

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Handles document ingestion and indexing for RAG."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_gpu: bool = False,
        gpu_device: int = 0,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.embedding: Optional[HuggingFaceEmbeddings] = None

    def _repair_text_encoding(self, text: str) -> str:
        """Repair common mojibake caused by UTF-8 decoded as Latin-1/CP1252."""
        suspicious_patterns = ("Ã", "â", "ð", "�")
        if not any(pattern in text for pattern in suspicious_patterns):
            return text

        for source_encoding in ("latin-1", "cp1252"):
            try:
                repaired = text.encode(source_encoding).decode("utf-8")
                if repaired != text:
                    return repaired
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue

        return text

    def _get_embedding(self) -> HuggingFaceEmbeddings:
        if self.embedding is None:
            device = get_device(self.use_gpu, self.gpu_device)
            logger.info(f"Loading embedding model: {self.embedding_model} on {device}")
            self.embedding = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 512},
            )
            logger.info("Embedding model loaded with batch_size=512 optimization")
        return self.embedding

    def load_documents(
        self, data_dir: str, file_types: Optional[List[str]] = None
    ) -> List[Document]:
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
                loader_kwargs = {}
                if file_type == "txt":
                    loader_kwargs = {"autodetect_encoding": True, "encoding": "utf-8"}

                loader = DirectoryLoader(
                    data_dir,
                    glob=glob_pattern,
                    loader_cls=self._get_loader_for_type(file_type),
                    loader_kwargs=loader_kwargs,
                    show_progress=True,
                )
                docs = loader.load()
                for doc in docs:
                    doc.page_content = self._repair_text_encoding(doc.page_content)
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} {file_type} documents")
            except Exception as exc:
                logger.warning(f"Error loading {file_type} files: {exc}")

        if not all_docs:
            raise ValueError(f"No documents found in {data_dir}")

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    def _get_loader_for_type(self, file_type: str):
        loaders = {
            "txt": TextLoader,
            "pdf": PyPDFLoader,
            "md": UnstructuredMarkdownLoader,
        }
        return loaders.get(file_type, TextLoader)

    def split_documents(self, documents: List[Document]) -> List[Document]:
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
        logger.info(f"Creating vector store with {len(documents)} documents")
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
        try:
            docs = self.load_documents(data_dir, file_types)
            splits = self.split_documents(docs)
            vectorstore = self.create_vectorstore(splits, persist_dir)
            logger.info(
                f"[SUCCESS] Successfully ingested {len(docs)} documents "
                f"({len(splits)} chunks) into {persist_dir}"
            )
            return vectorstore
        except Exception as exc:
            logger.error(f"Error during ingestion: {exc}")
            raise


class IngestionService:
    """Higher-level ingestion service that orchestrates the ingestion pipeline."""

    def __init__(self, ingestor: DocumentIngestor):
        self.ingestor = ingestor

    def run(
        self,
        data_dir: str,
        persist_dir: str,
        file_types: Optional[List[str]] = None,
    ) -> Chroma:
        return self.ingestor.ingest(data_dir, persist_dir, file_types)


def ingest_documents(
    data_dir: str,
    persist_dir: str,
    file_types: Optional[List[str]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Chroma:
    """Convenience function for document ingestion."""
    ingestor = DocumentIngestor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    return ingestor.ingest(data_dir, persist_dir, file_types)
