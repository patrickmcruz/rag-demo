"""Centralized configuration management for the RAG demo.

This module consolidates environment-driven settings into a single
data container with sane defaults. Import this instead of sprinkling
``os.getenv`` throughout the codebase.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class AppConfig:
    """Application configuration loaded from environment variables."""

    data_dir: Path
    vectorstore_dir: Path
    model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_documents: int
    temperature: float
    log_level: str

    @classmethod
    def load(cls, env: Optional[dict] = None) -> "AppConfig":
        """Create configuration from environment variables."""
        env = env or os.environ
        return cls(
            data_dir=Path(env.get("DATA_DIR", "./data")),
            vectorstore_dir=Path(env.get("VECTORSTORE_DIR", "./vectorstore")),
            model=env.get("OLLAMA_MODEL", "llama3"),
            embedding_model=env.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(env.get("CHUNK_SIZE", 500)),
            chunk_overlap=int(env.get("CHUNK_OVERLAP", 50)),
            top_k_documents=int(env.get("TOP_K_DOCUMENTS", 3)),
            temperature=float(env.get("TEMPERATURE", 0.0)),
            log_level=env.get("LOG_LEVEL", "INFO"),
        )


def resolve_path(path: Path) -> Path:
    """Expand user/home and return absolute path."""
    return path.expanduser().resolve()
