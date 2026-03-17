"""Shared pytest configuration for the test suite."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to sys.path for module imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path():
    """Local replacement for pytest's tmp_path to avoid Windows temp permission issues."""
    path = Path(tempfile.mkdtemp(prefix="rag-demo-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def app_config(tmp_path):
    """Shared AppConfig fixture with isolated paths for each test."""
    from src.config import AppConfig

    return AppConfig(
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vs",
        model="llama3",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=100,
        chunk_overlap=10,
        top_k_documents=3,
        temperature=0.0,
        log_level="WARNING",
        use_gpu=False,
        gpu_device=0,
    )
