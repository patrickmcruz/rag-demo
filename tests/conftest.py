"""Test configuration for ensuring src is importable."""

import sys
from pathlib import Path

import pytest

# Add project root to sys.path for module imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Use a local basetemp to avoid Windows permission errors on system temp."""
    local_tmp = ROOT / ".pytest_tmp"
    local_tmp.mkdir(exist_ok=True)
    config.option.basetemp = local_tmp


@pytest.fixture
def app_config(tmp_path):
    """Shared AppConfig fixture with all required fields.

    Used across core/ and api/ tests.  Uses tmp_path so each test gets
    isolated data_dir and vectorstore_dir paths.
    """
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
