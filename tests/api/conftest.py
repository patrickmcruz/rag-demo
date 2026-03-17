"""Shared fixtures for API tests.

Strategy:
- Create the FastAPI app via create_app() (runs real lifespan in TestClient).
- Immediately after TestClient starts, override app.state singletons with
  test mocks. The lifespan creates a RAGApplication with default config
  (lazy — no real LLM/embedding calls on startup), which we replace before
  any endpoint is called.

The app_config fixture is defined in tests/conftest.py and shared across
both core/ and api/ tests.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.tasks import IngestJobStore
from src.core.rag_service import RAGApplication


@pytest.fixture
def mock_rag_app(app_config):
    """RAGApplication backed by the shared app_config fixture."""
    return RAGApplication(app_config)


@pytest.fixture
def mock_job_store():
    """Empty IngestJobStore for each test."""
    return IngestJobStore()


@pytest.fixture
def api_client(mock_rag_app, mock_job_store):
    """
    TestClient with shared mock singletons injected into app.state.

    The TestClient runs the lifespan on __enter__.  We immediately override
    app.state to replace the lifespan-created singletons with our mocks.
    All endpoints retrieve singletons via DI (get_rag_app / get_job_store),
    which read from app.state, so the override is transparent to route code.
    """
    fastapi_app = create_app()
    with TestClient(fastapi_app, raise_server_exceptions=True) as client:
        fastapi_app.state.rag_app = mock_rag_app
        fastapi_app.state.job_store = mock_job_store
        yield client
