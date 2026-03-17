"""FastAPI dependency providers for shared singletons."""

from fastapi import Depends, HTTPException, Request, status

from src.core.rag_service import RAGApplication
from src.api.tasks import IngestJobStore


def get_rag_app(request: Request) -> RAGApplication:
    """Inject the RAGApplication singleton from app.state."""
    return request.app.state.rag_app


def get_job_store(request: Request) -> IngestJobStore:
    """Inject the IngestJobStore singleton from app.state."""
    return request.app.state.job_store


def get_ready_rag_app(request: Request) -> RAGApplication:
    """
    Like get_rag_app but raises 503 if the vectorstore does not exist.
    Use for endpoints that require an initialized chain (e.g. /query, /chat/stream).
    """
    rag_app: RAGApplication = request.app.state.rag_app
    if not rag_app.config.vectorstore_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vectorstore not initialized. "
                "Run POST /ingest first."
            ),
        )
    return rag_app
