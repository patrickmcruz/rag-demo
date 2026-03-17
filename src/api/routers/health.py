"""Health check endpoints."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from src import __version__
from src.api.deps import get_rag_app
from src.api.schemas import HealthResponse, ReadinessResponse
from src.core.rag_service import RAGApplication

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe — always returns 200 while the process is running."""
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    response: Response,
    rag_app: Annotated[RAGApplication, Depends(get_rag_app)],
) -> ReadinessResponse:
    """
    Readiness probe — returns 503 when the vectorstore has not been initialized.
    The chain itself can be lazy-built on first query; only the vectorstore
    directory is required to consider the service ready.
    """
    chain_ready = rag_app._rag_chain is not None
    vs_exists = rag_app.config.vectorstore_dir.exists()
    is_ready = vs_exists

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        chain_ready=chain_ready,
        vectorstore_exists=vs_exists,
    )
