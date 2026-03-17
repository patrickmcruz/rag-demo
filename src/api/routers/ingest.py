"""Ingestion endpoints."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from src.api.deps import get_job_store, get_rag_app
from src.api.models import IngestAcceptedResponse, IngestRequest, IngestStatusResponse
from src.api.tasks import IngestJob, IngestJobStore, JobStatus
from src.app import RAGApplication

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest")


def _run_ingest(
    job: IngestJob,
    job_store: IngestJobStore,
    rag_app: RAGApplication,
    request: IngestRequest,
) -> None:
    """
    Synchronous ingestion worker executed in Starlette's threadpool.
    Updates job state in IngestJobStore throughout.
    """
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    job_store.update(job)

    try:
        rag_app.apply_overrides(
            data_dir=Path(request.data_dir) if request.data_dir else None,
            vectorstore_dir=Path(request.vectorstore_dir) if request.vectorstore_dir else None,
            embedding_model=request.embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        rag_app.ingest(
            data_dir=request.data_dir,
            persist_dir=request.vectorstore_dir,
            file_types=request.file_types,
        )

        job.status = JobStatus.DONE
        job.completed_at = datetime.now(timezone.utc)
        logger.info(f"Ingestion job {job.job_id} completed successfully.")

    except Exception as exc:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(timezone.utc)
        job.error = str(exc)
        logger.error(f"Ingestion job {job.job_id} failed: {exc}")

    finally:
        job_store.update(job)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestAcceptedResponse,
)
async def trigger_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    rag_app: Annotated[RAGApplication, Depends(get_rag_app)],
    job_store: Annotated[IngestJobStore, Depends(get_job_store)],
) -> IngestAcceptedResponse:
    """Trigger document ingestion as a background job."""
    if job_store.has_running_job():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion job is already running. Poll /ingest/{job_id} for status.",
        )

    job = job_store.create()
    background_tasks.add_task(_run_ingest, job, job_store, rag_app, request)

    return IngestAcceptedResponse(job_id=job.job_id)


@router.get("/{job_id}", response_model=IngestStatusResponse)
async def get_ingest_status(
    job_id: str,
    job_store: Annotated[IngestJobStore, Depends(get_job_store)],
) -> IngestStatusResponse:
    """Poll the status of an ingestion job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    return IngestStatusResponse(
        job_id=job.job_id,
        status=job.status,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        docs_loaded=job.docs_loaded,
        chunks_created=job.chunks_created,
    )
