"""Background task tracking for long-running ingestion jobs."""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class IngestJob:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    docs_loaded: Optional[int] = None
    chunks_created: Optional[int] = None


class IngestJobStore:
    """
    Thread-safe in-memory store for ingestion job state.

    Uses a threading.Lock because BackgroundTasks run in a different
    thread from the request handler.

    For production with multiple uvicorn workers, replace with a
    Redis or database-backed store.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def create(self) -> IngestJob:
        job = IngestJob(job_id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[IngestJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: IngestJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def has_running_job(self) -> bool:
        with self._lock:
            return any(j.status == JobStatus.RUNNING for j in self._jobs.values())
