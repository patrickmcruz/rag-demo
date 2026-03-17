"""Unit tests for IngestJobStore state machine and thread safety."""

import threading

from src.api.tasks import IngestJob, IngestJobStore, JobStatus


def test_create_returns_queued_job():
    store = IngestJobStore()
    job = store.create()
    assert isinstance(job, IngestJob)
    assert job.status == JobStatus.QUEUED
    assert job.job_id is not None
    assert len(job.job_id) > 0


def test_get_missing_job_returns_none():
    store = IngestJobStore()
    assert store.get("nonexistent-id") is None


def test_get_known_job_returns_it():
    store = IngestJobStore()
    job = store.create()
    retrieved = store.get(job.job_id)
    assert retrieved is not None
    assert retrieved.job_id == job.job_id


def test_update_persists_status_change():
    store = IngestJobStore()
    job = store.create()
    job.status = JobStatus.DONE
    store.update(job)
    retrieved = store.get(job.job_id)
    assert retrieved.status == JobStatus.DONE


def test_update_persists_error():
    store = IngestJobStore()
    job = store.create()
    job.status = JobStatus.FAILED
    job.error = "Something went wrong"
    store.update(job)
    retrieved = store.get(job.job_id)
    assert retrieved.status == JobStatus.FAILED
    assert retrieved.error == "Something went wrong"


def test_has_running_job_false_when_empty():
    store = IngestJobStore()
    assert store.has_running_job() is False


def test_has_running_job_false_when_all_done():
    store = IngestJobStore()
    job = store.create()
    job.status = JobStatus.DONE
    store.update(job)
    assert store.has_running_job() is False


def test_has_running_job_true_when_running():
    store = IngestJobStore()
    job = store.create()
    job.status = JobStatus.RUNNING
    store.update(job)
    assert store.has_running_job() is True


def test_concurrent_creates_unique_ids():
    """Thread-safe creation must produce unique job IDs."""
    store = IngestJobStore()
    ids = []
    lock = threading.Lock()

    def create_and_collect():
        job = store.create()
        with lock:
            ids.append(job.job_id)

    threads = [threading.Thread(target=create_and_collect) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ids) == 50
    assert len(set(ids)) == 50, "Duplicate job IDs detected"
