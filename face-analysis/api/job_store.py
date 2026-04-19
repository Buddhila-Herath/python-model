from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Dict


@dataclass
class JobRecord:
    status: str
    future: Future | None = None
    result: dict | None = None
    error: str | None = None
    code: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_JOBS: Dict[str, JobRecord] = {}
_LOCK = Lock()
MAX_CONCURRENT_JOBS = 2
JOB_TTL_SECONDS = 15 * 60


def _touch(job: JobRecord) -> None:
    job.updated_at = time()


def cleanup_expired_jobs() -> None:
    now = time()
    with _LOCK:
        expired_job_ids = [
            job_id
            for job_id, job in _JOBS.items()
            if job.status in {"completed", "failed"} and (now - job.updated_at) > JOB_TTL_SECONDS
        ]
        for job_id in expired_job_ids:
            del _JOBS[job_id]


def can_accept_new_job() -> bool:
    cleanup_expired_jobs()
    with _LOCK:
        active = sum(1 for job in _JOBS.values() if job.status in {"pending", "processing"})
        return active < MAX_CONCURRENT_JOBS


def create_job(job_id: str) -> None:
    with _LOCK:
        now = time()
        _JOBS[job_id] = JobRecord(status="pending", created_at=now, updated_at=now)


def set_running(job_id: str, future: Future) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        job.status = "processing"
        job.future = future
        _touch(job)


def set_completed(job_id: str, result: dict) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        job.status = "completed"
        job.result = result
        job.future = None
        _touch(job)


def set_failed(job_id: str, code: str, error: str) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        job.status = "failed"
        job.code = code
        job.error = error
        job.future = None
        _touch(job)


def get_job(job_id: str) -> JobRecord | None:
    cleanup_expired_jobs()
    with _LOCK:
        return _JOBS.get(job_id)


def submit(function, *args, **kwargs) -> Future:
    return _EXECUTOR.submit(function, *args, **kwargs)
