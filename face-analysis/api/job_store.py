from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
from threading import Lock
from time import time
from typing import Callable, Dict


logger = logging.getLogger(__name__)


class JobLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[job_id={self.extra['job_id']}] {msg}", kwargs


def _job_logger(job_id: str) -> JobLoggerAdapter:
    return JobLoggerAdapter(logger, {"job_id": job_id})


@dataclass
class JobRecord:
    status: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    duration: float | None = None
    result: dict | None = None
    error: str | None = None
    code: str | None = None
    runner: Callable[[], dict] | None = None
    future: Future | None = None


class JobExecutionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


MAX_CONCURRENT = 2
MAX_QUEUE = 10
MAX_JOB_TIME = 300
JOB_TTL_SECONDS = 15 * 60

_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
_JOBS: Dict[str, JobRecord] = {}
_PENDING_QUEUE: deque[str] = deque()
_PROCESSING_IDS: set[str] = set()
_LOCK = Lock()

def _cleanup_expired_locked(now: float) -> None:
    expired_job_ids = [
        job_id
        for job_id, job in _JOBS.items()
        if job.status in {"completed", "failed"}
        and job.completed_at is not None
        and (now - job.completed_at) > JOB_TTL_SECONDS
    ]

    if expired_job_ids:
        logger.info("Cleaning up expired jobs: count=%s", len(expired_job_ids))
        expired = set(expired_job_ids)
        for job_id in expired_job_ids:
            del _JOBS[job_id]
        # Defensive cleanup in case of inconsistent state.
        while _PENDING_QUEUE and _PENDING_QUEUE[0] in expired:
            _PENDING_QUEUE.popleft()


def _fail_timed_out_jobs_locked(now: float) -> None:
    timed_out = []
    for job_id in list(_PROCESSING_IDS):
        job = _JOBS.get(job_id)
        if job is None or job.started_at is None:
            continue
        if (now - job.started_at) > MAX_JOB_TIME:
            timed_out.append(job_id)

    for job_id in timed_out:
        job = _JOBS.get(job_id)
        if job is None or job.started_at is None:
            continue
        job_logger = _job_logger(job_id)
        job.status = "failed"
        job.code = "job_timeout"
        job.error = f"Job exceeded max runtime of {MAX_JOB_TIME} seconds"
        job.completed_at = now
        job.duration = round(now - job.started_at, 4)
        job.runner = None
        job.future = None
        _PROCESSING_IDS.discard(job_id)
        job_logger.warning("timed out duration=%.2fs", job.duration)


def _finalize_job(job_id: str, *, result: dict | None, code: str | None, error: str | None) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job_logger = _job_logger(job_id)

        # Ignore late completion if job has already been marked failed (for example timeout watchdog).
        if job.status != "processing":
            return

        now = time()
        job.completed_at = now
        job.duration = round(now - job.started_at, 4) if job.started_at is not None else None
        job.future = None
        job.runner = None

        if error is None:
            job.status = "completed"
            job.result = result
            job.error = None
            job.code = None
            job_logger.info("completed duration=%.2fs", job.duration or 0.0)
        else:
            job.status = "failed"
            job.result = None
            job.error = error
            job.code = code
            job_logger.error("failed code=%s error=%s", code, error)

        _PROCESSING_IDS.discard(job_id)
        _start_next_jobs_locked()


def _run_job(job_id: str, runner: Callable[[], dict]) -> None:
    try:
        result = runner()
        _finalize_job(job_id, result=result, code=None, error=None)
    except JobExecutionError as exc:
        _finalize_job(job_id, result=None, code=exc.code, error=exc.message)
    except RuntimeError as exc:
        _finalize_job(job_id, result=None, code="inference_error", error=str(exc))
    except Exception as exc:
        _finalize_job(job_id, result=None, code="internal_error", error=f"Unexpected server error: {exc}")


def _start_next_jobs_locked() -> None:
    while len(_PROCESSING_IDS) < MAX_CONCURRENT and _PENDING_QUEUE:
        next_job_id = _PENDING_QUEUE.popleft()
        job = _JOBS.get(next_job_id)
        if job is None or job.status != "pending" or job.runner is None:
            continue
        job_logger = _job_logger(next_job_id)

        job.status = "processing"
        job.started_at = time()
        runner = job.runner
        future = _EXECUTOR.submit(_run_job, next_job_id, runner)
        job.future = future
        _PROCESSING_IDS.add(next_job_id)
        job_logger.info("started processing=%s pending=%s", len(_PROCESSING_IDS), len(_PENDING_QUEUE))


def maintain_jobs() -> None:
    with _LOCK:
        now = time()
        _fail_timed_out_jobs_locked(now)
        _cleanup_expired_locked(now)
        _start_next_jobs_locked()


def enqueue_job(job_id: str, runner: Callable[[], dict]) -> tuple[bool, str | None]:
    with _LOCK:
        job_logger = _job_logger(job_id)
        now = time()
        _fail_timed_out_jobs_locked(now)
        _cleanup_expired_locked(now)

        active_and_queued = len(_PROCESSING_IDS) + len(_PENDING_QUEUE)
        if active_and_queued >= (MAX_CONCURRENT + MAX_QUEUE):
            job_logger.warning("queue full, rejecting")
            return False, None

        _JOBS[job_id] = JobRecord(status="pending", created_at=now, runner=runner)
        _PENDING_QUEUE.append(job_id)
        job_logger.info("enqueued pending=%s", len(_PENDING_QUEUE))
        _start_next_jobs_locked()
        return True, _JOBS[job_id].status


def get_job_snapshot(job_id: str) -> dict | None:
    with _LOCK:
        now = time()
        _fail_timed_out_jobs_locked(now)
        _cleanup_expired_locked(now)
        _start_next_jobs_locked()

        job = _JOBS.get(job_id)
        if job is None:
            return None

        return {
            "job_id": job_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "duration": job.duration,
            "result": job.result,
            "error": job.error,
            "code": job.code,
        }


def get_queue_metrics() -> dict:
    with _LOCK:
        now = time()
        _fail_timed_out_jobs_locked(now)
        _cleanup_expired_locked(now)
        _start_next_jobs_locked()
        return {
            "active_jobs": len(_PROCESSING_IDS),
            "pending_jobs": len(_PENDING_QUEUE),
            "max_concurrent": MAX_CONCURRENT,
            "max_queue": MAX_QUEUE,
        }
