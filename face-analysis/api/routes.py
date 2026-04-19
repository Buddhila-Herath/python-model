import os
import tempfile
import uuid
from collections import deque
import logging
from pathlib import Path
from threading import Lock
from time import time
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.job_store import JobExecutionError, enqueue_job, get_job_snapshot, get_queue_metrics
from api.models import (
    AnalyzeAsyncAcceptedResponse,
    AnalyzeJobResponse,
    AnalyzePathRequest,
    AnalyzeResponse,
    ErrorResponse,
)
from config import AppConfig
from services.analysis_service import analyze_video


router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_SIZE_MB = 200
MAX_UPLOAD_BYTES = MAX_SIZE_MB * 1024 * 1024
ALLOW_JSON_PATH = False
RATE_LIMIT = 20
WINDOW_SECONDS = 60
_RATE_LOCK = Lock()
_REQUESTS_BY_IP: dict[str, deque[float]] = {}
_RATE_CONFIG_SNAPSHOT = (RATE_LIMIT, WINDOW_SECONDS)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": message, "code": code},
    )


def _validate_extension(file_path: str) -> None:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{ext or 'unknown'}'. Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )


def _validate_file_size(file_size_bytes: int) -> None:
    if file_size_bytes > MAX_UPLOAD_BYTES:
        raise OverflowError(f"File too large. Maximum allowed size is {MAX_SIZE_MB}MB")


def _check_rate_limit(request: Request) -> JSONResponse | None:
    global _RATE_CONFIG_SNAPSHOT
    now = time()
    client_host = request.client.host if request.client else "unknown"

    with _RATE_LOCK:
        current_config = (RATE_LIMIT, WINDOW_SECONDS)
        if current_config != _RATE_CONFIG_SNAPSHOT:
            _REQUESTS_BY_IP.clear()
            _RATE_CONFIG_SNAPSHOT = current_config

        request_times = _REQUESTS_BY_IP.setdefault(client_host, deque())
        while request_times and (now - request_times[0]) > WINDOW_SECONDS:
            request_times.popleft()

        if len(request_times) >= RATE_LIMIT:
            return _error_response(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "rate_limited",
                f"Rate limit exceeded: {RATE_LIMIT} requests per {WINDOW_SECONDS} seconds",
            )

        request_times.append(now)

    return None


def _validate_json_video_path(video_path: str) -> str:
    candidate = Path(video_path)
    resolved = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    resolved = resolved.resolve()

    # Restrict JSON path access to project files only.
    if PROJECT_ROOT not in resolved.parents and resolved != PROJECT_ROOT:
        raise PermissionError("video_path must point to a file under the project directory")

    return str(resolved)


async def _extract_input_video(request: Request) -> tuple[str | None, str | None, JSONResponse | None]:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_UPLOAD_BYTES:
                    return (
                        None,
                        None,
                        _error_response(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            "file_too_large",
                            f"Upload exceeds max size of {MAX_SIZE_MB}MB.",
                        ),
                    )
            except ValueError:
                pass

        form = await request.form()
        uploaded_file = form.get("file")
        if uploaded_file is None:
            return (
                None,
                None,
                _error_response(
                    status.HTTP_400_BAD_REQUEST,
                    "missing_file",
                    "Provide a video file using multipart form field 'file'.",
                ),
            )

        # Content-Type is a practical first-line check, but it can be spoofed.
        # For stricter validation in production, add magic-byte/header inspection.
        uploaded_content_type = (getattr(uploaded_file, "content_type", "") or "").lower()
        if not uploaded_content_type.startswith("video/"):
            return (
                None,
                None,
                _error_response(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "invalid_request",
                    f"Invalid MIME type '{uploaded_content_type}'. Expected a video/* content type.",
                ),
            )

        original_name = getattr(uploaded_file, "filename", "") or "uploaded_video"
        _validate_extension(original_name)

        suffix = os.path.splitext(original_name)[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_file_path = tmp.name
            file_bytes = await uploaded_file.read()
            if not file_bytes:
                return (
                    None,
                    None,
                    _error_response(
                        status.HTTP_400_BAD_REQUEST,
                        "missing_file",
                        "Uploaded video file is empty.",
                    ),
                )
            try:
                _validate_file_size(len(file_bytes))
            except OverflowError:
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
                return (
                    None,
                    None,
                    _error_response(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        "file_too_large",
                        f"Upload exceeds max size of {MAX_SIZE_MB}MB.",
                    ),
                )
            tmp.write(file_bytes)

        return temp_file_path, temp_file_path, None

    if not ALLOW_JSON_PATH:
        return (
            None,
            None,
            _error_response(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "invalid_request",
                "JSON video_path disabled",
            ),
        )

    try:
        body = await request.json()
    except Exception:
        return (
            None,
            None,
            _error_response(
                status.HTTP_400_BAD_REQUEST,
                "missing_file",
                "Provide either multipart form-data with 'file' or JSON with 'video_path'.",
            ),
        )

    try:
        payload = AnalyzePathRequest.model_validate(body)
    except ValidationError as exc:
        return (
            None,
            None,
            _error_response(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "invalid_request",
                f"Invalid request body: {exc.errors()}",
            ),
        )

    input_video_path = _validate_json_video_path(payload.video_path)
    _validate_extension(input_video_path)
    if os.path.exists(input_video_path):
        _validate_file_size(os.path.getsize(input_video_path))
    return input_video_path, None, None


def _run_analysis(video_path: str) -> dict:
    return analyze_video(AppConfig(video_path=video_path), include_summary=True)


def _build_job_runner(input_video_path: str, temp_file_path: str | None):
    def _runner() -> dict:
        try:
            if not os.path.exists(input_video_path):
                raise JobExecutionError("missing_file", f"Video file not found: {input_video_path}")

            result = _run_analysis(input_video_path)
            if not result.get("timeline"):
                raise JobExecutionError("invalid_request", "Analysis completed but timeline is empty.")

            return result
        except JobExecutionError:
            raise
        except OverflowError as exc:
            raise JobExecutionError("file_too_large", str(exc)) from exc
        except ValueError as exc:
            raise JobExecutionError("invalid_request", str(exc)) from exc
        except RuntimeError as exc:
            raise JobExecutionError("inference_error", str(exc)) from exc
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

    return _runner


def _project_result(
    result: dict,
    include_timeline: bool,
    max_entries: int | None,
) -> dict:
    projected = {
        "summary": result.get("summary"),
        "confidence_score": result.get("confidence_score"),
        "truncated": False,
    }

    timeline = list(result.get("timeline", []))

    if not include_timeline:
        return projected

    if max_entries is not None and max_entries >= 0 and len(timeline) > max_entries:
        projected["timeline"] = timeline[:max_entries]
        projected["truncated"] = True
        return projected

    projected["timeline"] = timeline
    return projected


@router.get("/health")
async def health(request: Request) -> Any:
    rate_error = _check_rate_limit(request)
    if rate_error is not None:
        return rate_error

    metrics = get_queue_metrics()
    logger.info("Health check: active_jobs=%s pending_jobs=%s", metrics["active_jobs"], metrics["pending_jobs"])
    return {
        "status": "ok",
        "active_jobs": metrics["active_jobs"],
        "pending_jobs": metrics["pending_jobs"],
    }


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    response_model_exclude_none=True,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze(
    request: Request,
    include_timeline: bool = True,
    max_entries: int | None = None,
) -> Any:
    rate_error = _check_rate_limit(request)
    if rate_error is not None:
        return rate_error

    input_video_path: str | None = None
    temp_file_path: str | None = None

    try:
        input_video_path, temp_file_path, early_response = await _extract_input_video(request)
        if early_response is not None:
            return early_response

        if not input_video_path or not os.path.exists(input_video_path):
            return _error_response(
                status.HTTP_404_NOT_FOUND,
                "missing_file",
                f"Video file not found: {input_video_path}",
            )

        result = await run_in_threadpool(_run_analysis, input_video_path)

        if not result.get("timeline"):
            return _error_response(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "invalid_request",
                "Analysis completed but timeline is empty.",
            )

        return _project_result(result, include_timeline=include_timeline, max_entries=max_entries)

    except PermissionError as exc:
        return _error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_request",
            str(exc),
        )
    except OverflowError as exc:
        return _error_response(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "file_too_large",
            str(exc),
        )
    except ValueError as exc:
        return _error_response(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "invalid_request",
            str(exc),
        )
    except RuntimeError as exc:
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "inference_error",
            str(exc),
        )
    except Exception as exc:
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            f"Unexpected server error: {exc}",
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


@router.post(
    "/analyze/async",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AnalyzeAsyncAcceptedResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_async(request: Request) -> Any:
    rate_error = _check_rate_limit(request)
    if rate_error is not None:
        return rate_error

    try:
        input_video_path, temp_file_path, early_response = await _extract_input_video(request)
        if early_response is not None:
            return early_response

        if not input_video_path or not os.path.exists(input_video_path):
            return _error_response(
                status.HTTP_404_NOT_FOUND,
                "missing_file",
                f"Video file not found: {input_video_path}",
            )

        job_id = str(uuid.uuid4())
        accepted, status_name = enqueue_job(job_id, _build_job_runner(input_video_path, temp_file_path))
        if not accepted:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            return _error_response(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "too_many_jobs",
                "Too many jobs in processing/queue. Try again later.",
            )

        logger.info("Async job accepted: job_id=%s status=%s", job_id, status_name)
        return {"job_id": job_id, "status": status_name}

    except PermissionError as exc:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_request", str(exc))
    except OverflowError as exc:
        return _error_response(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file_too_large", str(exc))
    except ValueError as exc:
        return _error_response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "invalid_request", str(exc))
    except Exception as exc:
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            f"Unexpected server error: {exc}",
        )


@router.get(
    "/analyze/jobs/{job_id}",
    response_model=AnalyzeJobResponse,
    response_model_exclude_none=True,
    responses={404: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
async def analyze_job_status(
    request: Request,
    job_id: str,
    include_timeline: bool = True,
    max_entries: int | None = None,
) -> Any:
    rate_error = _check_rate_limit(request)
    if rate_error is not None:
        return rate_error

    snapshot = get_job_snapshot(job_id)
    if snapshot is None:
        return _error_response(status.HTTP_404_NOT_FOUND, "job_not_found", f"Job not found: {job_id}")

    result_payload = snapshot.get("result")
    projected_result = None
    if isinstance(result_payload, dict):
        projected_result = _project_result(
            result_payload,
            include_timeline=include_timeline,
            max_entries=max_entries,
        )

    return {
        "job_id": job_id,
        "status": snapshot.get("status"),
        "created_at": snapshot.get("created_at"),
        "started_at": snapshot.get("started_at"),
        "completed_at": snapshot.get("completed_at"),
        "duration": snapshot.get("duration"),
        "result": projected_result,
        "error": snapshot.get("error"),
        "code": snapshot.get("code"),
    }
