import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.job_store import can_accept_new_job, create_job, get_job, set_completed, set_failed, set_running, submit
from api.models import (
    AnalyzeAsyncAcceptedResponse,
    AnalyzeJobResponse,
    AnalyzePathRequest,
    AnalyzeResponse,
    ErrorResponse,
)
from config import AppConfig
from services.analysis_service import analyze_video


router = APIRouter()
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_SIZE_MB = 200
MAX_UPLOAD_BYTES = MAX_SIZE_MB * 1024 * 1024


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


def _run_analysis_job(job_id: str, input_video_path: str, temp_file_path: str | None) -> None:
    try:
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video file not found: {input_video_path}")

        result = _run_analysis(input_video_path)
        if not result.get("timeline"):
            raise ValueError("Analysis completed but timeline is empty.")

        set_completed(job_id, result)

    except FileNotFoundError as exc:
        set_failed(job_id, "missing_file", str(exc))
    except ValueError as exc:
        message = str(exc)
        code = "empty_timeline" if "timeline is empty" in message else "unsupported_format"
        set_failed(job_id, code, message)
    except RuntimeError as exc:
        set_failed(job_id, "inference_failure", str(exc))
    except Exception as exc:
        set_failed(job_id, "internal_error", f"Unexpected server error: {exc}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze(request: Request) -> Any:
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
                "empty_timeline",
                "Analysis completed but timeline is empty.",
            )

        return result

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
            "unsupported_format",
            str(exc),
        )
    except RuntimeError as exc:
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "inference_failure",
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

        if not can_accept_new_job():
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            return _error_response(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "too_many_jobs",
                "Too many active jobs. Try again later.",
            )

        job_id = str(uuid.uuid4())
        create_job(job_id)
        try:
            future = submit(_run_analysis_job, job_id, input_video_path, temp_file_path)
            set_running(job_id, future)
        except Exception as exc:
            set_failed(job_id, "internal_error", f"Failed to start job: {exc}")
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            return _error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "internal_error",
                f"Failed to start job: {exc}",
            )

        return {"job_id": job_id, "status": "pending"}

    except PermissionError as exc:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_request", str(exc))
    except OverflowError as exc:
        return _error_response(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file_too_large", str(exc))
    except ValueError as exc:
        return _error_response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported_format", str(exc))
    except Exception as exc:
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            f"Unexpected server error: {exc}",
        )


@router.get(
    "/analyze/jobs/{job_id}",
    response_model=AnalyzeJobResponse,
    responses={404: {"model": ErrorResponse}},
)
async def analyze_job_status(job_id: str) -> Any:
    job = get_job(job_id)
    if job is None:
        return _error_response(status.HTTP_404_NOT_FOUND, "job_not_found", f"Job not found: {job_id}")

    return {
        "job_id": job_id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "code": job.code,
    }
