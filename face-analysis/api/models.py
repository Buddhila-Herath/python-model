from pydantic import BaseModel, Field
from typing import Literal, Optional


class AnalyzePathRequest(BaseModel):
    video_path: str = Field(..., min_length=1, description="Local video file path")


class TimelineItem(BaseModel):
    time: int
    emotion: str
    emotion_confidence: Optional[float] = None


class SummaryResponse(BaseModel):
    positive_ratio: float = Field(..., ge=0.0, le=1.0)
    neutral_ratio: float = Field(..., ge=0.0, le=1.0)
    negative_ratio: float = Field(..., ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    timeline: list[TimelineItem] | None = None
    summary: SummaryResponse
    confidence_score: float
    engagement_score: float
    truncated: bool = False


class ErrorResponse(BaseModel):
    error: str
    code: str


class AnalyzeAsyncAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]


class AnalyzeJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    duration: float | None = None
    result: AnalyzeResponse | None = None
    error: str | None = None
    code: str | None = None
