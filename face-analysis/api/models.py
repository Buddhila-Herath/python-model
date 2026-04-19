from pydantic import BaseModel, Field
from typing import Literal


class AnalyzePathRequest(BaseModel):
    video_path: str = Field(..., min_length=1, description="Local video file path")


class TimelineItem(BaseModel):
    time: int
    emotion: str
    emotion_confidence: float


class SummaryResponse(BaseModel):
    positive_ratio: float = Field(..., ge=0.0, le=1.0)
    neutral_ratio: float = Field(..., ge=0.0, le=1.0)
    negative_ratio: float = Field(..., ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    timeline: list[TimelineItem]
    summary: SummaryResponse
    confidence_score: float


class ErrorResponse(BaseModel):
    error: str
    code: str


class AnalyzeAsyncAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]


class AnalyzeJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    result: AnalyzeResponse | None = None
    error: str | None = None
    code: str | None = None
