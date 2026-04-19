from dataclasses import dataclass, field
from typing import Set


POSITIVE_EMOTIONS: Set[str] = {"happy"}
NEUTRAL_EMOTIONS: Set[str] = {"neutral"}
NEGATIVE_EMOTIONS: Set[str] = {
    "fear",
    "sad",
    "angry",
    "disgust",
    "contempt",
}


@dataclass
class AppConfig:
    video_path: str
    output_path: str = "outputs/results.json"
    target_fps: int = 1
    min_face_confidence: float = 0.5
    positive_emotions: Set[str] = field(default_factory=lambda: set(POSITIVE_EMOTIONS))
    neutral_emotions: Set[str] = field(default_factory=lambda: set(NEUTRAL_EMOTIONS))
    debug: bool = False
