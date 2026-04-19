from collections import Counter
from typing import Dict, List

from config import NEGATIVE_EMOTIONS, NEUTRAL_EMOTIONS, POSITIVE_EMOTIONS


def smooth_emotions(timeline: List[Dict[str, object]], window: int = 3) -> List[Dict[str, object]]:
    if window <= 0:
        return timeline

    smoothed: List[Dict[str, object]] = []
    for i in range(len(timeline)):
        window_slice = timeline[max(0, i - window): i + 1]
        emotions = [str(item.get("emotion", "")) for item in window_slice]
        if emotions:
            most_common = Counter(emotions).most_common(1)[0][0]
            updated = dict(timeline[i])
            updated["emotion"] = most_common
            smoothed.append(updated)
        else:
            smoothed.append(dict(timeline[i]))
    return smoothed


def compute_confidence_score(timeline: List[Dict[str, object]]) -> float:
    total_frames = len(timeline)
    if total_frames == 0:
        return 0.0

    positive_frames = 0
    neutral_frames = 0
    negative_frames = 0

    for item in timeline:
        emotion = str(item.get("emotion", "")).strip().lower()
        if emotion in POSITIVE_EMOTIONS:
            positive_frames += 1
        if emotion in NEUTRAL_EMOTIONS:
            neutral_frames += 1
        if emotion in NEGATIVE_EMOTIONS:
            negative_frames += 1

    confidence = (positive_frames + 0.5 * neutral_frames) / total_frames
    negative_ratio = negative_frames / total_frames

    if negative_ratio > 0.4:
        confidence *= 0.7

    final_score = round(confidence * 100, 2)
    return final_score
