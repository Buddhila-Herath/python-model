from typing import Dict, List

from config import NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS


def compute_confidence_score(timeline: List[Dict[str, object]]) -> float:
    total_frames = len(timeline)
    if total_frames == 0:
        return 0.0

    positive_frames = 0
    negative_frames = 0

    for item in timeline:
        emotion = str(item.get("emotion", "")).strip().lower()
        if emotion in POSITIVE_EMOTIONS:
            positive_frames += 1
        if emotion in NEGATIVE_EMOTIONS:
            negative_frames += 1

    confidence = positive_frames / total_frames
    negative_ratio = negative_frames / total_frames

    if negative_ratio > 0.4:
        confidence *= 0.7

    final_score = round(confidence * 100, 2)
    return final_score
