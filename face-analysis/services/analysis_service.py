from typing import Dict, List

from config import AppConfig, NEGATIVE_EMOTIONS, NEUTRAL_EMOTIONS, POSITIVE_EMOTIONS, canonical_emotion_label
from services.emotion_detector import EmotionDetector
from services.face_detector import FaceDetector
from services.scoring import compute_confidence_score, smooth_emotions
from services.video_processor import VideoProcessor


def build_summary(timeline: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(timeline)
    if total == 0:
        return {
            "positive_ratio": 0.0,
            "neutral_ratio": 0.0,
            "negative_ratio": 0.0,
        }

    positive_count = 0
    neutral_count = 0
    negative_count = 0

    for item in timeline:
        emotion = canonical_emotion_label(str(item.get("emotion", "")))
        if emotion in POSITIVE_EMOTIONS:
            positive_count += 1
        if emotion in NEUTRAL_EMOTIONS:
            neutral_count += 1
        if emotion in NEGATIVE_EMOTIONS:
            negative_count += 1

    return {
        "positive_ratio": round(positive_count / total, 4),
        "neutral_ratio": round(neutral_count / total, 4),
        "negative_ratio": round(negative_count / total, 4),
    }


def analyze_video(config: AppConfig, include_summary: bool = True) -> Dict[str, object]:
    video_processor = VideoProcessor(target_fps=config.target_fps)
    face_detector = FaceDetector(
        min_detection_confidence=config.min_face_confidence,
        debug=config.debug,
    )
    emotion_detector = EmotionDetector(debug=config.debug)

    timeline: List[Dict[str, object]] = []

    for frame_data in video_processor.iter_frames(config.video_path):
        face_crop = face_detector.detect_and_crop(frame_data.frame)
        if face_crop is None:
            timeline.append(
                {
                    "time": frame_data.time_sec,
                    "emotion": "NoFace",
                    "emotion_confidence": 0.0,
                }
            )
            continue

        emotion, emotion_confidence = emotion_detector.predict(face_crop)
        timeline.append(
            {
                "time": frame_data.time_sec,
                "emotion": emotion,
                "emotion_confidence": round(emotion_confidence, 4),
            }
        )

    smoothed_timeline = smooth_emotions(timeline)
    confidence_score = compute_confidence_score(smoothed_timeline)

    response: Dict[str, object] = {
        "timeline": smoothed_timeline,
        "confidence_score": confidence_score,
    }

    if include_summary:
        response["summary"] = build_summary(smoothed_timeline)

    return response
