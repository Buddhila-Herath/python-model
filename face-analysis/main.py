import argparse
import json
import os
import sys
from typing import Dict, List

from config import AppConfig
from services.emotion_detector import EmotionDetector
from services.face_detector import FaceDetector
from services.scoring import compute_confidence_score
from services.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facial Emotion Analysis CLI")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument(
        "--output",
        default="outputs/results.json",
        help="Path for output JSON file (default: outputs/results.json)",
    )
    return parser.parse_args()


def run_pipeline(config: AppConfig) -> Dict[str, object]:
    video_processor = VideoProcessor(target_fps=config.target_fps)
    face_detector = FaceDetector(min_detection_confidence=config.min_face_confidence)
    emotion_detector = EmotionDetector()

    timeline: List[Dict[str, object]] = []

    for frame_data in video_processor.iter_frames(config.video_path):
        face_crop = face_detector.detect_and_crop(frame_data.frame)
        if face_crop is None:
            continue

        emotion, emotion_confidence = emotion_detector.predict(face_crop)
        timeline.append(
            {
                "time": frame_data.time_sec,
                "emotion": emotion,
                "emotion_confidence": round(emotion_confidence, 4),
            }
        )

    confidence_score = compute_confidence_score(timeline)
    return {
        "timeline": timeline,
        "confidence_score": confidence_score,
    }


def save_output(data: Dict[str, object], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    args = parse_args()
    config = AppConfig(video_path=args.video, output_path=args.output)

    try:
        if not os.path.exists(config.video_path):
            raise FileNotFoundError(f"Video file not found: {config.video_path}")

        result = run_pipeline(config)
        save_output(result, config.output_path)

        print(json.dumps(result, indent=2))
        print(f"\nResults written to: {config.output_path}")

        if not result["timeline"]:
            print("Warning: No faces detected in sampled frames (1 FPS).")

        return 0

    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
