# Facial Emotion Analysis (CLI)

Production-ready CLI module for Viva Evaluation style emotion analysis from videos.

## Features

- Reads an input video path from CLI
- Samples video at 1 frame per second
- Detects faces with MediaPipe
- Crops face region before inference
- Classifies facial emotion using HSEmotion ONNX
- Produces per-second emotion timeline
- Computes confidence score
- Saves output JSON to `outputs/results.json` (or custom path)

## Project Structure

```text
face-analysis/
├── main.py
├── config.py
├── services/
│   ├── video_processor.py
│   ├── face_detector.py
│   ├── emotion_detector.py
│   ├── scoring.py
├── outputs/
│   └── results.json
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- CPU environment (no cloud APIs)

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python main.py --video path/to/video.mp4
```

Optional custom output path:

```bash
python main.py --video path/to/video.mp4 --output outputs/results.json
```

## Output Format

```json
{
  "timeline": [
    { "time": 0, "emotion": "Neutral", "emotion_confidence": 0.9132 },
    { "time": 1, "emotion": "Fear", "emotion_confidence": 0.7421 }
  ],
  "confidence_score": 72.5
}
```

## Confidence Score Logic

- Positive emotions: `Neutral`, `Happy`
- Negative emotions: `Fear`, `Sad`, `Angry`, `Disgust`, `Contempt`

Formula:

- `confidence = positive_frames / total_frames`
- If `negative_ratio > 0.4`, apply penalty: `confidence *= 0.7`
- `final_score = confidence * 100`

## Notes

- Frames with no detected face are skipped.
- Emotion inference is **never** run on full frame; only on cropped face.
- If no faces are detected, timeline is empty and confidence score is `0.0`.

## Example Test Command

```bash
python main.py --video sample.mp4
```
