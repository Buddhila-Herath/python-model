# Face Analysis CLI

This project analyzes a video and produces a per-second facial emotion timeline plus a final confidence score.

It is fully local (CPU), command-line based, and built with:

- OpenCV for video decoding
- MediaPipe for face detection
- hsemotion-onnx for emotion inference

## What This System Does

Given a video file, the pipeline:

1. Reads frames from the video.
2. Samples frames at 1 FPS by default.
3. Detects the largest face in each sampled frame.
4. Crops the face region (with small padding).
5. Predicts emotion and confidence for the cropped face.
6. Applies temporal smoothing to reduce jitter.
7. Computes a final confidence score from the smoothed timeline.
8. Writes JSON output to disk and prints it to console.

## End-to-End Flow

```text
CLI Args (--video, --output, --debug)
        |
        v
VideoProcessor.iter_frames(target_fps=1)
        |
        v
FaceDetector.detect_and_crop(frame)
   |                    |
   | face found         | no face
   v                    v
EmotionDetector.predict  timeline += { emotion: "NoFace", confidence: 0.0 }
        |
        v
timeline += { emotion, emotion_confidence }
        |
        v
smooth_emotions(window=3)
        |
        v
compute_confidence_score(...)
        |
        v
save_output(...) + print JSON
```

## Project Structure

```text
face-analysis/
  config.py
  main.py
  create_sample_video.py
  requirements.txt
  README.md
  outputs/
    results.json
  services/
    video_processor.py
    face_detector.py
    emotion_detector.py
    scoring.py
```

## Module Responsibilities

### main.py

- Parses CLI flags: `--video`, `--output`, `--debug`
- Creates and wires all services
- Runs the pipeline
- Saves and prints results
- Handles errors (missing file / runtime failure)

### config.py

- Stores app defaults such as:
  - `target_fps = 1`
  - `min_face_confidence = 0.5`
- Defines emotion groups used by scoring:
  - Positive: `happy`
  - Neutral: `neutral`
  - Negative: `fear`, `sad`, `angry`, `disgust`, `contempt`

### services/video_processor.py

- Opens video via `cv2.VideoCapture`
- Computes sampling step from source FPS
- Yields `FrameData(time_sec, frame)` at target FPS
- Falls back to source FPS = 25 if metadata is unavailable

### services/face_detector.py

- Detects faces with MediaPipe
- Supports both:
  - MediaPipe Solutions API (if available)
  - MediaPipe Tasks API (fallback)
- For Tasks mode, auto-downloads detector model to `.models/`
- Picks the largest detected face
- Crops with configurable padding (`padding_ratio=0.1`)

### services/emotion_detector.py

- Runs HSEmotion ONNX model (`enet_b0_8_best_afew` by default)
- Handles multiple output shapes/types from model APIs
- Normalizes output to:
  - `emotion` as title-cased string
  - `emotion_confidence` in range [0, 1]

### services/scoring.py

- `smooth_emotions`: rolling majority vote over recent frames (window=3)
- `compute_confidence_score`:
  - Positive contributes full weight
  - Neutral contributes half weight
  - Negative contributes zero
  - Applies penalty if negative ratio > 0.4

## Scoring Formula

Let:

- `T` = total timeline entries
- `P` = positive frames
- `N` = neutral frames
- `G` = negative frames

Raw confidence:

```text
confidence = (P + 0.5 * N) / T
```

Penalty rule:

```text
if G / T > 0.4:
    confidence = confidence * 0.7
```

Final score:

```text
confidence_score = round(confidence * 100, 2)
```

## Requirements

- Python 3.10+
- Works on CPU (no cloud APIs required)

## Install

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Basic run:

```bash
python main.py --video path/to/video.mp4
```

Custom output path:

```bash
python main.py --video path/to/video.mp4 --output outputs/results.json
```

Debug mode (prints extra detector/model info):

```bash
python main.py --video path/to/video.mp4 --output outputs/results.json --debug
```

## Output JSON Schema

```json
{
  "timeline": [
    {
      "time": 0,
      "emotion": "Neutral",
      "emotion_confidence": 0.7645
    },
    {
      "time": 1,
      "emotion": "NoFace",
      "emotion_confidence": 0.0
    }
  ],
  "confidence_score": 50.0
}
```

### Important Behavior

- If no face is found at a sampled second, that second is still included with:
  - `emotion = "NoFace"`
  - `emotion_confidence = 0.0`
- Timeline is smoothed before final scoring.
- If video cannot be opened, the CLI returns an error.

## Quick Local Test

If you have a face image named `lena.jpg`, you can create a short sample video:

```bash
python create_sample_video.py
```

Then run:

```bash
python main.py --video sample_face.mp4 --output outputs/results.json --debug
```

## Troubleshooting

- Error: `Video file not found`
  - Check `--video` path.

- Error: `Could not open video file`
  - Verify codec/file integrity and OpenCV installation.

- Slow startup in some environments
  - First run may download a MediaPipe face detector model when Tasks API is used.

- Unexpected emotion labels
  - Model outputs vary by backend; parser normalizes many formats, but label sets depend on model behavior.
