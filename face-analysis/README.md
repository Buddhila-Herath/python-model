# Face Analysis (CLI + FastAPI)

This project analyzes a video and produces a per-second facial emotion timeline, summary ratios, and a final confidence score.

It is fully local (CPU), command-line based, and built with:

- OpenCV for video decoding
- MediaPipe for face detection
- hsemotion-onnx for emotion inference

It also includes a FastAPI backend with sync and async processing endpoints.

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
  api/
    app.py
    routes.py
    models.py
    job_store.py
  config.py
  main.py
  create_sample_video.py
  requirements.txt
  README.md
  outputs/
    results.json
  services/
    analysis_service.py
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

### services/analysis_service.py

- Shared analysis entry point used by both CLI and API.
- Builds summary ratios:
  - `positive_ratio`
  - `neutral_ratio`
  - `negative_ratio`
- Keeps core ML pipeline reusable and centralized.

### api/app.py

- FastAPI app bootstrap.
- Starts a background cleanup loop on startup.
- Configures logging format for API and job lifecycle traces.

### api/routes.py

- Versioned API routes under `/api/v1`.
- Sync endpoint: `POST /api/v1/analyze`
- Async endpoints:
  - `POST /api/v1/analyze/async`
  - `GET /api/v1/analyze/jobs/{job_id}`
- Health endpoint: `GET /api/v1/health`
- Request controls:
  - `include_timeline` (default true)
  - `max_entries` (optional timeline truncation)

### api/job_store.py

- Thread-safe in-memory queue + job state manager.
- FIFO pending queue.
- Concurrency, queue limit, timeout, and TTL cleanup.
- Job metadata tracking (`created_at`, `started_at`, `completed_at`, `duration`).

### config.py

- Stores app defaults such as:
  - `target_fps = 1`
  - `min_face_confidence = 0.5`
- Defines emotion groups and weights used by scoring:
  - Positive: `happy`
  - Neutral: `neutral`
  - Surprise: `surprise` (special weighted category)
  - Negative: `fear`, `sad`, `angry`, `disgust`, `contempt`
- Defines normalization aliases for model output labels (for example `happiness -> happy`).

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
- Normalizes output to canonical labels before storing in timeline:
  - `emotion` as title-cased string
  - `emotion_confidence` in range [0, 1]

## Model Details

- Model: `enet_b0_8_best_afew` (ONNX)
- Trained on AffectNet / AFEW datasets
- Output classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
  - Contempt

### services/scoring.py

- `smooth_emotions`: rolling majority vote over recent frames (window=3)
- `compute_confidence_score`:
  - Positive contributes full weight
  - Neutral contributes half weight
  - Surprise contributes weight `0.3`
  - Negative contributes zero
  - Applies penalty if negative ratio > 0.4
  - Uses canonical label mapping to avoid mismatches (for example `Happiness` is counted as `happy`)

## Scoring Formula

Let:

- `T` = total timeline entries
- `P` = positive frames
- `N` = neutral frames
- `S` = surprise frames
- `G` = negative frames

Raw confidence:

```text
confidence = (P + 0.5 * N + 0.3 * S) / T
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

### Emotion Contribution Table

The final score uses the smoothed timeline and applies these per-frame contributions:

| Emotion group | Examples | Contribution per frame |
| --- | --- | --- |
| Positive | Happy, Happiness (via alias mapping) | 1.0 |
| Neutral | Neutral | 0.5 |
| Surprise | Surprise | 0.3 |
| Negative | Fear, Sad, Angry, Disgust, Contempt | 0.0 |
| Other/unknown | NoFace or any unmapped label | 0.0 |

Notes:

- Label aliases are normalized before scoring (for example, `Happiness -> happy`).
- If negative frames are more than 40% of total frames, a 0.7 penalty multiplier is applied.

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

In debug mode, raw model predictions are printed to terminal as `Raw preds: ...` for easier label inspection.

## Output JSON Schema

CLI output:

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
  "summary": {
    "positive_ratio": 0.1,
    "neutral_ratio": 0.8,
    "negative_ratio": 0.1
  },
  "confidence_score": 50.0
}
```

## FastAPI API

Base path: `/api/v1`

### Endpoints

- `GET /api/v1/health`
- `POST /api/v1/analyze`
- `POST /api/v1/analyze/async`
- `GET /api/v1/analyze/jobs/{job_id}`

### Sync analyze request

- Content type: `multipart/form-data`
- Required file field: `file`
- Query params:
  - `include_timeline` (bool, default true)
  - `max_entries` (int, optional)

### Async analyze response

```json
{
  "job_id": "uuid",
  "status": "pending"
}
```

Job statuses:

- `pending`
- `processing`
- `completed`
- `failed`

### Async job status response

```json
{
  "job_id": "uuid",
  "status": "completed",
  "created_at": 1776620819.754,
  "started_at": 1776620819.754,
  "completed_at": 1776620820.312,
  "duration": 0.558,
  "result": {
    "summary": {
      "positive_ratio": 0.25,
      "neutral_ratio": 0.5,
      "negative_ratio": 0.0
    },
    "confidence_score": 55.0,
    "timeline": [],
    "truncated": false
  }
}
```

Failed jobs include:

```json
{
  "status": "failed",
  "error": "message",
  "code": "inference_error"
}
```

### Error contract (all endpoints)

```json
{
  "error": "message",
  "code": "ERROR_TYPE"
}
```

Common codes:

- `invalid_request`
- `missing_file`
- `file_too_large`
- `too_many_jobs`
- `rate_limited`
- `inference_error`
- `job_timeout`

## Backend Guards and Limits

- Allowed file extensions: `.mp4`, `.avi`, `.mov`
- MIME check: must be `video/*`
- Max upload size: `200MB`
- JSON path mode: disabled by default (`ALLOW_JSON_PATH = False`)
- Rate limit: `20 requests / 60 seconds` per IP (in-memory)
- Queue model:
  - max processing: `2`
  - max pending: `10`
  - FIFO scheduling
- Max job runtime: `300` seconds
- TTL cleanup for completed/failed jobs: `15` minutes

Note: MIME check is a practical first filter and can be spoofed; stricter production validation can add magic-byte inspection.

## Run API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Docs:

- `http://127.0.0.1:8000/docs`

## Validation Status

Validated on April 19, 2026 with a full end-to-end run.

Final result:

- `TOTAL=26 PASS=26 FAIL=0`

Coverage:

- CLI execution and output generation
- Sync API success path
- Async job submit/poll lifecycle
- Job metadata (`created_at`, `started_at`, `completed_at`, `duration`)
- MIME rejection
- JSON-path disabled behavior
- Queue limit rejection (`429`)
- Rate limiting behavior (`200`, `200`, then `429`) in isolated test setup
- Timeline shaping (`include_timeline=false`, `max_entries`)

### Full Validation Command

Use this to run a complete validation pass in the project virtual environment:

```bash
Set-Location "c:\Users\buddh\Desktop\New folder\face-analysis"
..\.venv\Scripts\python.exe main.py --video "videos/Video_Generation_From_Description.mp4" --output "outputs/final_validation_cli.json"
```

Then run API checks with `fastapi.testclient` (health, sync, async, queue, and rate-limit scenarios), as executed in the final validation run above.

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

- `moov atom not found` during queue-limit stress testing
  - This can appear when intentionally sending tiny dummy bytes as fake videos to test queue overflow behavior. It does not indicate a failure of the real video analysis path.

- `cannot schedule new futures after interpreter shutdown` during short TestClient runs
  - This can appear when the Python test process exits while background async jobs are still running. In normal API server runtime (`uvicorn`), this is not a user-facing request-path failure.
