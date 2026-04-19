# Backend API TODO

Goal: convert the current CLI pipeline into a FastAPI service while keeping the existing analysis logic reusable.

## Phase 1: Refactor for reuse

- Extract the pipeline entry point in `main.py` into a service function that accepts an input video path and returns the analysis payload.
- Keep `VideoProcessor`, `FaceDetector`, `EmotionDetector`, and scoring code unchanged as much as possible.
- Add a small response builder that can optionally include summary ratios.

## Phase 2: Define API contract

- Create a `POST /analyze` endpoint.
- Accept either:
  - a video file upload, or
  - a JSON body containing a video URL/path if local storage is used later.
- Return:
  - `timeline`
  - `summary`
  - `confidence_score`

## Phase 3: Add summary metrics

- Compute total counts from the final smoothed timeline.
- Add:
  - `positive_ratio`
  - `neutral_ratio`
  - `negative_ratio`
- Keep ratios normalized to `0.0` to `1.0`.

## Phase 4: Build FastAPI app

- Create a new app module, for example `api/app.py`.
- Add request validation with Pydantic models.
- Add structured error responses for:
  - missing file
  - unsupported format
  - empty timeline
  - inference failure

## Phase 5: Operational improvements

- Add request logging and timing.
- Add background processing for long videos if needed.
- Add CORS only if the frontend requires it.
- Add tests for the API response contract.

## Suggested folder shape

```text
face-analysis/
  api/
    app.py
    models.py
    routes.py
  services/
    ... existing analysis modules ...
  main.py
```

## Recommended implementation order

1. Refactor the existing CLI pipeline into a shared analysis function.
2. Add the `summary` builder.
3. Wrap the shared function with a FastAPI route.
4. Add tests and example request/response payloads.