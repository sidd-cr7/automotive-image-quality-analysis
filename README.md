# ENGINE INTELLIGENCE

Fine-grained vehicle recognition and analysis. Upload a photo; get a ranked make/model guess, exterior component
segmentation, an image-quality report and (when a vision model is available) a written visual analysis.
Built for CV Hackathon '26. Design language: *Editorial Surrealism*.

## What it does (and does not do)

| Stage | Implementation |
|---|---|
| Vehicle detection | YOLOv8n (`models/detection/yolov8n.pt`) |
| Vehicle classification | **Whole frame** resized to 224x224, ResNet-50 + EfficientNet-B0 probabilities averaged, 196 Stanford Cars classes |
| Exterior analysis | YOLO11n-seg (`exterior_best.pt`), 23 exterior component classes |
| Image quality | Blur, exposure, noise, tilt metrics plus adaptive enhancement (CLAHE, gamma, sharpening) |
| Vision analysis | **One combined request per image** to Gemini Flash (primary), else Groq (secondary, two keys); see "Vision providers" |
| Specs / valuation | Looked up from `data/automotive_reference_data.json`; left **unavailable** when the model is not in the data |

### Honest accuracy
Measured on 150 held-out Stanford Cars *test* photos through the real pipeline (CV models only, Groq off):

| Setup | Top-1 | Top-5 |
|---|---|---|
| Old: classify the YOLO crop | 7.3% | 24.7% |
| Current: full frame, ResNet-50 + EfficientNet-B0 | 36.7% | 70.7% |

The classifiers were trained on whole frames, so classifying a tight crop collapses accuracy. The weights are
modest (about 50% top-1 on their own training images). They only know the 196 Stanford Cars classes (mostly US
models, 1990 to 2012). **Modern, non-US, dark or occluded cars are usually not recognised by the classifier.**
Confidence is reasonably calibrated: above 0.40 was roughly 90% accurate in testing, below 0.20 about 25%.
Anything under the 0.40 threshold is shown as *uncertain*.

### Who identified the car
The result's `vehicle.identification_source` says which component produced the answer:

- `classifier_only`: local ResNet-50 + EfficientNet-B0 result; no vision provider answered.
- `classifier_confirmed_by_<provider>`: the vision provider reached the same make/model; classifier confidence is shown.
- `<provider>_vision` (`gemini_vision` / `groq_vision`): the vision provider changed the identification. There is
  **no calibrated confidence** (`confidence` is `null`), the result is flagged uncertain, `vehicle.vision_provider`
  names the provider, and the classifier's own guess is shown alongside. Year is an estimate.

In a small hands-on check (6 hard images: night/smoke Challenger, a modern UK Hyundai i20, four Stanford Cars photos),
Gemini named the right make and model on all 6 where the local classifier got none. That is six images, not a
benchmark, and year estimates were off by 1 to 3 years.

## Architecture

```
React 19 + Vite (frontend/)  ->  FastAPI (api.py)  ->  VehicleRecognitionPipeline (src/pipeline.py)
                                                         -> MultiModelOrchestrator (src/orchestrator.py)
   Branch 1: detector.py + classifier.py   Branch 2: exterior_model.py   Branch 3 + verdict: vision_service.py
   quality_assessment.py / adaptive_enhancer.py / prediction_comparator.py / vehicle_intelligence.py
```

`GET /progress/{job_id}` reports real server-side stage boundaries (detection, inspection, final) so the UI
loading panel never fakes progress. Production: FastAPI serves the built UI from `frontend/dist` on the same origin.

## Setup

Requirements: Python 3.10+ and Node 18+.

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your real keys in .env (never commit it)
cd frontend && npm install && npm run build && cd ..
python run_server.py        # http://localhost:8000  (UI + API)
```

Development with hot reload: run `python run_server.py` and, in `frontend/`, `npm run dev` (http://localhost:5173).

## Environment variables (`.env`)

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Gemini key (primary vision provider). | empty |
| `GEMINI_MODEL`, `GEMINI_TIMEOUT` | Model id and request timeout (s). `gemini-2.5-flash` is retired for new keys; `gemini-flash-latest` timed out in testing. | `gemini-flash-lite-latest`, 25 |
| `VISION_PROVIDER_ORDER` | Comma-separated provider order (`gemini`, `groq`) | `gemini,groq` |
| `VISION_CACHE_SIZE`, `VISION_CACHE_TTL` | Result cache entries and TTL (s) | 64, 3600 |
| `GROQ_API_KEY` / `GROQ_API_KEY_BACKUP` | Primary and backup Groq keys. Read only from the environment, never sent to the browser, logged or returned. | empty |
| `GROQ_VISION_MODEL`, `GROQ_VERDICT_MODEL` | Groq model ids | see `.env.example` |
| `GROQ_TIMEOUT`, `GROQ_MAX_RETRIES` | Request timeout (s) and per-key retries | 20, 2 |
| `GROQ_IMAGE_MAX_DIM`, `GROQ_IMAGE_JPEG_QUALITY` | Longest edge and JPEG quality for images sent to any vision provider | 1024, 78 |
| `MAX_UPLOAD_BYTES` | Reject larger uploads with HTTP 413 | 26214400 (25 MB) |
| `CORS_ORIGINS` | Comma-separated allowed origins for a separately hosted frontend | localhost 5173/8000 |
| `CLASSIFY_FULL_FRAME` | Classify the whole frame (keep `true`) | true |
| `CLASSIFIER_ENSEMBLE` | Average ResNet-50 with EfficientNet-B0 | true |
| `CLASSIFIER_UNCERTAINTY_THRESHOLD` | Below this top-1 probability a result is uncertain | 0.40 |
| `DETECTION_MODEL_PATH`, `CLASSIFICATION_MODEL_PATH`, `EFFICIENTNET_MODEL_PATH`, `EXTERIOR_MODEL_PATH`, `CLASSES_PATH` | Model file overrides | `models/...` |

## Vision providers and failover

```
Gemini Flash  ->  Groq (key 1, then key 2)  ->  local ResNet-50 + EfficientNet-B0 only
```

- **One combined request per image** returns identification, exterior notes and interior/engine analysis as a single
  JSON object (`src/vision_service.py`). The old per-purpose methods are views over that one cached result, so there
  are no repeat API calls for the same image.
- The image is resized (max 1024 px, JPEG q78) only for the API call; local models and the browser preview use the
  original.
- **Cache:** results are cached by a hash of the optimised image (`VISION_CACHE_SIZE`, `VISION_CACHE_TTL`).
  Concurrent branches share one call. Failures are cached for 20 s so a down provider is not hammered.
- **Failover:** each provider gets at most one attempt per image (Groq also fails over between its two keys, and
  waits once if both are limited for <= 15 s). Quota/rate limits, timeouts, auth and server errors put a provider on a
  cooldown (Google's/Groq's own retry hint when given, otherwise 15 s to 10 min), and later requests skip it.
- If every provider fails, classifier and exterior results are still returned and `identification_source` is
  `classifier_only`. `final_ai_verdict.provider_attempts` lists each provider tried and why it failed.
- `GET /health` shows provider status (configured, cooldown, ok/failed counts), never keys. Keys are read only from
  the environment and are scrubbed from logs and errors.

## API

- `POST /predict` multipart `file` (JPEG/PNG/WEBP/BMP, up to 25 MB). Optional query: `enable_groq_inspection`,
  `enable_final_verdict`, `market_region`, `job_id` (for `/progress`).
- `GET /progress/{job_id}`, `GET /health`, interactive docs at `/docs`.

## Tests

```bash
python -m pytest tests -q          # backend (includes mocked provider-chain and failover tests, no API quota used)
cd frontend && npm run lint && npm run build
```

## Repository hygiene

`.env`, datasets (Stanford Cars archives and folders), zips and local debug images are git-ignored. Model weights
under `models/` are tracked (largest is about 96 MB; GitHub's hard limit is 100 MB per file).

## Known limitations

- Classifier coverage is the 196 Stanford Cars classes; accuracy on unseen photos is about 37% top-1 / 71% top-5.
- Specifications and valuations exist only for models present in `data/automotive_reference_data.json`.
- Free-tier quotas (Gemini and Groq) can run out; the app then reports classifier-only results and says so.
- Gemini's year estimates can be a few years off, and its identification has no calibrated confidence. Treat it as an informed guess.
- Provider output is not verified against any database.
