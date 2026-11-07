"""Low-memory behaviour: nothing heavy loads at startup; models load once, on first use."""

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def test_app_starts_and_health_works_without_importing_or_loading_any_ml_model():
    code = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "import api\n"
        "from fastapi.testclient import TestClient\n"
        "with TestClient(api.app) as c:\n"
        "    r = c.get('/health')\n"
        "print(json.dumps({'code': r.status_code, 'status': r.json()['status'], 'loaded': r.json()['models']['models_loaded'],\n"
        "  'torch': 'torch' in sys.modules, 'ultralytics': 'ultralytics' in sys.modules, 'pipeline': api.pipeline is not None}))\n"
    ) % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {"code": 200, "status": "healthy", "loaded": False, "torch": False, "ultralytics": False, "pipeline": False}


def test_health_keeps_the_same_response_keys():
    import api
    from fastapi.testclient import TestClient
    body = TestClient(api.app).get("/health").json()
    assert {"status", "timestamp", "device", "models"} <= set(body)
    assert {"detector_ready", "classifier_ready", "exterior_ready", "groq_primary_configured", "groq_backup_configured",
            "vision_providers", "num_classes"} <= set(body["models"])


def test_lazy_model_is_not_built_until_used_and_is_built_exactly_once_under_concurrency():
    from src.pipeline import LazyModel
    built = []

    class Heavy:
        def __init__(self):
            built.append(1)
            self.value = 42

    lazy = LazyModel(Heavy, "test model")
    assert built == [] and lazy.loaded is False
    results = []
    threads = [threading.Thread(target=lambda: results.append(lazy.value)) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [42] * 12 and len(built) == 1 and lazy.loaded is True
    assert lazy.value == 42 and len(built) == 1


def test_pipeline_construction_loads_no_models_and_models_are_loaded_once_then_reused():
    from src.pipeline import VehicleRecognitionPipeline
    pipe = VehicleRecognitionPipeline()
    pipe.branch_1.groq_service = None  # CV only: no network
    assert not pipe.detector.loaded and not pipe.classifier.loaded and not pipe.exterior_detector.loaded

    crop = ROOT / "predictions" / "crops" / "sample_0_crop_0.png"
    img = np.array(Image.open(crop).convert("RGB")) if crop.exists() else np.full((200, 300, 3), 120, np.uint8)
    first = pipe.process(img, enable_multimodal=False)
    det = pipe.detector._obj
    assert pipe.detector.loaded and not pipe.exterior_detector.loaded  # exterior not needed here
    # The ~100 MB classifier is released right after Branch 1 uses it (UNLOAD_CLASSIFIER_AFTER_USE) ...
    assert not pipe.classifier.loaded
    second = pipe.process(img, enable_multimodal=False)
    assert pipe.detector._obj is det  # ... while the detector is built once and reused,
    assert not pipe.classifier.loaded and second["predictions"], "and the classifier transparently reloads when needed"
    assert set(first) == set(second) and "predictions" in first and "processing_time_ms" in first
