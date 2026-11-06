"""API integration tests for FastAPI vehicle recognition endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app

client = TestClient(app)


def create_test_image_bytes(format="JPEG", size=(128, 128), color=(100, 150, 200)) -> bytes:
    """Generate in-memory valid image bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_health_endpoint():
    """Verify GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "models" in data
    assert data["models"]["detector"] == "YOLOv8n"
    assert data["models"]["classifier"] == "ResNet50"
    assert data["models"]["num_classes"] == 196


def test_predict_endpoint_success():
    """Verify POST /predict succeeds on a valid uploaded vehicle image."""
    img_bytes = create_test_image_bytes(format="JPEG")
    files = {"file": ("test_car.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "predictions" in data
    assert len(data["predictions"]) >= 1

    first_pred = data["predictions"][0]
    assert "make" in first_pred
    assert "model" in first_pred
    assert "year" in first_pred
    assert "classification_confidence" in first_pred
    assert "box" in first_pred
    assert len(first_pred["box"]) == 4
    assert len(first_pred["top5"]) == 5
    assert "preprocessing" in first_pred
    assert "classical_cv" in first_pred


def test_predict_endpoint_empty_file():
    """Verify POST /predict returns 400 on empty upload."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_endpoint_unsupported_media_type():
    """Verify POST /predict returns 415 on unsupported media type."""
    files = {"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")}
    response = client.post("/predict", files=files)
    assert response.status_code == 415


def test_predict_endpoint_corrupt_image():
    """Verify POST /predict returns 422 on corrupted image bytes."""
    corrupt_bytes = b"bad_image_data_not_real_image_bytes_here"
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 422
