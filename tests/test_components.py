"""Unit tests for vehicle recognition components and pipeline."""

import pytest
import numpy as np
from PIL import Image

from src.config import (
    DETECTION_MODEL_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLASSES_PATH,
    NUM_CLASSES,
)
from src.detector import VehicleDetector
from src.classifier import VehicleClassifier
from src.preprocessing import AdaptivePreprocessor
from src.classical_cv import ClassicalCVAnalyzer
from src.pipeline import VehicleRecognitionPipeline


@pytest.fixture(scope="session")
def sample_crop():
    """Load or generate a sample RGB vehicle crop."""
    # Check if existing crop sample is available
    import os
    crop_path = os.path.join("predictions", "crops", "sample_0_crop_0.png")
    if os.path.exists(crop_path):
        return np.array(Image.open(crop_path).convert("RGB"), dtype=np.uint8)
    # Synthetic fallback crop
    return np.random.randint(50, 200, (100, 150, 3), dtype=np.uint8)


def test_model_files_exist():
    """Verify that all required weights and metadata files are present."""
    assert DETECTION_MODEL_PATH.exists(), f"Missing YOLO model at {DETECTION_MODEL_PATH}"
    assert CLASSIFICATION_MODEL_PATH.exists(), f"Missing ResNet50 model at {CLASSIFICATION_MODEL_PATH}"
    assert CLASSES_PATH.exists(), f"Missing classes mapping at {CLASSES_PATH}"


def test_classes_mapping_integrity():
    """Verify classes.json contains exactly 196 unique classes with make, model, year."""
    import json
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data.get("classes", [])
    assert len(classes) == NUM_CLASSES == 196
    
    unique_ids = set()
    for item in classes:
        assert "id" in item
        assert "label" in item
        assert "make" in item
        assert "model" in item
        assert "year" in item
        assert isinstance(item["year"], int)
        assert item["year"] >= 1990 and item["year"] <= 2025
        unique_ids.add(item["id"])

    assert len(unique_ids) == 196


def test_vehicle_detector_loading(sample_crop):
    """Test loading YOLOv8 vehicle detector."""
    detector = VehicleDetector()
    assert detector.model is not None
    assert 2 in detector.vehicle_class_ids  # car
    assert 7 in detector.vehicle_class_ids  # truck

    # Run detection on crop (might be empty or find vehicle)
    detections = detector.detect(sample_crop, conf_threshold=0.1)
    assert isinstance(detections, list)


def test_vehicle_classifier_loading(sample_crop):
    """Test ResNet50 classifier loading, output dimensions, and prediction."""
    classifier = VehicleClassifier()
    assert classifier.model is not None

    res = classifier.predict(sample_crop, top_k=5)
    assert res.top1 is not None
    assert len(res.top5) == 5
    assert res.top1.make != ""
    assert res.top1.model != ""
    assert res.top1.year > 0
    assert 0.0 <= res.top1.confidence <= 1.0


def test_adaptive_preprocessing(sample_crop):
    """Test adaptive preprocessing metrics and enhancement transforms."""
    preprocessor = AdaptivePreprocessor()
    metrics = preprocessor.compute_metrics(sample_crop)
    assert metrics.brightness >= 0
    assert metrics.contrast >= 0
    assert metrics.blur_score >= 0

    enhanced, report = preprocessor.enhance(sample_crop)
    assert enhanced.shape == sample_crop.shape
    assert len(report.applied_transforms) > 0
    assert report.processing_time_ms >= 0


def test_classical_cv_analysis(sample_crop):
    """Test Canny, Sobel, ORB, and HSV 99-d feature extraction."""
    analyzer = ClassicalCVAnalyzer()
    res = analyzer.extract_features(sample_crop)
    assert 0.0 <= res.edge_density <= 1.0
    assert res.sobel_gradient_mean >= 0
    assert res.orb_keypoint_count >= 0
    assert 0 <= res.dominant_hue <= 180
    assert len(res.feature_vector_99d) == 99


def test_pipeline_validation_error_handling():
    """Verify pipeline rejects corrupt, empty, and invalid inputs with clear errors."""
    pipeline = VehicleRecognitionPipeline()

    # Empty bytes
    with pytest.raises(ValueError, match="empty or too small"):
        pipeline.validate_and_decode_image(b"")

    # Corrupt bytes
    with pytest.raises(ValueError, match="Failed to decode"):
        pipeline.validate_and_decode_image(b"invalid_non_image_bytes_here_123456789")

    # Unsupported type
    with pytest.raises(TypeError):
        pipeline.validate_and_decode_image(12345)


def test_pipeline_end_to_end(sample_crop):
    """Test full pipeline execution on sample crop."""
    pipeline = VehicleRecognitionPipeline()
    res = pipeline.process(
        sample_crop,
        apply_preprocessing=True,
        include_classical_cv=True,
        fallback_to_full_image=True,
    )

    assert res["status"] == "success"
    assert res["predictions_count"] >= 1
    pred = res["predictions"][0]
    assert "make" in pred
    assert "model" in pred
    assert "year" in pred
    assert "box" in pred
    assert len(pred["box"]) == 4
    assert len(pred["top5"]) == 5
    assert "preprocessing" in pred
    assert "classical_cv" in pred
