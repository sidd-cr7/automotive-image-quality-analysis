"""Comprehensive Test Suite for Part 3: Prediction Comparison & Intelligent Image Selection."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.prediction_comparator import (
    PredictionComparator,
    calculate_box_iou,
    BranchComparisonResult,
    ComprehensiveComparisonReport,
)
from src.orchestrator import MultiModelOrchestrator
from src.groq_service import GroqService
from src.pipeline import VehicleRecognitionPipeline
from api import app

client = TestClient(app)


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Standard RGB dummy vehicle array."""
    arr = np.full((240, 320, 3), 128, dtype=np.uint8)
    arr[60:180, 50:270] = [20, 80, 160]
    return arr


# -----------------------------------------------------------------------------
# 1. Prediction Agreement Scenario
# -----------------------------------------------------------------------------
def test_predictions_agree_enhancement_selected():
    """When original and enhanced agree, confidence rises, and localization is stable -> selects enhanced."""
    comparator = PredictionComparator()

    orig_b1 = {
        "status": "success",
        "confidence": 0.61,
        "predictions": [{
            "make": "BMW",
            "model": "3 Series Sedan",
            "year": 2012,
            "box": [50, 60, 270, 180],
            "classification_confidence": 0.61,
            "detection_confidence": 0.85,
            "top5": [
                {"label": "BMW 3 Series Sedan 2012", "confidence": 0.61},
                {"label": "Audi A4 Sedan 2012", "confidence": 0.20},
            ],
        }]
    }

    enh_b1 = {
        "status": "success",
        "confidence": 0.74,
        "predictions": [{
            "make": "BMW",
            "model": "3 Series Sedan",
            "year": 2012,
            "box": [51, 60, 269, 181],  # High IoU
            "classification_confidence": 0.74,
            "detection_confidence": 0.91,
            "top5": [
                {"label": "BMW 3 Series Sedan 2012", "confidence": 0.74},
                {"label": "Audi A4 Sedan 2012", "confidence": 0.12},
            ],
        }]
    }

    res = comparator.compare_vehicle_identification(
        original_b1=orig_b1,
        enhanced_b1=enh_b1,
        quality_score_orig=68.0,
        quality_gain=8.5,
    )

    assert res.predictions_agree is True
    assert res.selected_version == "enhanced"
    assert res.comparison_score > 0.0
    assert "increased by +13.0%" in res.selection_reason
    assert res.metrics["primary_bbox_iou"] >= 0.90


# -----------------------------------------------------------------------------
# 2. Original Prediction is Better (Class Flip on Good Image)
# -----------------------------------------------------------------------------
def test_original_prediction_is_better():
    """When enhancement causes a class flip on an already good image -> penalizes and retains original."""
    comparator = PredictionComparator()

    orig_b1 = {
        "status": "success",
        "confidence": 0.82,
        "predictions": [{
            "make": "Ford",
            "model": "F-150",
            "year": 2018,
            "box": [30, 40, 290, 200],
            "classification_confidence": 0.82,
            "detection_confidence": 0.90,
            "top5": [{"label": "Ford F-150 2018", "confidence": 0.82}],
        }]
    }

    # Enhanced flipped prediction to Chevrolet Silverado
    enh_b1 = {
        "status": "success",
        "confidence": 0.65,
        "predictions": [{
            "make": "Chevrolet",
            "model": "Silverado",
            "year": 2017,
            "box": [30, 40, 290, 200],
            "classification_confidence": 0.65,
            "detection_confidence": 0.88,
            "top5": [{"label": "Chevrolet Silverado 2017", "confidence": 0.65}],
        }]
    }

    res = comparator.compare_vehicle_identification(
        original_b1=orig_b1,
        enhanced_b1=enh_b1,
        quality_score_orig=76.0,  # Good image
        quality_gain=1.0,
    )

    assert res.predictions_agree is False
    assert res.selected_version == "original"
    assert res.comparison_score < 0.0
    assert "prediction disagreement" in res.selection_reason


# -----------------------------------------------------------------------------
# 3. Enhanced Prediction is Better on Degraded Input
# -----------------------------------------------------------------------------
def test_enhanced_prediction_is_better_on_degraded_input():
    """When original was dark/uncertain and enhanced recovers decisive features -> selects enhanced."""
    comparator = PredictionComparator()

    # Original was uncertain on dark image
    orig_b1 = {
        "status": "uncertain",
        "confidence": 0.28,
        "predictions": [{
            "make": "Honda",
            "model": "Civic",
            "year": 2010,
            "box": [40, 50, 260, 170],
            "classification_confidence": 0.28,
            "detection_confidence": 0.50,
            "top5": [
                {"label": "Honda Civic 2010", "confidence": 0.28},
                {"label": "Toyota Corolla 2010", "confidence": 0.26},
            ],
        }]
    }

    # Enhanced restored exposure and gave decisive prediction
    enh_b1 = {
        "status": "success",
        "confidence": 0.78,
        "predictions": [{
            "make": "Honda",
            "model": "Civic",
            "year": 2010,
            "box": [40, 50, 260, 170],
            "classification_confidence": 0.78,
            "detection_confidence": 0.92,
            "top5": [
                {"label": "Honda Civic 2010", "confidence": 0.78},
                {"label": "Toyota Corolla 2010", "confidence": 0.10},
            ],
        }]
    }

    res = comparator.compare_vehicle_identification(
        original_b1=orig_b1,
        enhanced_b1=enh_b1,
        quality_score_orig=35.0,  # Poor quality image
        quality_gain=25.0,
    )

    assert res.selected_version == "enhanced"
    assert res.comparison_score > 0.0


# -----------------------------------------------------------------------------
# 4. Higher Confidence on Disagreeing Enhanced Image Does NOT Overrule Original
# -----------------------------------------------------------------------------
def test_enhanced_has_higher_confidence_but_disagrees():
    """Core rule: Higher confidence alone does NOT mean more accurate if class flipped on acceptable image."""
    comparator = PredictionComparator()

    orig_b1 = {
        "status": "success",
        "confidence": 0.65,
        "predictions": [{
            "make": "Audi",
            "model": "A4 Sedan",
            "year": 2014,
            "box": [50, 50, 270, 190],
            "classification_confidence": 0.65,
            "detection_confidence": 0.88,
            "top5": [{"label": "Audi A4 Sedan 2014", "confidence": 0.65}],
        }]
    }

    # Enhanced produces higher confidence (0.86) but changed class on an already good image
    enh_b1 = {
        "status": "success",
        "confidence": 0.86,
        "predictions": [{
            "make": "BMW",
            "model": "3 Series Sedan",
            "year": 2014,
            "box": [50, 50, 270, 190],
            "classification_confidence": 0.86,
            "detection_confidence": 0.90,
            "top5": [{"label": "BMW 3 Series Sedan 2014", "confidence": 0.86}],
        }]
    }

    res = comparator.compare_vehicle_identification(
        original_b1=orig_b1,
        enhanced_b1=enh_b1,
        quality_score_orig=75.0,  # Good image
        quality_gain=1.5,
        enhancements_applied=[{"technique": "CLAHE LAB Contrast Equalization"}],
    )

    assert res.predictions_agree is False
    # Must NOT select enhanced just because confidence is higher
    assert res.selected_version == "original"
    assert "prevent artifact-induced misclassification" in res.selection_reason


# -----------------------------------------------------------------------------
# 5. Segmentation Masks Remain Consistent
# -----------------------------------------------------------------------------
def test_segmentation_masks_consistent():
    """Verify mask consistency calculation across original and enhanced segmentations."""
    comparator = PredictionComparator()

    orig_b2 = {
        "status": "success",
        "confidence": 0.70,
        "predictions": {
            "components": [
                {"name": "front_bumper", "box": [50, 140, 260, 190], "confidence": 0.82, "area_ratio": 0.08},
                {"name": "hood", "box": [60, 90, 250, 145], "confidence": 0.75, "area_ratio": 0.12},
            ],
            "detected_classes": ["front_bumper", "hood"],
        }
    }

    enh_b2 = {
        "status": "success",
        "confidence": 0.78,
        "predictions": {
            "components": [
                {"name": "front_bumper", "box": [50, 139, 261, 191], "confidence": 0.88, "area_ratio": 0.08},
                {"name": "hood", "box": [59, 91, 251, 146], "confidence": 0.82, "area_ratio": 0.12},
            ],
            "detected_classes": ["front_bumper", "hood"],
        }
    }

    res = comparator.compare_exterior_segmentation(orig_b2, enh_b2)

    assert res.selected_version == "enhanced"
    assert res.metrics["mask_consistency_iou"] >= 0.90
    assert res.metrics["spurious_artifacts_flag"] is False


# -----------------------------------------------------------------------------
# 6. Enhancement Introduces False-Looking Edges
# -----------------------------------------------------------------------------
def test_enhancement_introduces_false_looking_edges():
    """Detect and penalize spurious low-confidence fragmented component detections."""
    comparator = PredictionComparator()

    orig_b2 = {
        "status": "success",
        "confidence": 0.72,
        "predictions": {
            "components": [
                {"name": "wheel", "box": [60, 150, 100, 190], "confidence": 0.80, "area_ratio": 0.03},
                {"name": "front_door", "box": [110, 80, 190, 160], "confidence": 0.75, "area_ratio": 0.09},
            ],
            "detected_classes": ["front_door", "wheel"],
        }
    }

    # Enhanced produced spurious noisy fragments
    enh_b2 = {
        "status": "success",
        "confidence": 0.38,
        "predictions": {
            "components": [
                {"name": "wheel", "box": [60, 150, 100, 190], "confidence": 0.78, "area_ratio": 0.03},
                {"name": "front_door", "box": [110, 80, 190, 160], "confidence": 0.72, "area_ratio": 0.09},
                # Spurious noisy fragments:
                {"name": "back_light", "box": [20, 20, 25, 25], "confidence": 0.21, "area_ratio": 0.001},
                {"name": "left_mirror", "box": [30, 30, 35, 35], "confidence": 0.22, "area_ratio": 0.001},
                {"name": "tailgate", "box": [40, 40, 45, 45], "confidence": 0.20, "area_ratio": 0.001},
            ],
            "detected_classes": ["back_light", "front_door", "left_mirror", "tailgate", "wheel"],
        }
    }

    res = comparator.compare_exterior_segmentation(orig_b2, enh_b2)

    assert res.selected_version == "original"
    assert res.metrics["spurious_artifacts_flag"] is True
    assert "spurious/noisy segmentation fragments" in res.selection_reason


# -----------------------------------------------------------------------------
# 7. Groq Primary Succeeds in Comparison
# -----------------------------------------------------------------------------
def test_groq_primary_succeeds_in_comparison(dummy_image):
    """Verify Groq primary client completes interior comparison successfully."""
    service = GroqService(primary_api_key="mock_key")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "view_perspective": "interior_cabin",
            "is_interior_visible": True,
            "is_engine_visible": False,
            "interior_components": [{"component": "steering_wheel", "visible": True, "details": "Sport leather wheel"}],
            "engine_components": [],
            "estimated_identity_cues": ["Audi rings on wheel"],
            "uncertainty_assessment": "Clear cabin view"
        })))
    ]
    mock_client.chat.completions.create.return_value = mock_resp
    service._primary_client = mock_client

    comparator = PredictionComparator()
    orig_b3 = service.analyze_interior_and_engine(dummy_image)
    enh_b3 = service.analyze_interior_and_engine(dummy_image)

    res = comparator.compare_interior_engine(orig_b3, enh_b3, quality_gain=2.0)
    assert res.branch_name == "interior_engine_analysis"
    assert res.selected_version in ["original", "enhanced"]


# -----------------------------------------------------------------------------
# 8. Groq Primary Fails and Backup Succeeds
# -----------------------------------------------------------------------------
def test_groq_primary_fails_and_backup_succeeds_in_comparison(dummy_image):
    """Verify failover to backup Groq client works seamlessly in comparison workflow."""
    service = GroqService(primary_api_key="mock_primary", backup_api_key="mock_backup")
    mock_primary = MagicMock()
    mock_primary.chat.completions.create.side_effect = Exception("Primary Quota Exhausted")

    mock_backup = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "view_perspective": "engine_bay",
            "is_interior_visible": False,
            "is_engine_visible": True,
            "interior_components": [],
            "engine_components": [{"component": "air_intake", "visible": True, "details": "V6 airbox"}],
            "estimated_identity_cues": [],
            "uncertainty_assessment": "Clear engine bay"
        })))
    ]
    mock_backup.chat.completions.create.return_value = mock_resp
    service._primary_client = mock_primary
    service._backup_client = mock_backup

    res = service.analyze_interior_and_engine(dummy_image)
    assert res["status"] == "success"
    assert res["provider"] == "backup"
    assert mock_backup.chat.completions.create.called


# -----------------------------------------------------------------------------
# 9. Both Images Effectively Identical (Fast Mode, Zero Duplicate Compute)
# -----------------------------------------------------------------------------
def test_both_images_effectively_identical_skips_duplicate_compute(dummy_image):
    """When enhanced image is identical to original, duplicate analysis is skipped."""
    orchestrator = MultiModelOrchestrator()

    report = orchestrator.analyze_sync(
        dummy_image,
        enable_quality_enhancement=False,  # Forces identical
        enable_prediction_comparison=True,
        enable_groq_inspection=False,
        enable_final_verdict=False,
    )

    comp = report["prediction_comparison"]
    assert comp["comparison_enabled"] is True
    assert comp["comparison_mode"] == "fast_identical"
    # All branches should select original
    assert comp["vehicle_identification"]["selected_version"] == "original"
    assert comp["exterior_analysis"]["selected_version"] == "original"


# -----------------------------------------------------------------------------
# 10. Comparison Fails and Original is Selected as Fallback
# -----------------------------------------------------------------------------
def test_comparison_fails_fallback_to_original():
    """If unexpected error occurs in comparison calculation, original is conservatively selected."""
    comparator = PredictionComparator()

    # Pass corrupt dictionary to simulate partial failure
    res = comparator.compare_vehicle_identification(
        original_b1={"predictions": [{"make": "Ford", "model": "Focus", "year": 2012, "classification_confidence": 0.70}]},
        enhanced_b1={"predictions": []},  # Empty enhanced
    )

    assert res.selected_version == "original"


# -----------------------------------------------------------------------------
# 11. Comparison Disabled Mode
# -----------------------------------------------------------------------------
def test_comparison_disabled_mode(dummy_image):
    """When enable_prediction_comparison=False, comparison is marked disabled."""
    orchestrator = MultiModelOrchestrator()

    report = orchestrator.analyze_sync(
        dummy_image,
        enable_prediction_comparison=False,
        enable_groq_inspection=False,
        enable_final_verdict=False,
    )

    comp = report["prediction_comparison"]
    assert comp["comparison_enabled"] is False
    assert comp["comparison_mode"] == "off"


# -----------------------------------------------------------------------------
# 12. One Branch Fails While Others Continue
# -----------------------------------------------------------------------------
def test_one_branch_fails_while_others_continue(dummy_image):
    """If exterior branch raises exception, vehicle identification comparison still succeeds."""
    mock_b2 = MagicMock()
    mock_b2.analyze.side_effect = RuntimeError("YOLO11n-seg GPU OOM")

    orchestrator = MultiModelOrchestrator(branch_2=mock_b2)

    report = orchestrator.analyze_sync(
        dummy_image,
        enable_groq_inspection=False,
        enable_final_verdict=False,
    )

    assert report["exterior_analysis"]["status"] == "error"
    # Vehicle identification must still be populated and compared
    assert report["vehicle"]["make"] != ""
    assert "prediction_comparison" in report
    assert report["prediction_comparison"]["vehicle_identification"]["selected_version"] in ["original", "enhanced"]


# -----------------------------------------------------------------------------
# 13. Model Reuse and Single Instance Verification
# -----------------------------------------------------------------------------
def test_model_reuse_and_no_duplicate_instances():
    """Verify orchestrator reuses single model instances without loading duplicates."""
    pipe = VehicleRecognitionPipeline()
    orch = pipe.orchestrator

    # Verify single instances shared across pipeline
    assert orch.branch_1.detector is pipe.detector
    assert orch.branch_1.classifier is pipe.classifier
    assert orch.branch_2.detector is pipe.exterior_detector
    assert orch.groq_service is pipe.groq_service


# -----------------------------------------------------------------------------
# 14. API Returns All Comparison Fields
# -----------------------------------------------------------------------------
def test_api_returns_all_comparison_fields():
    """Verify POST /predict returns prediction_comparison, selected_predictions, original_predictions."""
    img = Image.new("RGB", (200, 200), color=(100, 120, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    files = {"file": ("test_comparison.jpg", img_bytes, "image/jpeg")}
    response = client.post(
        "/predict?enable_prediction_comparison=true&enable_groq_inspection=false&enable_final_verdict=false",
        files=files,
    )

    assert response.status_code == 200
    data = response.json()

    assert "prediction_comparison" in data
    comp = data["prediction_comparison"]
    assert "vehicle_identification" in comp
    assert "exterior_analysis" in comp
    assert "interior_engine_analysis" in comp
    assert "selected_version" in comp["vehicle_identification"]

    assert "selected_predictions" in data
    assert "original_predictions" in data
    assert "enhanced_predictions" in data


# -----------------------------------------------------------------------------
# 15. Final Verdict Receives Correct Selected and Original Outputs
# -----------------------------------------------------------------------------
def test_final_verdict_receives_comparison_payload(dummy_image):
    """Verify Final Groq Verdict Engine receives the prediction comparison summary in its payload."""
    mock_groq = MagicMock()
    mock_groq.is_available = True
    mock_groq.generate_final_verdict.return_value = {
        "verdict_status": "verified",
        "final_make": "Honda",
        "final_model": "Accord",
        "final_estimated_year": 2015,
        "final_generation": "9th Gen",
        "corrections_made": [],
        "cross_model_disagreements": ["Original vs Enhanced agreed on Honda Accord."],
        "confirmed_facts": ["Vehicle confirmed"],
        "visual_inferences": [],
        "missing_or_unavailable_information": [],
        "verdict_summary": "Verified Honda Accord using comparison selection.",
    }

    orchestrator = MultiModelOrchestrator(groq_service=mock_groq)
    report = orchestrator.analyze_sync(dummy_image, enable_final_verdict=True)

    assert mock_groq.generate_final_verdict.called
    kwargs = mock_groq.generate_final_verdict.call_args[1]
    assert "prediction_comparison" in kwargs
    assert kwargs["prediction_comparison"] is not None
    assert report["final_ai_verdict"]["verdict_status"] == "verified"
