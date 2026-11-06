"""Comprehensive tests for AutoVision AI Multi-Model Architecture, Failovers, and Consolidated Report."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from PIL import Image

from src.config import (
    EXTERIOR_MODEL_PATH,
    EXTERIOR_CLASSES_PATH,
    CLASSIFIER_UNCERTAINTY_THRESHOLD,
)
from src.exterior_model import ExteriorComponentDetector, ExteriorSegmentationResult
from src.groq_service import GroqService, GroqServiceError
from src.branches.vehicle_id import VehicleIdentificationBranch
from src.branches.exterior import ExteriorIntelligenceBranch
from src.branches.interior_engine import InteriorEngineBranch
from src.orchestrator import MultiModelOrchestrator
from src.pipeline import VehicleRecognitionPipeline


@pytest.fixture(scope="session")
def sample_car_crop():
    """Load real vehicle crop if present, or synthetic image."""
    crop_path = Path("predictions") / "crops" / "sample_0_crop_0.png"
    if crop_path.exists():
        return np.array(Image.open(crop_path).convert("RGB"), dtype=np.uint8)
    # Generate realistic dummy vehicle-sized array
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[100:250, 50:350] = [120, 140, 180]
    return arr


# -----------------------------------------------------------------------------
# 1. Branch 1 Working Independently
# -----------------------------------------------------------------------------
def test_branch_1_independent(sample_car_crop):
    """Test Branch 1 vehicle identification runs independently and outputs standard structure."""
    branch1 = VehicleIdentificationBranch()
    res = branch1.analyze(
        image_rgb=sample_car_crop,
        conf_threshold=0.25,
        apply_preprocessing=True,
        include_classical_cv=True,
    )

    assert "status" in res
    assert res["status"] in ["success", "uncertain", "fallback"]
    assert "predictions" in res
    assert len(res["predictions"]) >= 1
    pred = res["predictions"][0]
    assert "make" in pred
    assert "model" in pred
    assert "year" in pred
    assert "classification_confidence" in pred
    assert "top5" in pred
    assert len(pred["top5"]) == 5
    assert "evidence" in res
    assert "processing_time_ms" in res
    assert res["processing_time_ms"] > 0


# -----------------------------------------------------------------------------
# 2. Branch 2 Working Independently with YOLO11n-seg Checkpoint
# -----------------------------------------------------------------------------
def test_branch_2_independent_exterior_model(sample_car_crop):
    """Test Branch 2 exterior intelligence using exterior_best.pt checkpoint."""
    branch2 = ExteriorIntelligenceBranch(conf_threshold=0.10)
    res = branch2.analyze(image_rgb=sample_car_crop, enable_groq_inspection=False)

    assert "status" in res
    assert res["status"] in ["success", "no_components_detected", "partial"]
    assert "predictions" in res
    preds = res["predictions"]
    assert "components" in preds
    assert "component_counts" in preds
    assert "detected_classes" in preds
    assert "possible_modifications" in preds

    # Strictly check that all detected classes belong to the 23 Carparts-Seg classes
    supported_classes = set(ExteriorComponentDetector.SUPPORTED_CLASSES_23)
    for c in preds["components"]:
        assert c["name"] in supported_classes
        assert len(c["box"]) == 4
        assert 0.0 <= c["confidence"] <= 1.0
        assert "area_ratio" in c
        assert "polygon_points_count" in c

    assert "evidence" in res
    assert "orientation_hints" in res["evidence"]
    assert "processing_time_ms" in res


# -----------------------------------------------------------------------------
# 3. Branch 3 Working Independently
# -----------------------------------------------------------------------------
def test_branch_3_independent_without_groq(sample_car_crop):
    """Test Branch 3 handles missing Groq gracefully without throwing exceptions."""
    groq_service = GroqService(primary_api_key="", backup_api_key="")
    branch3 = InteriorEngineBranch(groq_service=groq_service)
    res = branch3.analyze(sample_car_crop)

    assert res["status"] == "unavailable"
    assert "predictions" in res
    assert res["predictions"]["is_interior_visible"] is False
    assert res["predictions"]["is_engine_visible"] is False
    assert "evidence" in res
    assert len(res["errors"]) > 0


# -----------------------------------------------------------------------------
# 4. All Three Branches Running Concurrently
# -----------------------------------------------------------------------------
def test_all_branches_concurrent_execution(sample_car_crop):
    """Test MultiModelOrchestrator executes all branches concurrently and consolidates."""
    orchestrator = MultiModelOrchestrator()
    res = orchestrator.analyze_sync(
        image_rgb=sample_car_crop,
        enable_groq_inspection=False,
        enable_final_verdict=False,
    )

    assert "vehicle" in res
    assert "exterior_analysis" in res
    assert "interior_analysis" in res
    assert "visible_engine_info" in res
    assert "confidence_and_uncertainty" in res
    assert "original_model_outputs" in res
    assert "branch_1_vehicle_id" in res["original_model_outputs"]
    assert "branch_2_exterior" in res["original_model_outputs"]
    assert "branch_3_interior_engine" in res["original_model_outputs"]
    assert res["total_processing_time_ms"] > 0


@pytest.mark.asyncio
async def test_all_branches_async_concurrent_execution(sample_car_crop):
    """Test async execution of all branches concurrently."""
    orchestrator = MultiModelOrchestrator()
    res = await orchestrator.analyze_async(
        image_rgb=sample_car_crop,
        enable_groq_inspection=False,
        enable_final_verdict=False,
    )

    assert res["status"] in ["success", "partial"]
    assert res["vehicle"]["make"] != ""
    assert "exterior_analysis" in res


# -----------------------------------------------------------------------------
# 5. Primary Groq Failure and Backup Activation
# -----------------------------------------------------------------------------
def test_primary_groq_failure_and_backup_activation(sample_car_crop):
    """Verify that when the primary Groq API fails, the service automatically fails over to the backup Groq API."""
    service = GroqService(primary_api_key="mock_primary_key", backup_api_key="mock_backup_key")

    mock_primary_client = MagicMock()
    mock_primary_client.chat.completions.create.side_effect = Exception("Primary Rate Limit / 429 Too Many Requests")

    mock_backup_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "view_perspective": "interior_cabin",
            "is_interior_visible": True,
            "is_engine_visible": False,
            "interior_components": [{"component": "steering_wheel", "visible": True, "details": "Sport leather wheel"}],
            "engine_components": [],
            "estimated_identity_cues": ["BMW M badge on steering wheel"],
            "uncertainty_assessment": "Clear interior view"
        })))
    ]
    mock_backup_client.chat.completions.create.return_value = mock_response

    service._primary_client = mock_primary_client
    service._backup_client = mock_backup_client

    result = service.analyze_interior_and_engine(sample_car_crop)

    assert result["status"] == "success"
    assert result["provider"] == "backup"
    assert result["is_interior_visible"] is True
    assert "steering_wheel" in str(result["interior_components"])
    # Verify primary was attempted and backup was called
    assert mock_primary_client.chat.completions.create.called
    assert mock_backup_client.chat.completions.create.called


# -----------------------------------------------------------------------------
# 6. Both Groq APIs Failing
# -----------------------------------------------------------------------------
def test_both_groq_apis_failing(sample_car_crop):
    """Verify that when both Groq APIs fail, the system continues with CV outputs and marks interior analysis unavailable."""
    service = GroqService(primary_api_key="mock_primary", backup_api_key="mock_backup")

    mock_primary = MagicMock()
    mock_primary.chat.completions.create.side_effect = Exception("Primary network timeout")

    mock_backup = MagicMock()
    mock_backup.chat.completions.create.side_effect = Exception("Backup authentication failed")

    service._primary_client = mock_primary
    service._backup_client = mock_backup

    orchestrator = MultiModelOrchestrator(groq_service=service)
    res = orchestrator.analyze_sync(sample_car_crop)

    # Core vehicle and exterior CV should remain intact
    assert res["status"] in ["success", "partial"]
    assert res["vehicle"]["make"] != ""
    assert res["exterior_analysis"]["status"] in ["success", "no_components_detected"]
    # Interior must be marked unavailable without crashing
    assert res["interior_analysis"]["status"] == "unavailable"
    assert res["interior_analysis"]["is_interior_visible"] is False
    # Original model outputs must be transparently preserved
    assert res["original_model_outputs"]["branch_1_vehicle_id"]["status"] in ["success", "uncertain", "fallback"]


# -----------------------------------------------------------------------------
# 7. Exterior Model Failure Handling
# -----------------------------------------------------------------------------
def test_exterior_model_failure_handling(sample_car_crop):
    """Verify that an exterior model failure is isolated and does not disrupt vehicle identification or interior analysis."""
    mock_detector = MagicMock()
    mock_detector.detect_and_segment.side_effect = RuntimeError("CUDA Out of Memory in YOLO11n-seg")

    branch2 = ExteriorIntelligenceBranch(exterior_detector=mock_detector)
    orchestrator = MultiModelOrchestrator(branch_2=branch2)

    res = orchestrator.analyze_sync(sample_car_crop, enable_groq_inspection=False)

    assert res["exterior_analysis"]["status"] == "error"
    # Vehicle identification from Branch 1 must still succeed
    assert res["vehicle"]["make"] != ""
    assert len(res["original_model_outputs"]["branch_1_vehicle_id"]["predictions"]) >= 1


# -----------------------------------------------------------------------------
# 8. Vehicle Classifier Uncertainty Triggering Visual AI
# -----------------------------------------------------------------------------
def test_vehicle_classifier_uncertainty_triggering_visual_ai(sample_car_crop):
    """Verify that low classifier confidence triggers uncertainty flag and invokes visual AI reasoning."""
    mock_classifier = MagicMock()
    from src.classifier import ClassificationResult, PredictionItem
    # Force low confidence (0.22 < 0.40 threshold)
    low_conf_pred = PredictionItem(
        class_id=10,
        label="BMW 3 Series Sedan 2012",
        make="BMW",
        model="3 Series Sedan",
        year=2012,
        confidence=0.22,
    )
    alt_pred = PredictionItem(class_id=11, label="Audi A4 Sedan 2012", make="Audi", model="A4 Sedan", year=2012, confidence=0.20)
    mock_classifier.predict.return_value = ClassificationResult(
        top1=low_conf_pred,
        top5=[low_conf_pred, alt_pred, alt_pred, alt_pred, alt_pred],
    )

    mock_groq = MagicMock()
    mock_groq.is_available = True
    mock_groq.reason_vehicle_identity.return_value = {
        "status": "success",
        "visual_cues": ["Kidney grille visible", "Halo headlights"],
        "agreement_with_top1": True,
        "confidence_assessment": "medium",
        "reasoning": "Visible kidney grille strongly supports BMW 3 Series despite low classifier score.",
    }

    branch1 = VehicleIdentificationBranch(
        classifier=mock_classifier,
        groq_service=mock_groq,
        uncertainty_threshold=0.40,
    )
    res = branch1.analyze(sample_car_crop)

    assert res["status"] == "uncertain"
    assert res["predictions"][0]["is_uncertain"] is True
    # Verify visual AI reasoning was called
    assert mock_groq.reason_vehicle_identity.called
    assert "visual_ai_reasoning" in res["predictions"][0]
    assert "Kidney grille visible" in res["predictions"][0]["visual_ai_reasoning"]["visual_cues"]


# -----------------------------------------------------------------------------
# 9. Final Verdict Failure Preserves Original Model Outputs
# -----------------------------------------------------------------------------
def test_final_verdict_failure_preserves_original_outputs(sample_car_crop):
    """Verify that when final Groq verdict fails, original computer vision predictions are preserved without modification."""
    mock_groq = MagicMock()
    mock_groq.is_available = True
    mock_groq.generate_final_verdict.side_effect = GroqServiceError("Final verdict Groq API timeout")

    orchestrator = MultiModelOrchestrator(groq_service=mock_groq)
    res = orchestrator.analyze_sync(sample_car_crop)

    verdict = res["final_ai_verdict"]
    assert verdict["verdict_status"] == "fallback_to_original"
    assert len(verdict["corrections_made"]) == 0
    # Original predictions must be intact
    assert res["vehicle"]["make"] != ""
    assert res["original_model_outputs"]["branch_1_vehicle_id"]["predictions"][0]["make"] == res["vehicle"]["make"]


# -----------------------------------------------------------------------------
# 10. Successful Consolidated Report Generation (All Required Sections)
# -----------------------------------------------------------------------------
def test_successful_consolidated_report_generation(sample_car_crop):
    """Verify that the final output includes every section mandated by Section 6."""
    mock_groq = MagicMock()
    mock_groq.is_available = True
    mock_groq.generate_final_verdict.return_value = {
        "final_make": "Ford",
        "final_model": "F-150",
        "final_estimated_year": 2018,
        "final_generation": "13th Gen",
        "corrections_made": [],
        "cross_model_disagreements": ["None: exterior components align with pickup body style."],
        "confirmed_facts": ["Tailgate detected", "Back bumper detected"],
        "visual_inferences": ["Likely XLT trim based on wheel style"],
        "missing_or_unavailable_information": ["Engine bay not visible"],
        "verdict_summary": "Verified Ford F-150 (2018, 13th Gen).",
    }

    orchestrator = MultiModelOrchestrator(groq_service=mock_groq)
    report = orchestrator.analyze_sync(sample_car_crop)

    # Validate all 12 required sections
    assert "vehicle" in report
    assert report["vehicle"]["make"] == "Ford"
    assert report["vehicle"]["model"] == "F-150"
    assert report["vehicle"]["estimated_year"] == 2018
    assert report["vehicle"]["generation"] == "13th Gen"

    assert "exterior_analysis" in report
    assert "components" in report["exterior_analysis"]
    assert "possible_modifications" in report["exterior_analysis"]

    assert "interior_analysis" in report
    assert "visible_engine_info" in report
    assert "confidence_and_uncertainty" in report
    assert "cross_model_disagreements" in report
    assert "final_ai_verdict" in report
    assert "corrections_made" in report["final_ai_verdict"]
    assert "confirmed_facts" in report["final_ai_verdict"]
    assert "visual_inferences" in report["final_ai_verdict"]

    assert "missing_or_unavailable_information" in report
    # Ensure specs, performance, mileage, valuation are explicitly marked unassessed (not fabricated)
    missing_str = " ".join(report["missing_or_unavailable_information"]).lower()
    assert "specifications" in missing_str or "horsepower" in missing_str
    assert "performance" in missing_str or "dyno" in missing_str
    assert "mileage" in missing_str
    assert "valuation" in missing_str or "pricing" in missing_str

    assert "original_model_outputs" in report


# -----------------------------------------------------------------------------
# 11. Missing Model Checkpoint Identification
# -----------------------------------------------------------------------------
def test_missing_model_checkpoint_identification():
    """Verify that pointing to a missing checkpoint raises a clear FileNotFoundError naming the file."""
    with pytest.raises(FileNotFoundError, match="non_existent_weights.pt"):
        ExteriorComponentDetector(model_path=Path("non_existent_weights.pt"))


# -----------------------------------------------------------------------------
# 12. Invalid or Unsupported Image Handling
# -----------------------------------------------------------------------------
def test_invalid_image_handling():
    """Verify pipeline rejects corrupt bytes without crashing."""
    pipeline = VehicleRecognitionPipeline()
    with pytest.raises(ValueError, match="empty or too small"):
        pipeline.validate_and_decode_image(b"short")

    with pytest.raises(ValueError, match="Failed to decode"):
        pipeline.validate_and_decode_image(b"012345678901234567890123456789")
