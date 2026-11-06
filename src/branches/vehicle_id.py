"""Branch 1: Vehicle Identification (YOLO Detector + ResNet50 Classifier + Visual AI Fallback)."""

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

from ..config import (
    DEFAULT_DETECTION_CONF,
    CLASSIFIER_UNCERTAINTY_THRESHOLD,
)
from ..detector import VehicleDetector, DetectionResult
from ..classifier import VehicleClassifier, ClassificationResult
from ..preprocessing import AdaptivePreprocessor
from ..classical_cv import ClassicalCVAnalyzer
from ..groq_service import GroqService

logger = logging.getLogger("autovision.branch1.vehicle_id")


class VehicleIdentificationBranch:
    """
    Branch 1: Vehicle Identification
    - YOLOv8 vehicle detection.
    - ResNet50 fine-grained make/model/year classification.
    - Top-5 alternatives and confidence reporting.
    - Fallback to visual AI reasoning when classification is uncertain or detector misses.
    """

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        classifier: Optional[VehicleClassifier] = None,
        preprocessor: Optional[AdaptivePreprocessor] = None,
        classical_analyzer: Optional[ClassicalCVAnalyzer] = None,
        groq_service: Optional[GroqService] = None,
        uncertainty_threshold: float = CLASSIFIER_UNCERTAINTY_THRESHOLD,
    ):
        self.detector = detector or VehicleDetector()
        self.classifier = classifier or VehicleClassifier()
        self.preprocessor = preprocessor or AdaptivePreprocessor()
        self.classical_analyzer = classical_analyzer or ClassicalCVAnalyzer()
        self.groq_service = groq_service or GroqService()
        self.uncertainty_threshold = uncertainty_threshold

    def analyze(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = DEFAULT_DETECTION_CONF,
        apply_preprocessing: bool = True,
        include_classical_cv: bool = True,
        fallback_to_full_image: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Branch 1 analysis synchronously.

        Returns:
            Standardized structured dictionary:
            - status: "success" | "uncertain" | "fallback" | "error"
            - predictions: list of vehicle prediction entries
            - confidence: primary vehicle confidence
            - evidence: classical CV metrics, preprocessing info, visual AI reasoning
            - errors: list of error strings
            - processing_time_ms: execution time
        """
        start_time = time.perf_counter()
        errors: List[str] = []
        h, w = image_rgb.shape[:2]

        # 1. Vehicle Detection
        used_fallback = False
        detections: List[DetectionResult] = []
        try:
            detections = self.detector.detect(image_rgb, conf_threshold=conf_threshold)
        except Exception as det_err:
            logger.warning(f"Vehicle detector error: {det_err}. Attempting image-based fallback...")
            errors.append(f"Detector failed: {str(det_err)}")

        if len(detections) == 0 and fallback_to_full_image:
            detections = [
                DetectionResult(
                    box=[0, 0, w, h],
                    confidence=1.0,
                    class_id=2,  # car default
                    class_name="vehicle_fallback",
                    crop=image_rgb.copy(),
                )
            ]
            used_fallback = True

        if len(detections) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "error" if errors else "no_vehicle_detected",
                "predictions": [],
                "confidence": 0.0,
                "evidence": {"vehicles_detected": 0},
                "errors": errors or ["No vehicle detected in image."],
                "processing_time_ms": round(elapsed_ms, 2),
            }

        predictions_list: List[Dict[str, Any]] = []
        overall_uncertainty = False
        visual_reasoning_result = None

        for idx, det in enumerate(detections):
            crop = det.crop
            classified_crop = crop
            prep_report_dict = None

            # 2. Adaptive Preprocessing
            if apply_preprocessing:
                try:
                    classified_crop, prep_report = self.preprocessor.enhance(crop)
                    prep_report_dict = prep_report.to_dict()
                except Exception as prep_err:
                    errors.append(f"Preprocessing warning for vehicle {idx}: {prep_err}")

            # 3. Classical CV Feature Extraction
            classical_dict = None
            if include_classical_cv:
                try:
                    classical_res = self.classical_analyzer.extract_features(classified_crop)
                    classical_dict = classical_res.to_dict()
                except Exception as cv_err:
                    errors.append(f"Classical CV warning for vehicle {idx}: {cv_err}")

            # 4. Deep Learning Classification (ResNet-50)
            try:
                clf_result = self.classifier.predict(classified_crop, top_k=5)
                clf_dict = clf_result.to_dict()
            except Exception as clf_err:
                errors.append(f"ResNet50 classification failed for vehicle {idx}: {clf_err}")
                continue

            top1_conf = float(clf_dict["confidence"])
            is_uncertain = (top1_conf < self.uncertainty_threshold) or used_fallback
            if is_uncertain:
                overall_uncertainty = True

            # 5. Visual AI Reasoning Fallback if Uncertain
            ai_reasoning = None
            if is_uncertain and self.groq_service and self.groq_service.is_available:
                try:
                    ai_reasoning = self.groq_service.reason_vehicle_identity(
                        image=classified_crop,
                        top1_prediction=clf_dict,
                        top5_predictions=clf_dict["top5"],
                    )
                    visual_reasoning_result = ai_reasoning
                except Exception as groq_err:
                    errors.append(f"Visual reasoning fallback failed: {groq_err}")

            pred_entry = {
                "vehicle_index": idx,
                "box": det.box,
                "detection_confidence": round(det.confidence, 4),
                "detection_class": det.class_name,
                "is_fallback": used_fallback,
                "make": clf_dict["make"],
                "model": clf_dict["model"],
                "year": clf_dict["year"],
                "label": clf_dict["label"],
                "classification_confidence": round(top1_conf, 4),
                "is_uncertain": is_uncertain,
                "top5": [
                    {
                        "class_id": item["class_id"],
                        "make": item["make"],
                        "model": item["model"],
                        "year": item["year"],
                        "label": item["label"],
                        "confidence": round(item["confidence"], 4),
                    }
                    for item in clf_dict["top5"]
                ],
            }

            if prep_report_dict is not None:
                pred_entry["preprocessing"] = prep_report_dict
            if classical_dict is not None:
                pred_entry["classical_cv"] = classical_dict
            if ai_reasoning is not None:
                pred_entry["visual_ai_reasoning"] = ai_reasoning

            predictions_list.append(pred_entry)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not predictions_list:
            status = "error"
        elif overall_uncertainty:
            status = "uncertain"
        elif used_fallback:
            status = "fallback"
        else:
            status = "success"

        primary_confidence = predictions_list[0]["classification_confidence"] if predictions_list else 0.0

        evidence = {
            "vehicles_detected": len(predictions_list) if not used_fallback else 0,
            "fallback_used": used_fallback,
            "uncertainty_threshold": self.uncertainty_threshold,
            "top1_confidence": primary_confidence,
        }
        if visual_reasoning_result:
            evidence["visual_ai_reasoning_summary"] = visual_reasoning_result.get("reasoning", "")
            evidence["visual_cues"] = visual_reasoning_result.get("visual_cues", [])

        return {
            "status": status,
            "predictions": predictions_list,
            "confidence": primary_confidence,
            "evidence": evidence,
            "errors": errors,
            "processing_time_ms": round(elapsed_ms, 2),
        }
