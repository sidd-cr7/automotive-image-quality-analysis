"""Branch 2: Exterior Intelligence (YOLO11n-seg + Aftermarket Modifications & Groq Inspection)."""

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

from ..config import DEFAULT_EXTERIOR_CONF, DEFAULT_EXTERIOR_IOU
from ..exterior_model import ExteriorComponentDetector, ExteriorSegmentationResult
from ..groq_service import GroqService

logger = logging.getLogger("autovision.branch2.exterior")


class ExteriorIntelligenceBranch:
    """
    Branch 2: Exterior Intelligence
    - Integrates trained YOLO11n-seg checkpoint (exterior_best.pt).
    - Detects and segments supported exterior components (23 Carparts-Seg classes).
    - Extracts component names, bounding boxes, segmentation mask details, and confidence scores.
    - Inspects for potential aftermarket modifications when visually supported.
    - Strictly never invents unsupported classes for the CV model.
    - Allows Groq to inspect the exterior image when the CV model misses something.
    """

    def __init__(
        self,
        exterior_detector: Optional[ExteriorComponentDetector] = None,
        groq_service: Optional[GroqService] = None,
        conf_threshold: float = DEFAULT_EXTERIOR_CONF,
        iou_threshold: float = DEFAULT_EXTERIOR_IOU,
    ):
        self.detector = exterior_detector or ExteriorComponentDetector()
        self.groq_service = groq_service or GroqService()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def analyze(
        self,
        image_rgb: np.ndarray,
        enable_groq_inspection: bool = True,
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute Branch 2 analysis synchronously.

        Returns:
            Standardized structured dictionary:
            - status: "success" | "partial" | "no_components_detected" | "error"
            - predictions: dict of components, counts, detected_classes, possible_modifications
            - confidence: mean component detection confidence
            - evidence: visible panels, exterior orientation hints, Groq inspection cues
            - errors: list of error strings
            - processing_time_ms: execution time
        """
        start_time = time.perf_counter()
        errors: List[str] = []
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # 1. Execute YOLO11n-seg segmentation
        seg_result: Optional[ExteriorSegmentationResult] = None
        try:
            seg_result = self.detector.detect_and_segment(
                image=image_rgb,
                conf_threshold=conf,
                iou_threshold=self.iou_threshold,
            )
        except Exception as e:
            logger.error(f"Exterior segmentation model failed: {e}")
            errors.append(f"Exterior CV model failure: {str(e)}")

        components_data = []
        counts = {}
        detected_classes = []
        mean_conf = 0.0

        if seg_result is not None:
            components_data = [c.to_dict() for c in seg_result.components]
            counts = seg_result.component_counts
            detected_classes = seg_result.detected_classes
            mean_conf = seg_result.mean_confidence

        # 2. Aftermarket Modifications & Missed Cues Inspection (via Groq Vision)
        modifications: List[Dict[str, Any]] = []
        missed_cues: List[str] = []
        inspection_notes: str = ""

        # Trigger Groq exterior inspection if enabled and available
        # (especially if no components detected or to spot modifications)
        if enable_groq_inspection and self.groq_service and self.groq_service.is_available:
            try:
                inspection_res = self.groq_service.inspect_exterior(
                    image=image_rgb,
                    cv_detected_classes=detected_classes,
                )
                modifications = inspection_res.get("possible_modifications", [])
                missed_cues = inspection_res.get("missed_cues", [])
                inspection_notes = inspection_res.get("inspection_notes", "")
            except Exception as groq_err:
                errors.append(f"Groq exterior inspection warning: {groq_err}")

        # Determine Branch Status
        if errors and not components_data:
            status = "error"
        elif not components_data and not modifications:
            status = "no_components_detected"
        elif not components_data and modifications:
            status = "partial"
        else:
            status = "success"

        # Formulate orientation / angle evidence
        orientation_hints = []
        if "front_bumper" in counts or "hood" in counts:
            orientation_hints.append("Front-facing elements visible")
        if "back_bumper" in counts or "trunk" in counts or "tailgate" in counts:
            orientation_hints.append("Rear-facing elements visible")
        if any(k in counts for k in ["back_left_door", "front_left_door", "left_mirror"]):
            orientation_hints.append("Left flank visible")
        if any(k in counts for k in ["back_right_door", "front_right_door", "right_mirror"]):
            orientation_hints.append("Right flank visible")

        evidence = {
            "orientation_hints": orientation_hints,
            "distinct_classes_count": len(detected_classes),
            "total_components_detected": len(components_data),
            "cv_classes_strictly_enforced": True,
            "groq_inspection_performed": bool(inspection_notes),
            "inspection_notes": inspection_notes,
            "missed_cues_noted": missed_cues,
        }

        predictions = {
            "components": components_data,
            "component_counts": counts,
            "detected_classes": detected_classes,
            "possible_modifications": modifications,
        }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": status,
            "predictions": predictions,
            "confidence": round(mean_conf, 4),
            "evidence": evidence,
            "errors": errors,
            "processing_time_ms": round(elapsed_ms, 2),
        }
