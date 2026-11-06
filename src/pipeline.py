"""End-to-End Vehicle Recognition Pipeline with Multi-Model Architecture, Quality Enhancement, and Prediction Comparison."""

import io
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

import numpy as np
import cv2
from PIL import Image

from .config import (
    DEFAULT_DETECTION_CONF,
    DEFAULT_EXTERIOR_CONF,
    ENABLE_PREDICTION_COMPARISON,
    COMPARE_GROQ_VERSIONS,
    PREDICTION_COMPARISON_MODE,
)
from .detector import VehicleDetector, DetectionResult
from .classifier import VehicleClassifier, ClassificationResult
from .preprocessing import AdaptivePreprocessor
from .classical_cv import ClassicalCVAnalyzer
from .exterior_model import ExteriorComponentDetector
from .groq_service import GroqService
from .quality_assessment import ImageQualityAssessor
from .adaptive_enhancer import AdaptiveEnhancer
from .prediction_comparator import PredictionComparator
from .vehicle_intelligence import VehicleIntelligenceService
from .branches.vehicle_id import VehicleIdentificationBranch
from .branches.exterior import ExteriorIntelligenceBranch
from .branches.interior_engine import InteriorEngineBranch
from .orchestrator import MultiModelOrchestrator


class VehicleRecognitionPipeline:
    """
    Integrated Vehicle Recognition Pipeline combining:
    - Pre-Inference: Intelligent Image Quality Assessment and Adaptive Enhancement (Part 2)
    - Part 3: Original vs Enhanced Prediction Comparison & Intelligent Image Selection
    - Branch 1: Vehicle Detection (YOLOv8) & Fine-Grained Classification (ResNet-50) with Classical CV
    - Branch 2: Exterior Component Detection & Segmentation (YOLO11n-seg)
    - Branch 3: Interior & Engine Cabin Intelligence (Groq Vision Primary + Backup)
    - Final Groq Verdict Engine for consolidated cross-branch synthesis and verification.
    """

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        classifier: Optional[VehicleClassifier] = None,
        preprocessor: Optional[AdaptivePreprocessor] = None,
        classical_analyzer: Optional[ClassicalCVAnalyzer] = None,
        exterior_detector: Optional[ExteriorComponentDetector] = None,
        groq_service: Optional[GroqService] = None,
        quality_assessor: Optional[ImageQualityAssessor] = None,
        adaptive_enhancer: Optional[AdaptiveEnhancer] = None,
        prediction_comparator: Optional[PredictionComparator] = None,
        vehicle_intelligence_service: Optional[VehicleIntelligenceService] = None,
        orchestrator: Optional[MultiModelOrchestrator] = None,
    ):
        self.detector = detector or VehicleDetector()
        self.classifier = classifier or VehicleClassifier()
        self.preprocessor = preprocessor or AdaptivePreprocessor()
        self.classical_analyzer = classical_analyzer or ClassicalCVAnalyzer()
        self.groq_service = groq_service or GroqService()
        self.quality_assessor = quality_assessor or ImageQualityAssessor()
        self.adaptive_enhancer = adaptive_enhancer or AdaptiveEnhancer(assessor=self.quality_assessor)
        self.prediction_comparator = prediction_comparator or PredictionComparator()
        self.vehicle_intelligence_service = vehicle_intelligence_service or VehicleIntelligenceService()

        # Try to initialize exterior detector gracefully
        self.exterior_detector = exterior_detector
        if self.exterior_detector is None:
            try:
                self.exterior_detector = ExteriorComponentDetector()
            except Exception as e:
                self.exterior_detector = None

        # Build branches
        self.branch_1 = VehicleIdentificationBranch(
            detector=self.detector,
            classifier=self.classifier,
            preprocessor=self.preprocessor,
            classical_analyzer=self.classical_analyzer,
            groq_service=self.groq_service,
        )
        self.branch_2 = ExteriorIntelligenceBranch(
            exterior_detector=self.exterior_detector,
            groq_service=self.groq_service,
        )
        self.branch_3 = InteriorEngineBranch(
            groq_service=self.groq_service,
        )

        # Multi-model orchestrator
        self.orchestrator = orchestrator or MultiModelOrchestrator(
            branch_1=self.branch_1,
            branch_2=self.branch_2,
            branch_3=self.branch_3,
            groq_service=self.groq_service,
            quality_assessor=self.quality_assessor,
            adaptive_enhancer=self.adaptive_enhancer,
            prediction_comparator=self.prediction_comparator,
            vehicle_intelligence_service=self.vehicle_intelligence_service,
        )

    @staticmethod
    def validate_and_decode_image(image_input: Union[bytes, np.ndarray, Image.Image, Path, str]) -> np.ndarray:
        """
        Validate input format, integrity, and dimensions, then return as RGB numpy array.

        Args:
            image_input: Raw image bytes, PIL Image, numpy array, or file path.

        Returns:
            RGB numpy array of shape (H, W, 3) with dtype uint8.

        Raises:
            ValueError: If image is empty, corrupt, or invalid dimensions.
            TypeError: If input type is not supported.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise ValueError(f"Image file does not exist: {path}")
            with open(path, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image_input, bytes):
            image_bytes = image_input
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input image array is empty.")
            if image_input.ndim == 2:
                # Grayscale to RGB
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    # RGBA to RGB
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                elif image_input.shape[2] == 3:
                    return image_input.copy()
            raise ValueError(f"Unsupported image array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if not image_bytes or len(image_bytes) < 16:
            raise ValueError("Input image bytes are empty or too small.")

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()  # Check file integrity
            # Reopen after verify
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image data: {str(e)}") from e

        rgb_arr = np.array(pil_img, dtype=np.uint8)
        h, w = rgb_arr.shape[:2]

        if h < 16 or w < 16:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small (minimum 16x16 required).")
        if h > 10000 or w > 10000:
            raise ValueError(f"Image dimensions ({w}x{h}) exceed maximum allowed size (10000x10000).")

        return rgb_arr

    def process(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image, Path, str],
        conf_threshold: float = DEFAULT_DETECTION_CONF,
        apply_preprocessing: bool = True,
        include_classical_cv: bool = True,
        fallback_to_full_image: bool = True,
        enable_multimodal: bool = True,
        enable_groq_inspection: bool = True,
        enable_final_verdict: bool = True,
        exterior_conf_threshold: Optional[float] = None,
        enable_quality_enhancement: bool = True,
        enable_prediction_comparison: Optional[bool] = None,
        compare_groq_versions: Optional[bool] = None,
        comparison_mode: Optional[str] = None,
        market_region: Optional[str] = None,
        mileage: Optional[int] = None,
        vehicle_condition: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute vehicle recognition pipeline synchronously.
        Supports Part 2 Quality Enhancement, Part 3 Prediction Comparison,
        and Part 4 Specifications & Market Price Intelligence.
        """
        rgb_image = self.validate_and_decode_image(image_input)

        if enable_multimodal:
            return self.orchestrator.analyze_sync(
                image_rgb=rgb_image,
                conf_threshold=conf_threshold,
                apply_preprocessing=apply_preprocessing,
                include_classical_cv=include_classical_cv,
                fallback_to_full_image=fallback_to_full_image,
                enable_groq_inspection=enable_groq_inspection,
                enable_final_verdict=enable_final_verdict,
                exterior_conf_threshold=exterior_conf_threshold,
                enable_quality_enhancement=enable_quality_enhancement,
                enable_prediction_comparison=enable_prediction_comparison,
                compare_groq_versions=compare_groq_versions,
                comparison_mode=comparison_mode,
                market_region=market_region,
                mileage=mileage,
                vehicle_condition=vehicle_condition,
                currency=currency,
            )
        else:
            # Legacy branch 1 direct execution
            start_time = time.perf_counter()
            b1 = self.branch_1.analyze(
                image_rgb=rgb_image,
                conf_threshold=conf_threshold,
                apply_preprocessing=apply_preprocessing,
                include_classical_cv=include_classical_cv,
                fallback_to_full_image=fallback_to_full_image,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            h, w = rgb_image.shape[:2]
            return {
                "status": "success",
                "image_dimensions": {"width": w, "height": h},
                "vehicles_detected": len(b1["predictions"]) if b1["status"] != "fallback" else 0,
                "predictions_count": len(b1["predictions"]),
                "processing_time_ms": round(elapsed_ms, 2),
                "predictions": b1["predictions"],
            }

    async def process_async(
        self,
        image_input: Union[bytes, np.ndarray, Image.Image, Path, str],
        conf_threshold: float = DEFAULT_DETECTION_CONF,
        apply_preprocessing: bool = True,
        include_classical_cv: bool = True,
        fallback_to_full_image: bool = True,
        enable_multimodal: bool = True,
        enable_groq_inspection: bool = True,
        enable_final_verdict: bool = True,
        exterior_conf_threshold: Optional[float] = None,
        enable_quality_enhancement: bool = True,
        enable_prediction_comparison: Optional[bool] = None,
        compare_groq_versions: Optional[bool] = None,
        comparison_mode: Optional[str] = None,
        market_region: Optional[str] = None,
        mileage: Optional[int] = None,
        vehicle_condition: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute vehicle recognition pipeline asynchronously.
        Non-blocking execution for FastAPI.
        """
        rgb_image = self.validate_and_decode_image(image_input)

        if enable_multimodal:
            return await self.orchestrator.analyze_async(
                image_rgb=rgb_image,
                conf_threshold=conf_threshold,
                apply_preprocessing=apply_preprocessing,
                include_classical_cv=include_classical_cv,
                fallback_to_full_image=fallback_to_full_image,
                enable_groq_inspection=enable_groq_inspection,
                enable_final_verdict=enable_final_verdict,
                exterior_conf_threshold=exterior_conf_threshold,
                enable_quality_enhancement=enable_quality_enhancement,
                enable_prediction_comparison=enable_prediction_comparison,
                compare_groq_versions=compare_groq_versions,
                comparison_mode=comparison_mode,
                market_region=market_region,
                mileage=mileage,
                vehicle_condition=vehicle_condition,
                currency=currency,
            )
        else:
            import asyncio
            b1 = await asyncio.to_thread(
                self.branch_1.analyze,
                image_rgb=rgb_image,
                conf_threshold=conf_threshold,
                apply_preprocessing=apply_preprocessing,
                include_classical_cv=include_classical_cv,
                fallback_to_full_image=fallback_to_full_image,
            )
            h, w = rgb_image.shape[:2]
            return {
                "status": "success",
                "image_dimensions": {"width": w, "height": h},
                "vehicles_detected": len(b1["predictions"]) if b1["status"] != "fallback" else 0,
                "predictions_count": len(b1["predictions"]),
                "processing_time_ms": b1["processing_time_ms"],
                "predictions": b1["predictions"],
            }
