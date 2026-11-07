"""Multi-Model Asynchronous Orchestrator and Consolidated Intelligence Engine.

Integrates:
- Part 2: Intelligent Image Quality Assessment and Adaptive Computer Vision Enhancement
- Part 3: Original vs Enhanced Prediction Comparison and Intelligent Image Selection
- Branch 1: Vehicle Identification (YOLO + ResNet-50 + Visual AI Reasoning fallback)
- Branch 2: Exterior Intelligence (YOLO11n-seg 23 classes + Modifications inspection)
- Branch 3: Interior and Engine Intelligence (Groq Vision with automatic backup failover)
- Final Groq Verdict Engine (cross-branch synthesis, verification, and factual report generation)
"""

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .config import (
    ENABLE_PREDICTION_COMPARISON,
    COMPARE_GROQ_VERSIONS,
    PREDICTION_COMPARISON_MODE,
)
from .quality_assessment import ImageQualityAssessor, QualityAssessmentResult
from .adaptive_enhancer import AdaptiveEnhancer, AdaptiveEnhancementResult
from .prediction_comparator import (
    PredictionComparator,
    ComprehensiveComparisonReport,
    BranchComparisonResult,
)
from .vehicle_intelligence import VehicleIntelligenceService
from .branches.vehicle_id import VehicleIdentificationBranch
from .branches.exterior import ExteriorIntelligenceBranch
from .branches.interior_engine import InteriorEngineBranch
from .groq_service import GroqService
from .vision_service import VisionService, CURRENT_FRAME, REQUEST_ID

from . import progress
logger = logging.getLogger("autovision.orchestrator")


class MultiModelOrchestrator:
    """
    Coordinates concurrent execution of the three perception branches with
    pre-inference Image Quality Assessment, Adaptive Enhancement, and
    Original vs Enhanced Prediction Comparison & Intelligent Image Selection.
    """

    def __init__(
        self,
        branch_1: Optional[VehicleIdentificationBranch] = None,
        branch_2: Optional[ExteriorIntelligenceBranch] = None,
        branch_3: Optional[InteriorEngineBranch] = None,
        groq_service: Optional[GroqService] = None,
        quality_assessor: Optional[ImageQualityAssessor] = None,
        adaptive_enhancer: Optional[AdaptiveEnhancer] = None,
        prediction_comparator: Optional[PredictionComparator] = None,
        vehicle_intelligence_service: Optional[VehicleIntelligenceService] = None,
    ):
        self.groq_service = groq_service or VisionService()
        self.branch_1 = branch_1 or VehicleIdentificationBranch(groq_service=self.groq_service)
        self.branch_2 = branch_2 or ExteriorIntelligenceBranch(groq_service=self.groq_service)
        self.branch_3 = branch_3 or InteriorEngineBranch(groq_service=self.groq_service)
        self.quality_assessor = quality_assessor or ImageQualityAssessor()
        self.adaptive_enhancer = adaptive_enhancer or AdaptiveEnhancer(assessor=self.quality_assessor)
        self.prediction_comparator = prediction_comparator or PredictionComparator()
        self.vehicle_intelligence_service = vehicle_intelligence_service or VehicleIntelligenceService()

    # -------------------------------------------------------------------------
    # Branch-Specific Image Static Selection (Part 2 fallback/baseline)
    # -------------------------------------------------------------------------
    @staticmethod
    def select_branch_images(
        original_image: np.ndarray,
        enhancement_result: AdaptiveEnhancementResult,
    ) -> Tuple[Dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
        """Determines the baseline image version per branch based on quality indicators."""
        issues = set(enhancement_result.assessment_before.detected_issues)
        success = enhancement_result.enhancement_success
        enhanced_available = (
            success
            and len(enhancement_result.enhancements_applied) > 0
            and enhancement_result.quality_gain >= -2.0
        )

        enhanced_img = enhancement_result.enhanced_image
        orig_img = original_image

        b1_triggers = {"underexposed", "overexposed", "low_contrast", "haze_or_fog", "letterbox_borders", "tilted_image"}
        if enhanced_available and (issues & b1_triggers) and enhancement_result.assessment_before.overall_score < 78.0:
            sel_b1 = "enhanced"
            img_b1 = enhanced_img
        else:
            sel_b1 = "original"
            img_b1 = orig_img

        b2_triggers = {"low_contrast", "blur", "defocus_blur", "motion_blur", "haze_or_fog", "letterbox_borders"}
        if enhanced_available and (issues & b2_triggers):
            sel_b2 = "enhanced"
            img_b2 = enhanced_img
        else:
            sel_b2 = "original"
            img_b2 = orig_img

        b3_triggers = {"underexposed", "uneven_illumination", "color_cast", "jpeg_compression_artifacts", "haze_or_fog"}
        if enhanced_available and (issues & b3_triggers):
            sel_b3 = "enhanced"
            img_b3 = enhanced_img
        else:
            sel_b3 = "original"
            img_b3 = orig_img

        selection_record = {
            "original_preserved": True,
            "enhanced_available": enhanced_available,
            "selected_for_vehicle_identification": sel_b1,
            "selected_for_exterior_analysis": sel_b2,
            "selected_for_interior_analysis": sel_b3,
        }

        return selection_record, img_b1, img_b2, img_b3

    # -------------------------------------------------------------------------
    # Safe Branch Execution
    # -------------------------------------------------------------------------
    def _execute_branch_1_safe(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float,
        apply_preprocessing: bool,
        include_classical_cv: bool,
        fallback_to_full_image: bool,
        enable_visual_reasoning: bool = True,
    ) -> Dict[str, Any]:
        """Safely execute Branch 1 without unhandled crash."""
        try:
            return self.branch_1.analyze(
                image_rgb=image_rgb,
                conf_threshold=conf_threshold,
                apply_preprocessing=apply_preprocessing,
                include_classical_cv=include_classical_cv,
                fallback_to_full_image=fallback_to_full_image,
                enable_visual_reasoning=enable_visual_reasoning,
            )
        except Exception as e:
            logger.error(f"Branch 1 unhandled exception: {e}")
            return {
                "status": "error",
                "predictions": [],
                "confidence": 0.0,
                "evidence": {},
                "errors": [f"Branch 1 unhandled failure: {str(e)}"],
                "processing_time_ms": 0.0,
            }

    def _execute_branch_2_safe(
        self,
        image_rgb: np.ndarray,
        enable_groq_inspection: bool,
        exterior_conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Safely execute Branch 2 without unhandled crash."""
        try:
            return self.branch_2.analyze(
                image_rgb=image_rgb,
                enable_groq_inspection=enable_groq_inspection,
                conf_threshold=exterior_conf_threshold,
            )
        except Exception as e:
            logger.error(f"Branch 2 unhandled exception: {e}")
            return {
                "status": "error",
                "predictions": {
                    "components": [],
                    "component_counts": {},
                    "detected_classes": [],
                    "possible_modifications": [],
                },
                "confidence": 0.0,
                "evidence": {},
                "errors": [f"Branch 2 unhandled failure: {str(e)}"],
                "processing_time_ms": 0.0,
            }

    def _execute_branch_3_safe(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Safely execute Branch 3 without unhandled crash."""
        try:
            return self.branch_3.analyze(image_rgb=image_rgb)
        except Exception as e:
            logger.error(f"Branch 3 unhandled exception: {e}")
            return {
                "status": "error",
                "predictions": {
                    "view_perspective": "unknown",
                    "is_interior_visible": False,
                    "is_engine_visible": False,
                    "interior_components": [],
                    "engine_components": [],
                    "estimated_identity_cues": [],
                },
                "confidence": 0.0,
                "evidence": {},
                "errors": [f"Branch 3 unhandled failure: {str(e)}"],
                "processing_time_ms": 0.0,
            }

    def _execute_verdict_safe(
        self,
        b1: Dict[str, Any],
        b2: Dict[str, Any],
        b3: Dict[str, Any],
        image_rgb: np.ndarray,
        enable_final_verdict: bool,
        prediction_comparison: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Safely execute final Groq verdict with fallback."""
        if not enable_final_verdict:
            return GroqService._build_verdict_fallback(
                b1, b2, b3, reason="Final Groq verification disabled."
            )
        try:
            return self.groq_service.generate_final_verdict(
                branch_1_result=b1,
                branch_2_result=b2,
                branch_3_result=b3,
                image=image_rgb,
                prediction_comparison=prediction_comparison,
            )
        except Exception as v_err:
            logger.warning(f"Final verdict execution error: {v_err}. Retaining original results.")
            return GroqService._build_verdict_fallback(
                b1, b2, b3, reason=f"Final verdict exception: {str(v_err)}"
            )

    # -------------------------------------------------------------------------
    # Synchronous Execution (Pytest / CLI / Scripts)
    # -------------------------------------------------------------------------
    def analyze_sync(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        apply_preprocessing: bool = True,
        include_classical_cv: bool = True,
        fallback_to_full_image: bool = True,
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
        Execute full image quality assessment, adaptive enhancement, Part 3 prediction comparison,
        and consolidated multi-model intelligence analysis synchronously.
        """
        start_time = time.perf_counter()
        h, w = image_rgb.shape[:2]
        progress.begin("detection")
        _frame_token = CURRENT_FRAME.set(image_rgb)  # single combined vision request shares this frame
        _request_token = REQUEST_ID.set(uuid.uuid4().hex)

        do_comparison = ENABLE_PREDICTION_COMPARISON if enable_prediction_comparison is None else enable_prediction_comparison
        do_groq_comp = COMPARE_GROQ_VERSIONS if compare_groq_versions is None else compare_groq_versions
        comp_mode = PREDICTION_COMPARISON_MODE if comparison_mode is None else comparison_mode

        # 1. Image Quality Assessment & Adaptive Enhancement
        assessment = self.quality_assessor.assess(image_rgb)
        if enable_quality_enhancement:
            try:
                enhancement = self.adaptive_enhancer.enhance(image_rgb, assessment=assessment)
            except Exception as q_err:
                logger.error(f"Adaptive enhancement failed: {q_err}. Using original image.")
                enhancement = AdaptiveEnhancementResult(
                    enhanced_image=image_rgb,
                    original_preserved=True,
                    enhancements_applied=[],
                    enhancement_success=False,
                    assessment_before=assessment,
                    assessment_after=assessment,
                    quality_gain=0.0,
                )
        else:
            enhancement = AdaptiveEnhancementResult(
                enhanced_image=image_rgb,
                original_preserved=True,
                enhancements_applied=[],
                enhancement_success=True,
                assessment_before=assessment,
                assessment_after=assessment,
                quality_gain=0.0,
            )

        baseline_versions, _, _, _ = self.select_branch_images(image_rgb, enhancement)
        images_identical = bool(
            not enhancement.enhancements_applied or
            np.array_equal(image_rgb, enhancement.enhanced_image)
        )

        # 2. Branch Inferences and Prediction Comparison
        if not do_comparison:
            # Comparison disabled: execute single pass using baseline routing
            sel_b1_img = enhancement.enhanced_image if baseline_versions["selected_for_vehicle_identification"] == "enhanced" else image_rgb
            sel_b2_img = enhancement.enhanced_image if baseline_versions["selected_for_exterior_analysis"] == "enhanced" else image_rgb
            sel_b3_img = enhancement.enhanced_image if baseline_versions["selected_for_interior_analysis"] == "enhanced" else image_rgb

            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="autovision-branch") as executor:
                fut1 = executor.submit(contextvars.copy_context().run, self._execute_branch_1_safe, sel_b1_img, conf_threshold, apply_preprocessing, include_classical_cv, fallback_to_full_image)
                fut2 = executor.submit(contextvars.copy_context().run, self._execute_branch_2_safe, sel_b2_img, enable_groq_inspection, exterior_conf_threshold)
                fut3 = executor.submit(contextvars.copy_context().run, self._execute_branch_3_safe, sel_b3_img)

                b1_selected = fut1.result()
                b2_selected = fut2.result()
                b3_selected = fut3.result()

            progress.end("detection")
            b1_orig, b1_enh = b1_selected, b1_selected
            b2_orig, b2_enh = b2_selected, b2_selected
            b3_orig, b3_enh = b3_selected, b3_selected

            comp_report = ComprehensiveComparisonReport(
                vehicle_identification=BranchComparisonResult(
                    branch_name="vehicle_identification",
                    selected_version=baseline_versions["selected_for_vehicle_identification"],
                    comparison_score=0.0,
                    selection_reason="Comparison disabled; selected via Part 2 baseline image routing.",
                    original_prediction=b1_selected.get("predictions", [{}])[0] if b1_selected.get("predictions") else {},
                    enhanced_prediction=b1_selected.get("predictions", [{}])[0] if b1_selected.get("predictions") else {},
                    predictions_agree=True,
                    metrics={},
                ),
                exterior_analysis=BranchComparisonResult(
                    branch_name="exterior_analysis",
                    selected_version=baseline_versions["selected_for_exterior_analysis"],
                    comparison_score=0.0,
                    selection_reason="Comparison disabled; selected via Part 2 baseline image routing.",
                    original_prediction=b2_selected.get("predictions", {}),
                    enhanced_prediction=b2_selected.get("predictions", {}),
                    predictions_agree=True,
                    metrics={},
                ),
                interior_engine_analysis=BranchComparisonResult(
                    branch_name="interior_engine_analysis",
                    selected_version=baseline_versions["selected_for_interior_analysis"],
                    comparison_score=0.0,
                    selection_reason="Comparison disabled; selected via Part 2 baseline image routing.",
                    original_prediction=b3_selected.get("predictions", {}),
                    enhanced_prediction=b3_selected.get("predictions", {}),
                    predictions_agree=True,
                    metrics={},
                ),
                comparison_enabled=False,
                comparison_mode="off",
            )
        elif images_identical:
            # Images identical: run inference once and share results (zero duplicate compute)
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="autovision-branch") as executor:
                fut1 = executor.submit(contextvars.copy_context().run, self._execute_branch_1_safe, image_rgb, conf_threshold, apply_preprocessing, include_classical_cv, fallback_to_full_image)
                fut2 = executor.submit(contextvars.copy_context().run, self._execute_branch_2_safe, image_rgb, enable_groq_inspection, exterior_conf_threshold)
                fut3 = executor.submit(contextvars.copy_context().run, self._execute_branch_3_safe, image_rgb)

                b1_res = fut1.result()
                b2_res = fut2.result()
                b3_res = fut3.result()

            b1_orig, b1_enh, b1_selected = b1_res, b1_res, b1_res
            b2_orig, b2_enh, b2_selected = b2_res, b2_res, b2_res
            b3_orig, b3_enh, b3_selected = b3_res, b3_res, b3_res

            comp_report = self.prediction_comparator.compare_all(
                b1_original=b1_orig,
                b1_enhanced=b1_enh,
                b2_original=b2_orig,
                b2_enhanced=b2_enh,
                b3_original=b3_orig,
                b3_enhanced=b3_enh,
                quality_score_orig=assessment.overall_score,
                quality_gain=0.0,
                enhancements_applied=[],
                comparison_mode="fast_identical",
            )
        else:
            # Full Dual Inference & Comparison (CV-only during comparison to eliminate duplicate Groq tokens)
            enhanced_image = enhancement.enhanced_image
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="autovision-comp") as executor:
                # Branch 1 Dual Inference (fast CV only, no Groq during comparison)
                fut_b1_orig = executor.submit(contextvars.copy_context().run, self._execute_branch_1_safe, image_rgb, conf_threshold, apply_preprocessing, include_classical_cv, fallback_to_full_image, False)
                fut_b1_enh = executor.submit(contextvars.copy_context().run, self._execute_branch_1_safe, enhanced_image, conf_threshold, apply_preprocessing, include_classical_cv, fallback_to_full_image, False)

                # Branch 2 Dual Inference (fast YOLO11n-seg, no Groq during comparison)
                fut_b2_orig = executor.submit(contextvars.copy_context().run, self._execute_branch_2_safe, image_rgb, False, exterior_conf_threshold)
                fut_b2_enh = executor.submit(contextvars.copy_context().run, self._execute_branch_2_safe, enhanced_image, False, exterior_conf_threshold)

                b1_orig = fut_b1_orig.result()
                b1_enh = fut_b1_enh.result()
                b2_orig = fut_b2_orig.result()
                b2_enh = fut_b2_enh.result()

            progress.end("detection")
            progress.begin("inspection")
            # Branch 3 executes once on the selected baseline image
            target_b3_img = enhanced_image if baseline_versions["selected_for_interior_analysis"] == "enhanced" else image_rgb
            b3_single = self._execute_branch_3_safe(target_b3_img)
            b3_orig = b3_single
            b3_enh = b3_single

            # Execute Part 3 Prediction Comparison
            comp_report = self.prediction_comparator.compare_all(
                b1_original=b1_orig,
                b1_enhanced=b1_enh,
                b2_original=b2_orig,
                b2_enhanced=b2_enh,
                b3_original=b3_orig,
                b3_enhanced=b3_enh,
                quality_score_orig=assessment.overall_score,
                quality_gain=enhancement.quality_gain,
                enhancements_applied=[r.to_dict() for r in enhancement.enhancements_applied],
                comparison_mode=comp_mode,
            )

            # Assign selected predictions based on intelligent comparison
            b1_selected = b1_enh if comp_report.vehicle_identification.selected_version == "enhanced" else b1_orig
            b2_selected = b2_enh if comp_report.exterior_analysis.selected_version == "enhanced" else b2_orig
            b3_selected = b3_enh if comp_report.interior_engine_analysis.selected_version == "enhanced" else b3_orig

            # Now run Groq Visual AI reasoning ONCE on the selected prediction ONLY if uncertain
            target_b1_img = enhanced_image if comp_report.vehicle_identification.selected_version == "enhanced" else image_rgb
            if (b1_selected.get("status") == "uncertain" or any(p.get("is_uncertain") for p in b1_selected.get("predictions", []))) and self.groq_service and self.groq_service.is_available:
                for pred in b1_selected.get("predictions", []):
                    if pred.get("is_uncertain"):
                        try:
                            crop = target_b1_img
                            if "box" in pred and len(pred["box"]) == 4:
                                x1, y1, x2, y2 = [int(c) for c in pred["box"]]
                                if x2 > x1 and y2 > y1:
                                    crop = target_b1_img[y1:y2, x1:x2]
                            ai_reasoning = self.groq_service.reason_vehicle_identity(
                                image=crop,
                                top1_prediction=pred,
                                top5_predictions=pred.get("top5", []),
                            )
                            pred["visual_ai_reasoning"] = ai_reasoning
                            b1_selected.setdefault("evidence", {})["visual_ai_reasoning"] = ai_reasoning
                        except Exception as e:
                            logger.warning(f"Single visual reasoning failed: {e}")

            # Now run Groq exterior inspection ONCE on the selected b2 image if enabled
            if enable_groq_inspection and self.groq_service and self.groq_service.is_available:
                target_b2_img = enhanced_image if comp_report.exterior_analysis.selected_version == "enhanced" else image_rgb
                try:
                    cv_classes = b2_selected.get("predictions", {}).get("detected_classes", [])
                    insp = self.groq_service.inspect_exterior(
                        image=target_b2_img,
                        cv_detected_classes=cv_classes,
                    )
                    b2_selected.setdefault("evidence", {})["groq_inspection_performed"] = True
                    b2_selected["evidence"]["inspection_notes"] = insp.get("inspection_notes")
                    b2_selected["evidence"]["missed_cues_noted"] = insp.get("missed_cues", [])
                    if insp.get("possible_modifications"):
                        b2_selected.setdefault("predictions", {})["possible_modifications"] = insp.get("possible_modifications")
                except Exception as e:
                    logger.warning(f"Single exterior inspection failed: {e}")

        progress.end("inspection")
        progress.begin("final")
        # 3. Final Groq Verdict Stage
        verdict_image = enhancement.enhanced_image if comp_report.interior_engine_analysis.selected_version == "enhanced" else image_rgb
        verdict = self._execute_verdict_safe(
            b1_selected, b2_selected, b3_selected, verdict_image, enable_final_verdict,
            prediction_comparison=comp_report.to_dict(),
        )

        # 4. Part 4: Vehicle Specifications & Current Market Price Intelligence
        primary_b1_pred = (b1_selected.get("predictions", []) or [{}])[0]
        final_make = verdict.get("final_make") or primary_b1_pred.get("make", "Unknown")
        provider_used = verdict.get("verdict_status") in ("verified", "provider_assessed")
        # Provider verdict: its model may legitimately be unknown (make only). Classifier fallback: its own model.
        final_model = verdict.get("final_model") if provider_used else primary_b1_pred.get("model")
        final_year = verdict.get("final_estimated_year")
        if final_year is None and not (provider_used and verdict.get("final_make")):
            final_year = primary_b1_pred.get("year", None)
        generation = verdict.get("final_generation")
        identity = self._vehicle_identity_block(
            final_make, final_model, final_year, generation, primary_b1_pred, b1_selected.get("confidence", 0.0), verdict
        )

        try:
            vehicle_specs, market_valuation = self.vehicle_intelligence_service.analyze_vehicle_intelligence(
                vehicle_identity={
                    "make": final_make,
                    "model": final_model or "",
                    "estimated_year": final_year,
                    "generation": generation,
                    "variant": None,
                },
                market_region=market_region,
                mileage=mileage,
                vehicle_condition=vehicle_condition,
                currency=currency,
            )
        except Exception as vi_err:
            logger.warning(f"Vehicle intelligence service failure: {vi_err}. Retaining pipeline continuity.")
            vehicle_specs = {
                "make": final_make,
                "model": final_model,
                "year": final_year,
                "generation": generation,
                "source": f"Vehicle intelligence failure: {str(vi_err)}",
                "verification_status": "unavailable",
            }
            market_valuation = {
                "currency": currency or "INR",
                "market_region": market_region or "India",
                "valuation_type": "unavailable",
                "source": "Unavailable due to service error",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "reliability": "unavailable",
                "assumptions": ["Market valuation unavailable due to transient service error."],
            }

        progress.end("final")
        CURRENT_FRAME.reset(_frame_token)
        REQUEST_ID.reset(_request_token)
        if not identity["valuation_eligible"]:
            # Never price a vehicle whose identity is uncertain, in conflict, or missing its model/year.
            market_valuation = {
                "currency": currency or "INR",
                "market_region": market_region or "India",
                "valuation_type": "unavailable",
                "source": "Not computed",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "reliability": "unavailable",
                "assumptions": ["Market value unavailable: " + identity["valuation_blocker"]],
            }

        elapsed_total_ms = (time.perf_counter() - start_time) * 1000.0

        # Update image versions record to reflect intelligent selection
        final_image_versions = {
            "original_preserved": True,
            "enhanced_available": bool(enhancement.enhancements_applied),
            "selected_for_vehicle_identification": comp_report.vehicle_identification.selected_version,
            "selected_for_exterior_analysis": comp_report.exterior_analysis.selected_version,
            "selected_for_interior_analysis": comp_report.interior_engine_analysis.selected_version,
        }

        return self._assemble_consolidated_report(
            b1=b1_selected,
            b2=b2_selected,
            b3=b3_selected,
            verdict=verdict,
            image_dimensions={"width": w, "height": h},
            total_time_ms=elapsed_total_ms,
            assessment=assessment,
            enhancement=enhancement,
            image_versions=final_image_versions,
            prediction_comparison=comp_report,
            original_predictions={"vehicle_identification": b1_orig, "exterior_analysis": b2_orig, "interior_engine": b3_orig},
            enhanced_predictions={"vehicle_identification": b1_enh, "exterior_analysis": b2_enh, "interior_engine": b3_enh},
            selected_predictions={"vehicle_identification": b1_selected, "exterior_analysis": b2_selected, "interior_engine": b3_selected},
            vehicle_specifications=vehicle_specs,
            market_valuation=market_valuation,
        )

    # -------------------------------------------------------------------------
    # Asynchronous Execution (FastAPI)
    # -------------------------------------------------------------------------
    async def analyze_async(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        apply_preprocessing: bool = True,
        include_classical_cv: bool = True,
        fallback_to_full_image: bool = True,
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
        Execute full image quality assessment, adaptive enhancement, Part 3 prediction comparison,
        and consolidated multi-model intelligence analysis asynchronously.
        """
        return await asyncio.to_thread(
            self.analyze_sync,
            image_rgb=image_rgb,
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

    # Alias analyze to analyze_async
    analyze = analyze_async

    # -------------------------------------------------------------------------
    # Consolidated Intelligence Synthesis
    # -------------------------------------------------------------------------
    @staticmethod
    def _vehicle_identity_block(final_make, final_model, final_year, generation, primary_b1, b1_conf, verdict):
        """Build the `vehicle` block and say who produced the identification.

        The classifier's confidence is only reported for the classifier's own answer. If a vision
        provider (Gemini/Groq) changed make/model, that identification has no calibrated
        confidence, so `confidence` is None and the result is flagged uncertain.
        """
        norm = lambda v: str(v or "").strip().lower()
        provider = verdict.get("verdict_provider")
        provider = {"primary": "groq", "backup": "groq"}.get(provider, provider)  # legacy key labels
        used = verdict.get("verdict_status") in ("verified", "provider_assessed") and bool(provider)
        clf_make, clf_model = primary_b1.get("make"), primary_b1.get("model")
        changed = used and (norm(final_make) != norm(clf_make) or norm(final_model) != norm(clf_model))
        if changed:
            source, confidence, uncertain = f"{provider}_vision", None, True
            note = f"Identified by the {provider.title()} vision model. It has no calibrated confidence score and is not verified."
        elif used:
            source, confidence, uncertain = f"classifier_confirmed_by_{provider}", round(b1_conf, 4), primary_b1.get("is_uncertain", False)
            note = f"Classifier result; the {provider.title()} vision model reached the same make and model."
        else:
            source, confidence, uncertain = "classifier_only", round(b1_conf, 4), primary_b1.get("is_uncertain", False)
            note = "Local classifier result only. No vision provider was available for this image."
        model_known = bool(final_model) and norm(final_model) not in ("unknown", "unidentified")
        conflict = bool(used and clf_make and (
            norm(final_make) != norm(clf_make) or (model_known and norm(final_model) != norm(clf_model))))
        if not model_known:
            blocker = "only the manufacturer could be identified, not the exact model."
        elif conflict:
            blocker = "the vision model and the classifier disagree about the vehicle."
        elif uncertain:
            blocker = "the identification is uncertain, so a precise valuation would be misleading."
        elif final_year is None:
            blocker = "the exact model year is not established."
        else:
            blocker = ""
        return {
            "make": final_make,
            "model": final_model if model_known else None,
            "model_known": model_known,
            "identity_conflict": conflict,
            "valuation_eligible": not blocker,
            "valuation_blocker": blocker,
            "estimated_year": final_year,
            "generation": generation,
            "confidence": confidence,
            "classifier_confidence": round(b1_conf, 4),
            "classifier_top1": f"{clf_make} {clf_model}".strip() if clf_make else None,
            "identification_source": source,
            "identification_note": note,
            "vision_provider": provider if used else None,
            "is_uncertain": uncertain,
        }

    @staticmethod
    def _assemble_consolidated_report(
        b1: Dict[str, Any],
        b2: Dict[str, Any],
        b3: Dict[str, Any],
        verdict: Dict[str, Any],
        image_dimensions: Dict[str, int],
        total_time_ms: float,
        assessment: Optional[QualityAssessmentResult] = None,
        enhancement: Optional[AdaptiveEnhancementResult] = None,
        image_versions: Optional[Dict[str, Any]] = None,
        prediction_comparison: Optional[ComprehensiveComparisonReport] = None,
        original_predictions: Optional[Dict[str, Any]] = None,
        enhanced_predictions: Optional[Dict[str, Any]] = None,
        selected_predictions: Optional[Dict[str, Any]] = None,
        vehicle_specifications: Optional[Dict[str, Any]] = None,
        market_valuation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize all perception branches, quality assessments, prediction comparisons, and verdict verification."""
        b1_preds = b1.get("predictions", [])
        primary_b1 = b1_preds[0] if b1_preds else {}

        final_make = verdict.get("final_make") or primary_b1.get("make", "Unknown")
        final_model = verdict.get("final_model") or primary_b1.get("model", "Unknown")
        final_year = verdict.get("final_estimated_year")
        if final_year is None:
            final_year = primary_b1.get("year", None)

        generation = verdict.get("final_generation")

        b1_conf = b1.get("confidence", 0.0)
        b2_conf = b2.get("confidence", 0.0)
        b3_conf = b3.get("confidence", 0.0)

        if b1_conf >= 0.70 and b1.get("status") == "success":
            overall_uncertainty = "Low (high CV classification confidence)"
        elif b1_conf >= 0.40:
            overall_uncertainty = "Moderate (standard CV classification confidence)"
        else:
            overall_uncertainty = "High (low CV classification confidence or fallback used)"

        missing_info = list(verdict.get("missing_or_unavailable_information", []))
        standard_unassessed = [
            "Vehicle specifications (horsepower, torque, dimensions) verified from catalog, not measured directly.",
            "Powertrain performance and dyno specs not evaluated from static imagery.",
            "Vehicle mileage and history records not evaluated from static imagery.",
            "Market valuation and pricing are indicative estimates based on secondary market indices.",
        ]
        if vehicle_specifications and vehicle_specifications.get("verification_status") == "unavailable":
            standard_unassessed.append("Vehicle technical specifications unavailable from verified automotive catalog.")
        if market_valuation and market_valuation.get("valuation_type") == "unavailable":
            standard_unassessed.append("Regional secondary market valuation index unavailable for this make/model.")
        for note in standard_unassessed:
            if note not in missing_info:
                missing_info.append(note)

        # Image Quality Payload (Part 2)
        if assessment is not None and enhancement is not None:
            image_quality_data = {
                "overall_score": assessment.overall_score,
                "detected_issues": assessment.detected_issues,
                "quality_metrics": assessment.quality_metrics,
                "enhancements_applied": [r.to_dict() for r in enhancement.enhancements_applied],
                "enhancement_success": enhancement.enhancement_success,
                "quality_gain": round(enhancement.quality_gain, 1),
                "recommendations": assessment.recommendations,
            }
        else:
            image_quality_data = {
                "overall_score": 100.0,
                "detected_issues": [],
                "quality_metrics": {},
                "enhancements_applied": [],
                "enhancement_success": True,
            }

        image_versions_data = image_versions or {
            "original_preserved": True,
            "enhanced_available": False,
            "selected_for_vehicle_identification": "original",
            "selected_for_exterior_analysis": "original",
            "selected_for_interior_analysis": "original",
        }

        # Prediction Comparison Payload (Part 3)
        comp_data = prediction_comparison.to_dict() if prediction_comparison else {}

        report = {
            "status": "success" if (b1.get("status") in ["success", "uncertain", "fallback"] or b2.get("status") == "success") else "partial",
            "image_dimensions": image_dimensions,
            "total_processing_time_ms": round(total_time_ms, 2),

            # Part 2: Image Quality Assessment & Adaptive Enhancement
            "image_quality": image_quality_data,
            "image_versions": image_versions_data,

            # Part 3: Prediction Comparison & Intelligent Image Selection
            "prediction_comparison": comp_data,
            "selected_predictions": selected_predictions or {},
            "original_predictions": original_predictions or {},
            "enhanced_predictions": enhanced_predictions or {},

            # 1. Core Consolidated Vehicle Identification (Selected)
            "vehicle": MultiModelOrchestrator._vehicle_identity_block(
                final_make, final_model, final_year, generation, primary_b1, b1_conf, verdict
            ),

            # Part 4: Vehicle Specifications & Current Market Price Intelligence
            "vehicle_specifications": vehicle_specifications or {},
            "market_valuation": market_valuation or {},

            # 2. Exterior Component Intelligence (Selected, Strictly 23 classes)
            "exterior_analysis": {
                "status": b2.get("status", "unknown"),
                "total_components_detected": len(b2.get("predictions", {}).get("components", [])),
                "component_counts": b2.get("predictions", {}).get("component_counts", {}),
                "detected_classes": b2.get("predictions", {}).get("detected_classes", []),
                "components": b2.get("predictions", {}).get("components", []),
                "possible_modifications": b2.get("predictions", {}).get("possible_modifications", []),
                "orientation_evidence": b2.get("evidence", {}).get("orientation_hints", []),
            },

            # 3. Interior Analysis (Selected)
            "interior_analysis": {
                "status": b3.get("status", "unavailable"),
                "is_interior_visible": b3.get("predictions", {}).get("is_interior_visible", False),
                "view_perspective": b3.get("predictions", {}).get("view_perspective", "unknown"),
                "interior_components": b3.get("predictions", {}).get("interior_components", []),
                "identity_cues": b3.get("predictions", {}).get("estimated_identity_cues", []),
                "notes": b3.get("evidence", {}).get("uncertainty_assessment", ""),
            },

            # 4. Visible Engine Information (Selected)
            "visible_engine_info": {
                "status": b3.get("status", "unavailable"),
                "is_engine_visible": b3.get("predictions", {}).get("is_engine_visible", False),
                "engine_components": b3.get("predictions", {}).get("engine_components", []),
                "notes": b3.get("evidence", {}).get("uncertainty_assessment", ""),
            },

            # 5. Model Confidence and Uncertainty
            "confidence_and_uncertainty": {
                "vehicle_identification_confidence": round(b1_conf, 4),
                "exterior_segmentation_confidence": round(b2_conf, 4),
                "interior_engine_confidence": round(b3_conf, 4),
                "overall_uncertainty_assessment": overall_uncertainty,
            },

            # 6. Cross-Model Disagreements & Final AI Verdict
            "cross_model_disagreements": verdict.get("cross_model_disagreements", []),
            "final_ai_verdict": {
                "verdict_status": verdict.get("verdict_status", "unverified"),
                "verdict_summary": verdict.get("verdict_summary", ""),
                "confirmed_facts": verdict.get("confirmed_facts", []),
                "visual_inferences": verdict.get("visual_inferences", []),
                "corrections_made": verdict.get("corrections_made", []),
                "provider": verdict.get("verdict_provider"),
                "provider_cached": verdict.get("provider_cached", False),
                "provider_attempts": verdict.get("provider_attempts", []),
            },

            # 7. Missing or Unavailable Information
            "missing_or_unavailable_information": missing_info,

            # 8. Transparent Original Model Outputs
            "original_model_outputs": {
                "branch_1_vehicle_id": {
                    "status": b1.get("status"),
                    "confidence": b1.get("confidence"),
                    "predictions": b1_preds,
                    "evidence": b1.get("evidence"),
                    "errors": b1.get("errors"),
                    "processing_time_ms": b1.get("processing_time_ms"),
                },
                "branch_2_exterior": {
                    "status": b2.get("status"),
                    "confidence": b2.get("confidence"),
                    "predictions": b2.get("predictions"),
                    "evidence": b2.get("evidence"),
                    "errors": b2.get("errors"),
                    "processing_time_ms": b2.get("processing_time_ms"),
                },
                "branch_3_interior_engine": {
                    "status": b3.get("status"),
                    "confidence": b3.get("confidence"),
                    "predictions": b3.get("predictions"),
                    "evidence": b3.get("evidence"),
                    "errors": b3.get("errors"),
                    "processing_time_ms": b3.get("processing_time_ms"),
                },
            },

            # Backwards Compatibility Fields for Existing Clients & Tests
            "vehicles_detected": len(b1_preds) if not primary_b1.get("is_fallback") else 0,
            "predictions_count": len(b1_preds),
            "processing_time_ms": round(total_time_ms, 2),
            "predictions": b1_preds,
        }

        return report
