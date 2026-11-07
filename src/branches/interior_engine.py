"""Branch 3: Interior and Engine Intelligence (Groq Vision Primary + Backup Failover)."""

import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

from ..groq_service import GroqService

logger = logging.getLogger("autovision.branch3.interior_engine")


class InteriorEngineBranch:
    """
    Branch 3: Interior and Engine Intelligence
    - Uses Groq vision as the primary analysis system.
    - Uses second Groq API configuration as automatic fallback.
    - Analyses dashboard, steering wheel, seats, infotainment, gear selector, visible engine-bay components.
    - Estimates vehicle identity using visual evidence.
    - Clearly communicates uncertainty (e.g. if only exterior is visible).
    - If both Groq APIs fail: continues with CV outputs and marks interior analysis unavailable.
    """

    def __init__(self, groq_service: Optional[GroqService] = None):
        self.groq_service = groq_service or GroqService()

    def analyze(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Execute Branch 3 analysis synchronously.

        Returns:
            Standardized structured dictionary:
            - status: "success" | "not_visible" | "uncertain" | "unavailable" | "error"
            - predictions: dict of view_perspective, is_interior_visible, is_engine_visible,
              interior_components, engine_components, estimated_identity_cues
            - confidence: confidence rating or float score
            - evidence: detailed visual observations and uncertainty assessment
            - errors: list of error strings
            - processing_time_ms: execution time
        """
        start_time = time.perf_counter()
        errors: List[str] = []

        if not self.groq_service or not self.groq_service.is_available:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "unavailable",
                "predictions": {
                    "view_perspective": "unknown",
                    "is_interior_visible": False,
                    "is_engine_visible": False,
                    "interior_components": [],
                    "engine_components": [],
                    "estimated_identity_cues": [],
                },
                "confidence": 0.0,
                "evidence": {
                    "uncertainty_assessment": "Groq Vision is not configured. Interior and engine intelligence unavailable.",
                    "provider": None,
                },
                "errors": ["Groq API keys not configured. Interior/engine intelligence unavailable."],
                "processing_time_ms": round(elapsed_ms, 2),
            }

        try:
            raw_res = self.groq_service.analyze_interior_and_engine(image_rgb)
        except Exception as e:
            logger.warning(f"Branch 3 execution failed: {e}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "unavailable",
                "predictions": {
                    "view_perspective": "unknown",
                    "is_interior_visible": False,
                    "is_engine_visible": False,
                    "interior_components": [],
                    "engine_components": [],
                    "estimated_identity_cues": [],
                },
                "confidence": 0.0,
                "evidence": {
                    "uncertainty_assessment": f"Failed to perform interior/engine inspection: {str(e)}",
                    "provider": None,
                },
                "errors": [str(e)],
                "processing_time_ms": round(elapsed_ms, 2),
            }

        if raw_res.get("status") == "unavailable":
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "unavailable",
                "predictions": {
                    "view_perspective": "unknown",
                    "is_interior_visible": False,
                    "is_engine_visible": False,
                    "interior_components": [],
                    "engine_components": [],
                    "estimated_identity_cues": [],
                },
                "confidence": 0.0,
                "evidence": {
                    "uncertainty_assessment": raw_res.get("uncertainty_assessment", "Groq API unavailable"),
                    "provider": None,
                },
                "errors": raw_res.get("errors", ["Groq vision unavailable"]),
                "processing_time_ms": round(elapsed_ms, 2),
            }

        is_int_vis = bool(raw_res.get("is_interior_visible", False))
        is_eng_vis = bool(raw_res.get("is_engine_visible", False))
        view_persp = raw_res.get("view_perspective", "unknown")

        if not is_int_vis and not is_eng_vis:
            status = "not_visible"
            confidence = 0.9  # Confident that it's NOT an interior/engine view
        else:
            status = "success"
            confidence = 0.8

        predictions = {
            "view_perspective": view_persp,
            "is_interior_visible": is_int_vis,
            "is_engine_visible": is_eng_vis,
            "interior_components": raw_res.get("interior_components", []),
            "engine_components": raw_res.get("engine_components", []),
            "estimated_identity_cues": raw_res.get("estimated_identity_cues", []),
            "scene_type": raw_res.get("scene_type"),
            "engine_details": raw_res.get("engine_details"),
        }

        evidence = {
            "uncertainty_assessment": raw_res.get("uncertainty_assessment", ""),
            "provider": raw_res.get("provider", "unknown"),
            "view_perspective": view_persp,
        }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": status,
            "predictions": predictions,
            "confidence": confidence,
            "evidence": evidence,
            "errors": errors,
            "processing_time_ms": round(elapsed_ms, 2),
        }
