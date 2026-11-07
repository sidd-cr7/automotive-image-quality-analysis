"""Groq Vision & Intelligence Service with Dual-API Failover and Resilient Fallbacks."""

import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from .config import (
    GROQ_API_KEY,
    GROQ_API_KEY_BACKUP,
    GROQ_VISION_MODEL,
    GROQ_VERDICT_MODEL,
    GROQ_TIMEOUT,
    GROQ_MAX_RETRIES,
    GROQ_IMAGE_MAX_DIM,
    GROQ_IMAGE_JPEG_QUALITY,
)

logger = logging.getLogger("autovision.groq")

_SECRET_PATTERN = re.compile(r"(gsk_[A-Za-z0-9]+|org_[A-Za-z0-9]+|Bearer\s+[A-Za-z0-9._\-]+)")


def _redact(text: Any) -> str:
    """Strip API keys, org ids and bearer tokens from text before it is logged or returned."""
    return _SECRET_PATTERN.sub("[redacted]", str(text))


# Longest wait (seconds) we will absorb when both keys are briefly rate-limited.
MAX_RATE_LIMIT_WAIT = 15.0

_RETRY_HINT = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?", re.IGNORECASE)


def _cooldown_seconds(err: Any) -> float:
    """How long to pause a rate-limited key: Groq's own 'try again in ...' hint (+1 s), else 60 s.
    Short per-minute limits clear in seconds, so don't lock a key for a fixed 5 minutes."""
    m = _RETRY_HINT.search(str(err))
    if m and any(m.groups()):
        h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
        return min(3600.0, max(2.0, h * 3600 + mi * 60 + sec + 1.0))
    return 60.0


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _redact(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True


logger.addFilter(_RedactingFilter())


class GroqServiceError(Exception):
    """Raised when both primary and backup Groq services fail."""
    pass


class GroqService:
    """
    Robust Groq API integration supporting vision, reasoning, and final verdict verification
    with automatic primary-to-backup failover and safe fallbacks.
    """

    def __init__(
        self,
        primary_api_key: Optional[str] = None,
        backup_api_key: Optional[str] = None,
        vision_model: Optional[str] = None,
        verdict_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self.primary_api_key = primary_api_key if primary_api_key is not None else GROQ_API_KEY
        self.backup_api_key = backup_api_key if backup_api_key is not None else GROQ_API_KEY_BACKUP
        self.vision_model = vision_model or GROQ_VISION_MODEL
        self.verdict_model = verdict_model or GROQ_VERDICT_MODEL
        self.timeout = timeout if timeout is not None else GROQ_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else GROQ_MAX_RETRIES

        self._primary_client = None
        self._backup_client = None
        self._primary_exhausted_until = 0.0
        self._backup_exhausted_until = 0.0
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize Groq client instances if keys are present."""
        try:
            from groq import Groq
            if self.primary_api_key and self.primary_api_key.strip():
                self._primary_client = Groq(api_key=self.primary_api_key.strip(), timeout=self.timeout, max_retries=0)
            if self.backup_api_key and self.backup_api_key.strip():
                self._backup_client = Groq(api_key=self.backup_api_key.strip(), timeout=self.timeout, max_retries=0)
        except Exception as e:
            logger.warning(f"Failed to initialize Groq clients: {e}")

    @property
    def is_available(self) -> bool:
        """Check if at least one Groq API key is configured."""
        return bool(self._primary_client or self._backup_client)

    # Single-entry cache: one request sends the same frame to several Groq calls,
    # so encode it once instead of once per call. Keyed by object identity (the cache holds a
    # reference, so the id cannot be reused); hashing pixels would cost more than re-encoding.
    _b64_cache: Tuple[Any, int, int, str] = (None, 0, 0, "")

    @staticmethod
    def encode_image_base64(
        image: Union[np.ndarray, bytes],
        max_dim: Optional[int] = None,
        quality: Optional[int] = None,
    ) -> str:
        """Resize (aspect-preserving, never upscaling) and JPEG-compress an image for Groq.

        Accepts an RGB ndarray or encoded image bytes. Raises ValueError for undecodable
        input; the raw bytes are never forwarded. The caller's image is not modified.
        """
        max_dim = int(max_dim or GROQ_IMAGE_MAX_DIM)
        quality = int(quality or GROQ_IMAGE_JPEG_QUALITY)
        quality = min(95, max(40, quality))

        if isinstance(image, (bytes, bytearray)):
            raw = bytes(image) if not isinstance(image, bytes) else image
            original_bytes = len(raw)
        elif isinstance(image, np.ndarray):
            arr = image
            original_bytes = arr.nbytes
        else:
            raise TypeError(f"Unsupported image format for base64 encoding: {type(image)}")

        c_img, c_dim, c_q, c_val = GroqService._b64_cache
        if c_img is image and c_dim == max_dim and c_q == quality:
            return c_val

        t0 = time.perf_counter()
        if isinstance(image, (bytes, bytearray)):
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Unsupported or corrupt image data; cannot prepare it for Groq.")
        else:
            if arr.ndim == 3 and arr.shape[2] >= 3:
                bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
            elif arr.ndim == 2:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Unsupported image array shape: {arr.shape}")

        h, w = bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("Failed to encode image to JPEG buffer.")
        jpeg = buffer.tobytes()
        b64 = base64.b64encode(jpeg).decode("utf-8")

        logger.info(
            "Groq image optimised: %dx%d (%.1f KB %s) -> %dx%d JPEG q%d (%.1f KB, %.1f KB base64) in %.0f ms",
            w, h, original_bytes / 1024, "encoded" if isinstance(image, (bytes, bytearray)) else "raw pixels",
            bgr.shape[1], bgr.shape[0], quality, len(jpeg) / 1024, len(b64) / 1024,
            (time.perf_counter() - t0) * 1000,
        )
        GroqService._b64_cache = (image, max_dim, quality, b64)
        return b64

    def _execute_with_failover(
        self,
        call_fn,
        description: str = "Groq call",
        _retried: bool = False,
    ) -> Tuple[Any, str]:
        """
        Execute an API call with automatic failover:
        Primary -> Backup -> Failure
        Returns (result, provider_name).
        """
        last_error = None

        # 1. Try Primary if available and not quota-exhausted
        now = time.time()
        if self._primary_client and now >= self._primary_exhausted_until:
            for attempt in range(1, self.max_retries + 1):
                try:
                    res = call_fn(self._primary_client)
                    return res, "primary"
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str or "tokens per day" in err_str
                    if is_rate_limit:
                        logger.warning(
                            f"Primary Groq hit quota/rate limit during {description}; pausing primary key: {e}"
                        )
                        self._primary_exhausted_until = time.time() + _cooldown_seconds(e)
                        break  # Immediately fail over to backup without waiting
                    logger.warning(
                        f"Primary Groq {description} failed (attempt {attempt}/{self.max_retries}): {e}"
                    )
                    time.sleep(0.2 * attempt)

        # 2. Try Backup if Primary failed or was unavailable/exhausted
        if self._backup_client and now >= self._backup_exhausted_until:
            logger.info(f"Failing over to Backup Groq for {description}...")
            for attempt in range(1, self.max_retries + 1):
                try:
                    res = call_fn(self._backup_client)
                    return res, "backup"
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str or "tokens per day" in err_str
                    if is_rate_limit:
                        logger.warning(
                            f"Backup Groq hit quota/rate limit during {description}; pausing backup key: {e}"
                        )
                        self._backup_exhausted_until = time.time() + _cooldown_seconds(e)
                        break
                    logger.warning(
                        f"Backup Groq {description} failed (attempt {attempt}/{self.max_retries}): {e}"
                    )
                    time.sleep(0.2 * attempt)

        # 3. Both keys rate-limited for only a few seconds (per-minute token limits): wait once and retry,
        #    which is better than silently dropping the vision analysis.
        if not _retried and self._primary_client and self._backup_client:
            now2 = time.time()
            waits = [t - now2 for t in (self._primary_exhausted_until, self._backup_exhausted_until) if t > now2]
            if len(waits) == 2 and min(waits) <= MAX_RATE_LIMIT_WAIT:
                wait = min(waits) + 0.5
                logger.info(f"Both Groq keys rate-limited for {description}; waiting {wait:.1f}s to retry once.")
                time.sleep(wait)
                return self._execute_with_failover(call_fn, description, _retried=True)

        # 4. Both failed or neither available
        err_msg = f"Both Groq APIs failed for {description}." if (self._primary_client or self._backup_client) else "No Groq API keys configured."
        if last_error:
            err_msg += f" Last error: {_redact(last_error)}"
        logger.error(err_msg)
        raise GroqServiceError(err_msg)

    # -------------------------------------------------------------------------
    # Branch 1 Fallback: Visual Reasoning when Vehicle ID is Uncertain
    # -------------------------------------------------------------------------
    def reason_vehicle_identity(
        self,
        image: Union[np.ndarray, bytes],
        top1_prediction: Dict[str, Any],
        top5_predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Request visual AI reasoning when classification confidence is low or detector missed.
        Does NOT overwrite raw predictions; provides supporting visual evidence.
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "reasoning": "Groq API keys not configured. Visual reasoning unavailable.",
                "visual_cues": [],
                "alternative_hypothesis": None,
                "confidence_assessment": "unassessed",
            }

        try:
            b64_img = self.encode_image_base64(image)
            top5_summary = ", ".join(
                [f"{p.get('make')} {p.get('model')} ({p.get('year')}) [{p.get('confidence', 0):.2f}]" for p in top5_predictions[:5]]
            )

            prompt = f"""You are an expert automotive forensic vision system.
A ResNet-50 computer vision classifier predicted this vehicle with low confidence or uncertainty.
Top-1: {top1_prediction.get('make')} {top1_prediction.get('model')} ({top1_prediction.get('year')}) with confidence {top1_prediction.get('confidence', 0):.2f}.
Top-5 alternatives: {top5_summary}.

Examine the vehicle image carefully.
Respond strictly in JSON format with these exact keys:
{{
  "visual_cues": ["list of notable visible exterior cues like grille shape, headlight design, badge, body style"],
  "agreement_with_top1": true or false,
  "confidence_assessment": "low" or "medium" or "high",
  "reasoning": "Concise forensic explanation of visual evidence supporting or questioning the prediction",
  "suggested_make": "Make name or null",
  "suggested_model": "Model name or null",
  "suggested_year_range": "Year range or null"
}}"""

            def _call(client):
                return client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=600,
                )

            res, provider = self._execute_with_failover(_call, "vehicle visual reasoning")
            content = res.choices[0].message.content
            parsed = json.loads(content)
            parsed["status"] = "success"
            parsed["provider"] = provider
            return parsed

        except Exception as e:
            logger.warning(f"Visual reasoning fallback failed: {e}")
            return {
                "status": "fallback_failed",
                "error": str(e),
                "reasoning": "Visual AI reasoning could not be completed.",
                "visual_cues": [],
                "confidence_assessment": "uncertain",
            }

    # -------------------------------------------------------------------------
    # Branch 2: Exterior Inspection for Aftermarket Modifications & Missed Cues
    # -------------------------------------------------------------------------
    def inspect_exterior(
        self,
        image: Union[np.ndarray, bytes],
        cv_detected_classes: List[str],
    ) -> Dict[str, Any]:
        """
        Inspect exterior image for visually supported aftermarket modifications and missed cues.
        Strictly avoids inventing unsupported classes for the CV model.
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "possible_modifications": [],
                "missed_cues": [],
                "inspection_notes": "Groq vision unavailable for exterior inspection.",
            }

        try:
            b64_img = self.encode_image_base64(image)
            cv_classes_str = ", ".join(cv_detected_classes) if cv_detected_classes else "None detected by CV"

            prompt = f"""You are an expert vehicle exterior and styling evaluator.
Our YOLO11n-seg model detected the following standard components: [{cv_classes_str}].

Inspect the image for:
1. Potential AFTERMARKET MODIFICATIONS (e.g., custom aftermarket wheels/rims, rear spoiler/wing, tinted windows, widebody/lip kit, custom exhaust, roof racks, non-factory paint/wrap, decals). Only report modifications if clearly visible! Do not guess.
2. Any major exterior items that may have been missed by the standard component detector.

Respond strictly in JSON format with these exact keys:
{{
  "possible_modifications": [
    {{
      "modification_type": "string (e.g. Tinted Windows, Aftermarket Rims, Rear Spoiler)",
      "confidence": "high" or "medium" or "low",
      "visual_evidence": "description of visual evidence"
    }}
  ],
  "missed_cues": ["string descriptions of any obvious exterior items not in standard components"],
  "inspection_notes": "Brief objective assessment of the exterior condition and modifications"
}}"""

            def _call(client):
                return client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=600,
                )

            res, provider = self._execute_with_failover(_call, "exterior inspection")
            content = res.choices[0].message.content
            parsed = json.loads(content)
            parsed["status"] = "success"
            parsed["provider"] = provider
            return parsed

        except Exception as e:
            logger.warning(f"Exterior Groq inspection failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "possible_modifications": [],
                "missed_cues": [],
                "inspection_notes": "Exterior visual inspection unavailable.",
            }

    # -------------------------------------------------------------------------
    # Branch 3: Interior & Engine Intelligence
    # -------------------------------------------------------------------------
    def analyze_interior_and_engine(
        self,
        image: Union[np.ndarray, bytes],
    ) -> Dict[str, Any]:
        """
        Analyze interior and engine-bay elements using Groq Vision.
        If exterior shot, clearly states interior/engine not visible and communicates uncertainty.
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "view_perspective": "unknown",
                "is_interior_visible": False,
                "is_engine_visible": False,
                "interior_components": [],
                "engine_components": [],
                "estimated_identity_cues": [],
                "uncertainty_assessment": "Groq vision service is not configured or unavailable.",
                "errors": ["Groq API keys not configured"],
            }

        try:
            b64_img = self.encode_image_base64(image)

            prompt = """You are an automotive interior and powertrain inspection AI.
Examine this image carefully:
1. Determine the view perspective: "exterior_full", "exterior_crop", "interior_cabin", "engine_bay", or "mixed".
2. Check if the interior cabin is visible:
   - Identify: dashboard, steering wheel (and badges), seats, infotainment screen, gear selector / shifter, instrument cluster, pedals, door cards.
3. Check if the engine bay or powertrain components are visible:
   - Identify: engine block/cover, air intake, radiator/coolant, battery, alternator, turbo/supercharger, strut towers.
4. Note any visual clues toward vehicle make/model/generation/trim from interior or engine details.
5. Clearly express uncertainty if interior or engine components are obscured, distant, or absent.

Respond strictly in JSON format with these exact keys:
{
  "view_perspective": "exterior_full | exterior_crop | interior_cabin | engine_bay | mixed",
  "is_interior_visible": true or false,
  "is_engine_visible": true or false,
  "interior_components": [
    {
      "component": "e.g. steering_wheel, infotainment, seats, dashboard, gear_selector",
      "visible": true or false,
      "details": "visual description or 'not visible'"
    }
  ],
  "engine_components": [
    {
      "component": "e.g. engine_cover, air_intake, battery",
      "visible": true or false,
      "details": "visual description or 'not visible'"
    }
  ],
  "estimated_identity_cues": ["visible interior/engine badges, styling language, or trim indicators"],
  "uncertainty_assessment": "Clear objective statement regarding visibility and certainty level"
}"""

            def _call(client):
                return client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=800,
                )

            res, provider = self._execute_with_failover(_call, "interior and engine intelligence")
            content = res.choices[0].message.content
            parsed = json.loads(content)
            parsed["status"] = "success"
            parsed["provider"] = provider
            return parsed

        except Exception as e:
            logger.warning(f"Interior and engine intelligence failed: {e}")
            return {
                "status": "unavailable",
                "error": str(e),
                "view_perspective": "unknown",
                "is_interior_visible": False,
                "is_engine_visible": False,
                "interior_components": [],
                "engine_components": [],
                "estimated_identity_cues": [],
                "uncertainty_assessment": f"Interior/engine analysis failed: {str(e)}",
                "errors": [str(e)],
            }

    # -------------------------------------------------------------------------
    # Section 4: Final Groq Verdict Engine
    # -------------------------------------------------------------------------
    def generate_final_verdict(
        self,
        branch_1_result: Dict[str, Any],
        branch_2_result: Dict[str, Any],
        branch_3_result: Dict[str, Any],
        image: Optional[Union[np.ndarray, bytes]] = None,
        prediction_comparison: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate results from all three branches and perform multi-modal verification:
        1. Compare vehicle ID with interior and exterior evidence.
        2. Identify contradictory predictions.
        3. Review confidence and uncertainty.
        4. Correct predictions ONLY when sufficient supporting evidence exists.
        5. Fill missing info (generation, trim, body style).
        6. Identify potentially missed exterior modifications.
        7. Distinguish confirmed facts from visual inferences.
        8. Factor in Original vs Enhanced prediction comparison results (Part 3).
        9. Preserve original model outputs for transparency.
        Must NOT blindly overwrite CV predictions.
        If final Groq fails, returns fallback preserving original outputs.
        """
        if not self.is_available:
            return self._build_verdict_fallback(
                branch_1_result, branch_2_result, branch_3_result,
                reason="Groq API keys not configured. Returning original model outputs without AI modification."
            )

        try:
            summary_payload = {
                "branch_1_vehicle_id": {
                    "status": branch_1_result.get("status"),
                    "confidence": branch_1_result.get("confidence"),
                    "predictions": branch_1_result.get("predictions", [])[:2],
                    "evidence": branch_1_result.get("evidence", {}),
                },
                "branch_2_exterior": {
                    "status": branch_2_result.get("status"),
                    "confidence": branch_2_result.get("confidence"),
                    "detected_classes": branch_2_result.get("predictions", {}).get("detected_classes", []),
                    "component_counts": branch_2_result.get("predictions", {}).get("component_counts", {}),
                    "possible_modifications": branch_2_result.get("predictions", {}).get("possible_modifications", []),
                },
                "branch_3_interior_engine": {
                    "status": branch_3_result.get("status"),
                    "is_interior_visible": branch_3_result.get("predictions", {}).get("is_interior_visible", False),
                    "is_engine_visible": branch_3_result.get("predictions", {}).get("is_engine_visible", False),
                    "estimated_identity_cues": branch_3_result.get("predictions", {}).get("estimated_identity_cues", []),
                    "uncertainty_assessment": branch_3_result.get("evidence", {}).get("uncertainty_assessment", ""),
                },
            }

            if prediction_comparison:
                summary_payload["original_vs_enhanced_comparison"] = {
                    "vehicle_id_agreed": prediction_comparison.get("vehicle_identification", {}).get("predictions_agree"),
                    "vehicle_id_selected": prediction_comparison.get("vehicle_identification", {}).get("selected_version"),
                    "vehicle_id_reason": prediction_comparison.get("vehicle_identification", {}).get("selection_reason"),
                    "exterior_selected": prediction_comparison.get("exterior_analysis", {}).get("selected_version"),
                    "exterior_reason": prediction_comparison.get("exterior_analysis", {}).get("selection_reason"),
                    "interior_selected": prediction_comparison.get("interior_engine_analysis", {}).get("selected_version"),
                }

            prompt = f"""You are the Chief Automotive Architect and Final Verdict Verification Engine of AUTOVISION AI.
Analyze the structured evidence from all three parallel perception branches:

{json.dumps(summary_payload, indent=2)}

STRICT VERIFICATION RULES:
1. Compare vehicle identification with exterior components and interior/engine evidence.
2. Identify any contradictions between branches (e.g. classifier predicted sedan but exterior detected tailgate or truck bed).
3. Review confidence and uncertainty.
4. DO NOT blindly overwrite computer vision predictions. ONLY correct predictions if there is overwhelming, undeniable visual contradiction.
5. Fill missing information when possible (e.g. vehicle generation, body style, sub-model/trim).
6. Note any potential aftermarket modifications.
7. Explicitly distinguish CONFIRMED FACTS (empirically detected components and high-confidence CV predictions) from VISUAL INFERENCES (hypotheses, estimated years, generation clues).
8. Clearly list any corrections made, or note "none".

Respond strictly in JSON format with these exact keys:
{{
  "final_make": "string (confirmed or validated make)",
  "final_model": "string (confirmed or validated model)",
  "final_estimated_year": integer or null,
  "final_generation": "string (e.g. '13th Gen', 'Mk7', 'S550') or null",
  "corrections_made": [
    {{
      "field": "e.g. model, year, body_style",
      "original_value": "original value",
      "corrected_value": "corrected value",
      "justification": "sufficient supporting evidence justification"
    }}
  ],
  "cross_model_disagreements": [
    "description of any conflict between classifier, exterior, or interior/engine evidence"
  ],
  "confirmed_facts": [
    "list of confirmed factual detections from CV and visual confirmation"
  ],
  "visual_inferences": [
    "list of plausible visual inferences, trim estimates, or generational nuances"
  ],
  "missing_or_unavailable_information": [
    "list of information that could not be verified from the image"
  ],
  "verdict_summary": "Concise authoritative final vehicle intelligence report"
}}"""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            # If image is available and using vision model for verdict, attach image
            if image is not None and "vision" in self.vision_model:
                try:
                    b64_img = self.encode_image_base64(image)
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    })
                    model_to_use = self.vision_model
                except Exception:
                    model_to_use = self.verdict_model
            else:
                model_to_use = self.verdict_model

            def _call(client):
                return client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=900,
                )

            res, provider = self._execute_with_failover(_call, "final verdict verification")
            content = res.choices[0].message.content
            parsed = json.loads(content)
            parsed["verdict_status"] = "verified"
            parsed["verdict_provider"] = provider
            return parsed

        except Exception as e:
            logger.warning(f"Final Groq verdict engine failed: {e}. Preserving original model outputs.")
            return self._build_verdict_fallback(
                branch_1_result, branch_2_result, branch_3_result,
                reason=f"Final verdict API failure: {str(e)}. Original aggregated model outputs preserved."
            )

    @staticmethod
    def _build_verdict_fallback(
        b1: Dict[str, Any],
        b2: Dict[str, Any],
        b3: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """Safe fallback verdict when final Groq fails or is unavailable."""
        b1_preds = b1.get("predictions", [])
        primary_pred = b1_preds[0] if b1_preds else {}

        return {
            "verdict_status": "fallback_to_original",
            "final_make": primary_pred.get("make", "Unknown"),
            "final_model": primary_pred.get("model", "Unknown"),
            "final_estimated_year": primary_pred.get("year", None),
            "final_generation": None,
            "corrections_made": [],
            "cross_model_disagreements": [],
            "confirmed_facts": [
                f"CV Primary Prediction: {primary_pred.get('make', 'Unknown')} {primary_pred.get('model', '')} ({primary_pred.get('year', '')})",
                f"Exterior components detected: {len(b2.get('predictions', {}).get('components', []))}",
            ],
            "visual_inferences": [],
            "missing_or_unavailable_information": [
                "Final AI verification unavailable; results reflect raw CV model predictions.",
                "Engine performance, specs, and market valuation not evaluated.",
            ],
            "verdict_summary": f"Original computer vision outputs retained. {reason}",
        }
