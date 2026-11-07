"""Multi-provider vision service: Gemini (primary) -> Groq (secondary) -> local classifier fallback.

* ONE combined vision request per image covers identification, exterior notes and interior/engine
  analysis. The old per-purpose methods are kept as thin views over that single cached result, so
  the branches and orchestrator need no extra API calls.
* Providers are tried in VISION_PROVIDER_ORDER. Each gets at most ONE attempt per image (Groq
  additionally fails over between its two keys). Quota / rate-limit / timeout / API errors put a
  provider on a cooldown so later requests skip it instead of hammering it.
* Results are cached by a hash of the optimised JPEG, with single-flight locking so concurrent
  branches share one call. Failures are cached briefly for the same reason.
* Provider output is always labelled with its provider; no numeric confidence is invented.
* Keys come from the environment only and are scrubbed from every log line and error.
"""

import base64
import contextvars
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np

from .config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TIMEOUT,
    VISION_CACHE_SIZE,
    VISION_CACHE_TTL,
    VISION_PROVIDER_ORDER,
)
from .groq_service import GroqService, GroqServiceError, _redact, _cooldown_seconds

logger = logging.getLogger("autovision.vision")

PROMPT_VERSION = "v1"
FAILURE_CACHE_SECONDS = 20.0

# The frame the orchestrator is currently analysing (the original, un-enhanced upload).
CURRENT_FRAME: contextvars.ContextVar[Optional[np.ndarray]] = contextvars.ContextVar("vision_frame", default=None)
# Identifies the request, so "cached" means "served from an EARLIER request", not "reused within this one".
REQUEST_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("vision_request", default=None)

COMBINED_PROMPT = """You are an expert automotive vision analyst. Examine this single image and answer everything below in ONE JSON object.
Be factual. If something is not visible or you are unsure, say so. Never invent badges, trims or specifications.

Respond strictly as JSON with exactly these keys:
{
  "vehicle": {
    "make": "string or null",
    "model": "string or null",
    "year_estimate": integer or null,
    "generation": "string or null",
    "body_style": "string or null",
    "identification_certainty": "low | medium | high",
    "visible_evidence": ["badges, grille, lamps, plate text, body shape actually visible"],
    "alternatives": ["other plausible 'Make Model' candidates, if unsure"]
  },
  "exterior": {
    "notes": "brief objective assessment of the exterior",
    "visible_modifications": ["aftermarket modifications actually visible"],
    "missed_cues": ["notable exterior items worth mentioning"]
  },
  "interior_engine": {
    "view_perspective": "exterior_full | exterior_crop | interior_cabin | engine_bay | mixed",
    "is_interior_visible": true or false,
    "is_engine_visible": true or false,
    "interior_components": [{"component": "name", "visible": true or false, "details": "description or 'not visible'"}],
    "engine_components": [{"component": "name", "visible": true or false, "details": "description or 'not visible'"}],
    "identity_cues": ["interior or engine details that hint at make/model/trim"],
    "uncertainty": "what could not be verified from this image"
  },
  "summary": "two sentences: what the vehicle appears to be and how sure you are"
}"""


class ProviderError(Exception):
    """A provider call failed. `retry_after` is how long to skip that provider."""

    def __init__(self, kind: str, message: str, retry_after: float):
        super().__init__(message)
        self.kind = kind  # rate_limit | timeout | auth | server | bad_response | error
        self.retry_after = retry_after


def _parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    data = json.loads(text)
    if not isinstance(data, dict) or "vehicle" not in data:
        raise ValueError("response is not the expected JSON object")
    return data


def _as_modifications(items: Any, provider: str) -> List[Dict[str, Any]]:
    """Providers may return plain strings or objects; the API schema wants objects. Confidence is never invented."""
    out: List[Dict[str, Any]] = []
    for it in items or []:
        if isinstance(it, dict):
            d = dict(it)
            d.setdefault("modification_type", str(d.get("modification") or d.get("type") or d.get("name") or "unspecified"))
            d.setdefault("confidence", "unassessed")
            d.setdefault("source", provider)
            out.append(d)
        elif it:
            out.append({"modification_type": str(it), "confidence": "unassessed", "source": provider})
    return out


class VisionService(GroqService):
    """Drop-in superset of GroqService; same method names, backed by the provider chain."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gemini_key = (GEMINI_API_KEY or "").strip()
        self.gemini_model = GEMINI_MODEL
        self.gemini_timeout = GEMINI_TIMEOUT
        self.order = [p.strip().lower() for p in VISION_PROVIDER_ORDER.split(",") if p.strip()]
        self._cooldown: Dict[str, float] = {}
        self._usage: Dict[str, Dict[str, Any]] = {}
        self._cache: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------ availability / status
    def _configured(self, name: str) -> bool:
        if name == "gemini":
            return bool(self.gemini_key)
        if name == "groq":
            return bool(self._primary_client or self._backup_client)
        return False

    @property
    def is_available(self) -> bool:
        return any(self._configured(p) for p in self.order)

    def status(self) -> Dict[str, Any]:
        """Provider state for /health. Never contains keys."""
        now = time.time()
        return {
            "order": self.order,
            "providers": {
                p: {
                    "configured": self._configured(p),
                    "cooldown_s": max(0, round(self._cooldown.get(p, 0) - now)),
                    **self._usage.get(p, {"ok": 0, "failed": 0, "last_error": None}),
                }
                for p in ("gemini", "groq")
            },
            "cache_entries": len(self._cache),
        }

    def _record(self, name: str, ok: bool, error: Optional[str] = None) -> None:
        u = self._usage.setdefault(name, {"ok": 0, "failed": 0, "last_error": None})
        u["ok" if ok else "failed"] += 1
        if error:
            u["last_error"] = error[:200]

    # ------------------------------------------------------------------ providers (one attempt each)
    def _call_gemini(self, b64_jpeg: str) -> Tuple[Dict[str, Any], str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        gen_cfg: Dict[str, Any] = {"temperature": 0.1, "responseMimeType": "application/json", "maxOutputTokens": 4096}
        if "2.5-flash" in self.gemini_model:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}  # faster, and thinking tokens would eat the output budget
        body = {
            "contents": [{"parts": [{"text": COMBINED_PROMPT}, {"inline_data": {"mime_type": "image/jpeg", "data": b64_jpeg}}]}],
            "generationConfig": gen_cfg,
        }
        try:
            resp = httpx.post(url, json=body, headers={"x-goog-api-key": self.gemini_key}, timeout=self.gemini_timeout)
        except httpx.TimeoutException as e:
            raise ProviderError("timeout", f"Gemini timed out after {self.gemini_timeout:.0f}s", 30.0) from e
        except httpx.HTTPError as e:
            raise ProviderError("error", f"Gemini network error: {_redact(e)}", 20.0) from e

        if resp.status_code != 200:
            try:
                msg = _redact(resp.json()["error"]["message"])[:160]
            except Exception:
                msg = " ".join(_redact(resp.text[:160]).split())
            if resp.status_code == 429 or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r'retry(?:Delay)?"?\s*:?\s*"?(\d+(?:\.\d+)?)s', resp.text)
                wait = float(m.group(1)) + 1 if m else _cooldown_seconds(msg)
                raise ProviderError("rate_limit", f"Gemini quota/rate limit (HTTP 429)", min(3600.0, max(5.0, wait)))
            if resp.status_code in (400, 401, 403):
                raise ProviderError("auth", f"Gemini rejected the request (HTTP {resp.status_code}): {msg}", 600.0)
            raise ProviderError("server", f"Gemini HTTP {resp.status_code}: {msg}", 20.0)
        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_json(text), "gemini"
        except Exception as e:
            raise ProviderError("bad_response", f"Gemini returned an unusable response: {_redact(e)}", 15.0) from e

    def _call_groq(self, b64_jpeg: str) -> Tuple[Dict[str, Any], str]:
        def _call(client):
            return client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": COMBINED_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"}},
                ]}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1400,
            )
        try:
            res, _key_label = self._execute_with_failover(_call, "combined vision analysis")
        except GroqServiceError as e:
            raise ProviderError("rate_limit" if "429" in str(e) else "error", _redact(e), _cooldown_seconds(e)) from e
        try:
            return _parse_json(res.choices[0].message.content), "groq"
        except Exception as e:
            raise ProviderError("bad_response", f"Groq returned an unusable response: {_redact(e)}", 15.0) from e

    # ------------------------------------------------------------------ the combined, cached request
    def _frame_for(self, image: Union[np.ndarray, bytes, None]) -> Union[np.ndarray, bytes]:
        frame = CURRENT_FRAME.get()
        return frame if frame is not None else image

    def analyze_frame(self, image: Union[np.ndarray, bytes, None]) -> Dict[str, Any]:
        """Return {ok, provider, result, cached, attempts, error}. At most one API attempt per provider."""
        frame = self._frame_for(image)
        if frame is None:
            return {"ok": False, "provider": None, "result": None, "cached": False, "attempts": [], "error": "no image"}
        b64 = self.encode_image_base64(frame)  # resized to <=1024px JPEG; original untouched
        key = hashlib.sha256(f"{PROMPT_VERSION}:{self.gemini_model}:{b64}".encode()).hexdigest()

        with self._cache_lock:
            lock = self._key_locks.setdefault(key, threading.Lock())
        with lock:
            hit = self._cache_get(key)
            rid = REQUEST_ID.get()
            if hit is not None:
                return {**hit, "cached": rid is None or hit.get("_rid") != rid}

            attempts: List[Dict[str, Any]] = []
            for name in self.order:
                if not self._configured(name):
                    attempts.append({"provider": name, "ok": False, "reason": "not configured"})
                    continue
                remaining = self._cooldown.get(name, 0) - time.time()
                if remaining > 0:
                    attempts.append({"provider": name, "ok": False, "reason": f"cooling down ({remaining:.0f}s left)"})
                    continue
                t0 = time.perf_counter()
                try:
                    result, provider = self._call_gemini(b64) if name == "gemini" else self._call_groq(b64)
                except ProviderError as e:
                    self._cooldown[name] = time.time() + e.retry_after
                    self._record(name, False, f"{e.kind}: {e}")
                    attempts.append({"provider": name, "ok": False, "reason": f"{e.kind}: {e}"})
                    logger.warning("Vision provider %s failed (%s): %s; skipping it for %.0fs", name, e.kind, e, e.retry_after)
                    continue
                except Exception as e:  # defensive: never let a provider bug break the pipeline
                    self._cooldown[name] = time.time() + 20.0
                    self._record(name, False, f"error: {_redact(e)}")
                    attempts.append({"provider": name, "ok": False, "reason": f"error: {_redact(e)}"})
                    logger.warning("Vision provider %s raised: %s", name, _redact(e))
                    continue
                ms = (time.perf_counter() - t0) * 1000
                self._record(name, True)
                attempts.append({"provider": provider, "ok": True, "reason": None, "latency_ms": round(ms)})
                logger.info("Vision analysis by %s in %.0f ms (attempts: %s)", provider, ms,
                            ", ".join(f"{a['provider']}={'ok' if a['ok'] else 'failed'}" for a in attempts))
                out = {"ok": True, "provider": provider, "result": result, "attempts": attempts, "error": None, "_rid": REQUEST_ID.get()}
                self._cache_put(key, out, VISION_CACHE_TTL)
                return {**out, "cached": False}

            err = "; ".join(f"{a['provider']}: {a['reason']}" for a in attempts) or "no provider configured"
            out = {"ok": False, "provider": None, "result": None, "attempts": attempts, "error": err}
            self._cache_put(key, out, FAILURE_CACHE_SECONDS)
            logger.error("All vision providers unavailable: %s", err)
            return {**out, "cached": False}

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._cache_lock:
            item = self._cache.get(key)
            if not item:
                return None
            expires, value = item
            if time.time() > expires:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return value

    def _cache_put(self, key: str, value: Dict[str, Any], ttl: float) -> None:
        with self._cache_lock:
            self._cache[key] = (time.time() + ttl, value)
            self._cache.move_to_end(key)
            while len(self._cache) > VISION_CACHE_SIZE:
                self._cache.popitem(last=False)

    # ------------------------------------------------------------------ GroqService-compatible views (no extra API calls)
    def reason_vehicle_identity(self, image, top1_prediction, top5_predictions) -> Dict[str, Any]:
        a = self.analyze_frame(image)
        if not a["ok"]:
            return {"status": "unavailable", "provider": None, "reasoning": f"Vision providers unavailable: {a['error']}",
                    "visual_cues": [], "alternative_hypothesis": None, "confidence_assessment": "unassessed"}
        v = a["result"].get("vehicle") or {}
        norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x or "").lower())
        agrees = bool(v.get("make")) and norm(v.get("make")) == norm(top1_prediction.get("make")) \
            and norm(v.get("model")) in norm(top1_prediction.get("model")) + norm(top1_prediction.get("label"))
        alt = None if agrees else (" ".join(x for x in (v.get("make"), v.get("model")) if x) or None)
        return {"status": "success", "provider": a["provider"], "visual_cues": v.get("visible_evidence", []),
                "agreement_with_top1": agrees, "confidence_assessment": v.get("identification_certainty", "unassessed"),
                "reasoning": a["result"].get("summary", ""), "alternative_hypothesis": alt}

    def inspect_exterior(self, image, cv_detected_classes=None) -> Dict[str, Any]:
        a = self.analyze_frame(image)
        if not a["ok"]:
            return {"status": "unavailable", "provider": None, "possible_modifications": [], "missed_cues": [],
                    "inspection_notes": "Vision providers unavailable for exterior inspection."}
        ex = a["result"].get("exterior") or {}
        return {"status": "success", "provider": a["provider"], "possible_modifications": _as_modifications(ex.get("visible_modifications"), a["provider"]),
                "missed_cues": ex.get("missed_cues", []), "inspection_notes": ex.get("notes", "")}

    def analyze_interior_and_engine(self, image) -> Dict[str, Any]:
        a = self.analyze_frame(image)
        if not a["ok"]:
            return {"status": "unavailable", "uncertainty_assessment": "Vision providers unavailable.",
                    "errors": [f"Vision providers unavailable: {a['error']}"]}
        ie = a["result"].get("interior_engine") or {}
        return {"status": "success", "provider": a["provider"],
                "view_perspective": ie.get("view_perspective", "unknown"),
                "is_interior_visible": bool(ie.get("is_interior_visible", False)),
                "is_engine_visible": bool(ie.get("is_engine_visible", False)),
                "interior_components": ie.get("interior_components", []),
                "engine_components": ie.get("engine_components", []),
                "estimated_identity_cues": ie.get("identity_cues", []),
                "uncertainty_assessment": ie.get("uncertainty", "")}

    def generate_final_verdict(self, branch_1_result, branch_2_result, branch_3_result, image=None,
                               prediction_comparison=None) -> Dict[str, Any]:
        """Built from the single cached vision result plus the CV outputs; makes no extra API call."""
        a = self.analyze_frame(image)
        if not a["ok"]:
            fb = self._build_verdict_fallback(
                branch_1_result, branch_2_result, branch_3_result,
                reason=f"No vision provider answered ({a['error']}). Classifier results only.")
            fb["provider_attempts"] = a["attempts"]
            return fb

        res = a["result"]
        v = res.get("vehicle") or {}
        preds = branch_1_result.get("predictions", [])
        top = preds[0] if preds else {}
        make = v.get("make") or top.get("make", "Unknown")
        model = v.get("model") or top.get("model", "Unknown")
        year = v.get("year_estimate") if v.get("year_estimate") is not None else top.get("year")
        norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x or "").lower())
        changed = bool(v.get("make")) and (norm(make) != norm(top.get("make")) or norm(model) not in (norm(top.get("model")) + norm(top.get("label"))))
        corrections = []
        if changed and top:
            corrections.append({"field": "make/model", "original_value": f"{top.get('make')} {top.get('model')}",
                                "corrected_value": f"{make} {model}",
                                "justification": "; ".join(v.get("visible_evidence", [])[:4]) or "vision model assessment"})
        comps = branch_2_result.get("predictions", {}).get("components", []) if isinstance(branch_2_result.get("predictions"), dict) else []
        return {
            "verdict_status": "provider_assessed",
            "verdict_provider": a["provider"],
            "provider_attempts": a["attempts"],
            "provider_cached": a["cached"],
            "final_make": make, "final_model": model, "final_estimated_year": year,
            "final_generation": v.get("generation"),
            "corrections_made": corrections,
            "cross_model_disagreements": ([f"Classifier: {top.get('make')} {top.get('model')}; {a['provider']}: {make} {model}"] if changed else []),
            "confirmed_facts": [f"Exterior components detected by segmentation model: {len(comps)}"]
                               + ([f"Classifier top-1: {top.get('label')}"] if top else []),
            "visual_inferences": [f"[{a['provider']}] {x}" for x in v.get("visible_evidence", [])],
            "missing_or_unavailable_information": [res.get("interior_engine", {}).get("uncertainty", "")] if res.get("interior_engine", {}).get("uncertainty") else [],
            "verdict_summary": res.get("summary", ""),
        }
