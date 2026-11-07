"""Provider-chain tests: Gemini -> Groq -> local fallback, caching, cooldowns. Provider calls are faked (no quota used)."""

import time

import numpy as np
import pytest

import src.vision_service as vs
from src.vision_service import ProviderError, VisionService

RESULT = {
    "vehicle": {"make": "Hyundai", "model": "i20", "year_estimate": 2016, "generation": None, "identification_certainty": "high",
                "visible_evidence": ["Hyundai badge"], "alternatives": []},
    "exterior": {"notes": "clean", "visible_modifications": [], "missed_cues": []},
    "interior_engine": {"view_perspective": "exterior_full", "is_interior_visible": False, "is_engine_visible": False,
                        "interior_components": [], "engine_components": [], "identity_cues": [], "uncertainty": "interior not visible"},
    "summary": "A white Hyundai i20.",
}
IMG = np.full((120, 160, 3), 90, np.uint8)


def make(gemini=None, groq=None, gemini_key="k", groq_configured=True):
    """VisionService with fake provider calls. Each fake is a callable or an exception to raise."""
    svc = VisionService(primary_api_key="", backup_api_key="")
    svc.gemini_key = gemini_key
    svc._primary_client = object() if groq_configured else None
    svc.calls = {"gemini": 0, "groq": 0}

    def _g(_b64):
        svc.calls["gemini"] += 1
        if isinstance(gemini, Exception):
            raise gemini
        return dict(RESULT), "gemini"

    def _q(_b64):
        svc.calls["groq"] += 1
        if isinstance(groq, Exception):
            raise groq
        return dict(RESULT), "groq"

    svc._call_gemini, svc._call_groq = _g, _q
    return svc


def test_gemini_is_primary_and_groq_untouched():
    svc = make()
    a = svc.analyze_frame(IMG)
    assert a["ok"] and a["provider"] == "gemini" and svc.calls == {"gemini": 1, "groq": 0}


@pytest.mark.parametrize("kind,retry", [("rate_limit", 30), ("timeout", 30), ("auth", 600), ("server", 20)])
def test_gemini_failures_fail_over_to_groq(kind, retry):
    svc = make(gemini=ProviderError(kind, f"{kind} failure", retry))
    a = svc.analyze_frame(IMG)
    assert a["ok"] and a["provider"] == "groq"
    assert [x["provider"] for x in a["attempts"]] == ["gemini", "groq"]
    assert kind in a["attempts"][0]["reason"]


def test_failed_provider_goes_on_cooldown_and_is_skipped_for_next_image():
    svc = make(gemini=ProviderError("rate_limit", "quota", 60))
    svc.analyze_frame(IMG)
    other = svc.analyze_frame(IMG + 1)  # different image -> not cached
    assert svc.calls["gemini"] == 1, "gemini must not be retried while cooling down"
    assert "cooling down" in other["attempts"][0]["reason"]
    assert other["provider"] == "groq"


def test_each_provider_attempted_at_most_once_per_image_and_no_infinite_retry():
    svc = make(gemini=ProviderError("server", "boom", 20), groq=ProviderError("error", "boom", 20))
    a = svc.analyze_frame(IMG)
    assert not a["ok"] and svc.calls == {"gemini": 1, "groq": 1}
    svc.analyze_frame(IMG)  # failure is briefly cached; providers are not hit again
    assert svc.calls == {"gemini": 1, "groq": 1}


def test_all_providers_down_degrades_to_classifier_only_verdict():
    svc = make(gemini=ProviderError("auth", "bad key", 600), groq=ProviderError("error", "down", 20))
    v = svc.generate_final_verdict({"predictions": [{"make": "Chevrolet", "model": "Sonic Sedan", "label": "x", "year": 2012}]}, {"predictions": {}}, {}, IMG)
    assert v["verdict_status"] == "fallback_to_original"
    assert v.get("verdict_provider") is None
    assert [x["provider"] for x in v["provider_attempts"]] == ["gemini", "groq"]


def test_same_image_hits_cache_and_compat_methods_make_no_extra_calls():
    svc = make()
    svc.analyze_frame(IMG)
    assert svc.analyze_frame(IMG)["cached"] is True
    svc.reason_vehicle_identity(IMG, {"make": "x", "model": "y"}, [])
    svc.inspect_exterior(IMG)
    svc.analyze_interior_and_engine(IMG)
    v = svc.generate_final_verdict({"predictions": [{"make": "Chevrolet", "model": "Sonic Sedan", "label": "x", "year": 2012}]}, {"predictions": {}}, {}, IMG)
    assert svc.calls == {"gemini": 1, "groq": 0}
    assert v["verdict_provider"] == "gemini" and v["final_make"] == "Hyundai"
    assert v["corrections_made"], "a changed identification must be recorded as a correction"


def test_one_frame_context_means_one_call_even_when_branches_pass_different_pixels():
    svc = make()
    tok = vs.CURRENT_FRAME.set(IMG)
    try:
        svc.analyze_interior_and_engine(IMG[:60])      # e.g. a crop
        svc.inspect_exterior(IMG + 5)                  # e.g. an enhanced copy
        svc.reason_vehicle_identity(IMG[:, :80], {"make": "x", "model": "y"}, [])
    finally:
        vs.CURRENT_FRAME.reset(tok)
    assert svc.calls["gemini"] == 1


def test_cache_expiry_forces_a_fresh_call(monkeypatch):
    svc = make()
    svc.analyze_frame(IMG)
    for k, (_exp, val) in list(svc._cache.items()):
        svc._cache[k] = (time.time() - 1, val)
    svc.analyze_frame(IMG)
    assert svc.calls["gemini"] == 2


def test_provider_order_is_configurable_and_unconfigured_providers_are_skipped():
    svc = make(gemini_key="")
    a = svc.analyze_frame(IMG)
    assert a["provider"] == "groq" and a["attempts"][0]["reason"] == "not configured"
    svc2 = make(groq_configured=False, gemini_key="")
    assert svc2.is_available is False
    assert svc2.analyze_frame(IMG)["ok"] is False


def test_status_reports_usage_without_secrets():
    svc = make(gemini=ProviderError("rate_limit", "quota", 60))
    svc.analyze_frame(IMG)
    st = svc.status()
    assert st["providers"]["gemini"]["failed"] == 1 and st["providers"]["gemini"]["cooldown_s"] > 0
    assert st["providers"]["groq"]["ok"] == 1
    assert "k" != str(st) and "gemini_key" not in str(st)


def test_secrets_are_scrubbed_from_provider_errors():
    from src.groq_service import _redact
    assert "gsk_abc123" not in _redact("bad key gsk_abc123 org_zzz")


def test_exterior_modifications_are_normalised_to_objects_for_the_api_schema():
    svc = make()
    svc._call_gemini = lambda b64: ({**RESULT, "exterior": {"notes": "n", "visible_modifications": ["tire smoke", {"modification": "spoiler"}], "missed_cues": []}}, "gemini")
    mods = svc.inspect_exterior(IMG)["possible_modifications"]
    assert all(isinstance(m, dict) and m["confidence"] == "unassessed" for m in mods)
    assert mods[0]["modification_type"] == "tire smoke" and mods[1]["modification_type"] == "spoiler"


def test_cached_flag_means_served_from_an_earlier_request():
    svc = make()
    tok = vs.REQUEST_ID.set("req-1")
    try:
        first = svc.analyze_frame(IMG)
        same_request = svc.analyze_frame(IMG)   # e.g. the verdict step reusing the combined result
    finally:
        vs.REQUEST_ID.reset(tok)
    tok = vs.REQUEST_ID.set("req-2")
    try:
        later = svc.analyze_frame(IMG)
    finally:
        vs.REQUEST_ID.reset(tok)
    assert first["cached"] is False and same_request["cached"] is False and later["cached"] is True
    assert svc.calls["gemini"] == 1
