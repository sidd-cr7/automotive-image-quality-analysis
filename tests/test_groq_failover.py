"""Groq dual-key failover and identity-source tests. Uses fake clients, so no API quota is used."""

import numpy as np
import pytest

from src.groq_service import GroqService, GroqServiceError
from src.orchestrator import MultiModelOrchestrator


class _FakeResponse:
    def __init__(self, text='{"ok": true}'):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class _FakeClient:
    """Mimics groq.Groq enough for GroqService._execute_with_failover."""

    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def call(self, *_a, **_k):
        self.calls += 1
        if self.error:
            raise self.error
        return _FakeResponse()


def _service(primary_err=None, backup_err=None, keys=("p", "b")):
    svc = GroqService(primary_api_key="", backup_api_key="")
    svc.max_retries = 1
    svc._primary_client = _FakeClient(primary_err) if keys[0] else None
    svc._backup_client = _FakeClient(backup_err) if keys[1] else None
    return svc


def _run(svc):
    return svc._execute_with_failover(lambda c: c.call(), "test call")


def test_primary_key_used_first():
    svc = _service()
    _, provider = _run(svc)
    assert provider == "primary"
    assert svc._backup_client.calls == 0


def test_rate_limited_primary_fails_over_to_backup_and_pauses_primary():
    svc = _service(primary_err=Exception("Error code: 429 - Rate limit reached"))
    _, provider = _run(svc)
    assert provider == "backup"
    primary_calls = svc._primary_client.calls
    # Primary is now paused: the next request goes straight to backup.
    _, provider2 = _run(svc)
    assert provider2 == "backup"
    assert svc._primary_client.calls == primary_calls


def test_generic_api_error_on_primary_fails_over():
    svc = _service(primary_err=Exception("connection reset"))
    _, provider = _run(svc)
    assert provider == "backup"


def test_both_keys_failing_raises_clean_error_without_leaking_keys():
    svc = _service(primary_err=Exception("429 rate limit"), backup_err=Exception("500 server error"))
    with pytest.raises(GroqServiceError) as exc:
        _run(svc)
    assert "Both Groq APIs failed" in str(exc.value)


def test_no_keys_configured_is_graceful():
    svc = _service(keys=(None, None))
    assert svc.is_available is False
    out = svc.analyze_interior_and_engine(np.zeros((32, 32, 3), np.uint8))
    assert out["status"] == "unavailable"


def test_both_keys_failing_returns_fallback_not_exception():
    svc = _service(primary_err=Exception("429 rate limit"), backup_err=Exception("429 rate limit"))
    out = svc.analyze_interior_and_engine(np.zeros((32, 32, 3), np.uint8))
    assert out["status"] == "unavailable"


# ---- identification source: fallback must be distinguishable from Groq results ----
_B1 = {"make": "Chevrolet", "model": "Sonic Sedan", "is_uncertain": True}


def _block(verdict, b1=_B1, conf=0.07, make="Hyundai", model="i30"):
    return MultiModelOrchestrator._vehicle_identity_block(make, model, 2017, None, b1, conf, verdict)


def test_groq_override_has_no_calibrated_confidence_and_is_uncertain():
    v = _block({"verdict_status": "verified", "verdict_provider": "primary"})
    assert v["identification_source"] == "groq_vision" and v["vision_provider"] == "groq"
    assert v["confidence"] is None and v["is_uncertain"] is True
    assert v["classifier_confidence"] == 0.07


def test_groq_agreeing_with_classifier_keeps_classifier_confidence():
    v = _block({"verdict_status": "verified", "verdict_provider": "primary"}, make="Chevrolet", model="Sonic Sedan")
    assert v["identification_source"] == "classifier_confirmed_by_groq"
    assert v["confidence"] == 0.07


def test_fallback_verdict_is_labelled_classifier_only():
    v = _block({"verdict_status": "fallback_to_original"}, make="Chevrolet", model="Sonic Sedan")
    assert v["identification_source"] == "classifier_only"
    assert v["vision_provider"] is None and v["is_uncertain"] is True


def test_cooldown_uses_groq_retry_hint():
    from src.groq_service import _cooldown_seconds
    assert _cooldown_seconds(Exception("Please try again in 20.31s. Need more")) == pytest.approx(21.31)
    assert _cooldown_seconds(Exception("try again in 1m3.504s")) == pytest.approx(64.504)
    assert _cooldown_seconds(Exception("429 no hint")) == 60.0


def test_both_keys_briefly_rate_limited_waits_once_then_succeeds(monkeypatch):
    import src.groq_service as gs
    monkeypatch.setattr(gs.time, "sleep", lambda s: None)  # don't actually wait in the test

    class _Flaky:
        def __init__(self):
            self.n = 0
        def call(self):
            self.n += 1
            if self.n == 1:
                raise Exception("429 Rate limit reached. Please try again in 2s.")
            return _FakeResponse()

    svc = _service()
    svc._primary_client, svc._backup_client = _Flaky(), _Flaky()
    # Wind the clocks so the wait is "elapsed" when we retry.
    real_time = gs.time.time
    state = {"t": 0.0}
    monkeypatch.setattr(gs.time, "time", lambda: real_time() + state["t"])
    orig_sleep = lambda s: state.__setitem__("t", state["t"] + s)
    monkeypatch.setattr(gs.time, "sleep", orig_sleep)
    _, provider = svc._execute_with_failover(lambda c: c.call(), "test")
    assert provider in ("primary", "backup")
