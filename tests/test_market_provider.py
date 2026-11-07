"""Live market pricing: strict validation of grounded search results. HTTP is faked (no network, no credits)."""
import json

import src.market_provider as mp
from src.vehicle_intelligence import VehicleIntelligenceService


class _R:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body or {}
    def json(self):
        return self._b


def _grounded(listings, found=True, sources=("cars24.com", "carwale.com"), notes="import listings"):
    text = "Here you go: " + json.dumps({"found": found, "official_in_region": False, "listings": listings, "notes": notes})
    chunks = [{"web": {"uri": "https://x/" + s, "title": s}} for s in sources]
    return _R(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "groundingMetadata": {"groundingChunks": chunks}}]})


GOOD = [{"price": 5500000, "year": 2017, "source": "a"}, {"price": 6200000, "year": 2018, "source": "b"}, {"price": 5900000, "year": 2018, "source": "c"}]


def _run(monkeypatch, resp):
    mp._cache.clear()
    monkeypatch.setattr(mp, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(mp.httpx, "post", lambda *a, **k: resp)
    return mp.lookup_live_prices("Dodge", "Challenger", 2018, "India", "INR")


def test_grounded_listings_give_min_median_max_and_sources(monkeypatch):
    r = _run(monkeypatch, _grounded(GOOD))
    assert (r["min"], r["median"], r["max"]) == (5500000, 5900000, 6200000) and r["sources"] == ["cars24.com", "carwale.com"]


def test_rejects_unverifiable_results(monkeypatch):
    assert _run(monkeypatch, _grounded(GOOD, sources=())) is None                      # no search grounding at all
    assert _run(monkeypatch, _grounded(GOOD[:1])) is None                              # a single listing is not a market
    assert _run(monkeypatch, _grounded([{"price": 55}, {"price": 62}])) is None        # '55 lakh' reported as 55
    assert _run(monkeypatch, _grounded(GOOD, found=False)) is None                     # model says it found nothing
    assert _run(monkeypatch, _grounded([{"price": 100000}, {"price": 9000000}])) is None  # absurd spread


def test_quota_error_returns_none_and_retries_soon(monkeypatch):
    assert _run(monkeypatch, _R(429)) is None
    expires = next(iter(mp._cache.values()))[0]
    assert expires - __import__("time").time() <= 301


def test_valuation_uses_live_listings_only_when_verified(monkeypatch):
    svc = VehicleIntelligenceService()
    svc.live_pricing = True
    monkeypatch.setattr("src.market_provider.lookup_live_prices", lambda *a, **k: {
        "min": 5500000, "median": 5900000, "max": 6200000, "listings": GOOD, "sources": ["carwale.com"], "notes": "", "official_in_region": False})
    v = svc.estimate_market_value("Dodge", "Challenger", 2018, market_region="India")
    assert v.estimated_price == 5900000 and v.valuation_type == "live_listings" and v.reliability == "indicative"
    assert "carwale.com" in v.source and any("not independently verified" in a for a in v.assumptions)
    monkeypatch.setattr("src.market_provider.lookup_live_prices", lambda *a, **k: None)
    svc._val_cache.clear()
    v2 = svc.estimate_market_value("Dodge", "Challenger", 2018, market_region="India")
    assert v2.estimated_price is None and v2.valuation_type == "unavailable"
