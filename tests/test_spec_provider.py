"""EPA spec lookup: real-data parsing, caching and failure handling. HTTP is faked (no network)."""
import src.spec_provider as sp
from src.vehicle_intelligence import VehicleIntelligenceService


class _R:
    def __init__(self, data, code=200):
        self._d, self.status_code = data, code
        self.text = "null" if data is None else "x"

    def json(self):
        return self._d


def _fake(calls):
    def get(url, params=None, **k):
        calls.append(url)
        if url.endswith("/menu/options"):
            return _R({"menuItem": [{"text": "Auto, 6 cyl, 3.6 L", "value": "1"}, {"text": "Auto, 8 cyl, 5.7 L", "value": "2"}]})
        d = {"1": ("3.6", "6"), "2": ("5.7", "8")}[url.rsplit("/", 1)[1]]
        return _R({"displ": d[0], "cylinders": d[1], "drive": "Rear-Wheel Drive", "trany": "Automatic 5-spd",
                   "fuelType1": "Regular Gasoline", "VClass": "Compact Cars"})
    return get


def test_multiple_configs_are_listed_and_only_agreed_fields_are_filled(monkeypatch):
    sp._cache.clear()
    monkeypatch.setattr(sp.httpx, "get", _fake([]))
    specs, _ = VehicleIntelligenceService().analyze_vehicle_intelligence({"make": "Dodge", "model": "Challenger", "estimated_year": 2011})
    assert specs["verification_status"] == "database_multiple_variants" and len(specs["variants"]) == 2
    assert specs["engine"]["displacement"] is None and specs["engine"]["cylinders"] is None  # configs differ: not guessed
    assert specs["drivetrain"] == "Rear-Wheel Drive" and specs["engine"]["fuel_type"] == "Regular Gasoline"  # all agree
    assert "EPA" in specs["source"]


def test_lookups_are_cached(monkeypatch):
    sp._cache.clear()
    calls = []
    monkeypatch.setattr(sp.httpx, "get", _fake(calls))
    sp.lookup_epa_variants("Dodge", "Challenger", 2011)
    n = len(calls)
    sp.lookup_epa_variants("dodge", "challenger", 2011)
    assert len(calls) == n


def test_not_in_database_or_api_failure_leaves_specs_unavailable(monkeypatch):
    sp._cache.clear()
    monkeypatch.setattr(sp.httpx, "get", lambda *a, **k: _R(None))
    assert sp.lookup_epa_variants("Hyundai", "i20", 2016) is None
    sp._cache.clear()

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(sp.httpx, "get", boom)
    assert sp.lookup_epa_variants("Dodge", "Challenger", 2011) is None
    specs, val = VehicleIntelligenceService().analyze_vehicle_intelligence({"make": "Dodge", "model": "Challenger", "estimated_year": 2011})
    assert specs["verification_status"] != "database_verified" and "variants" not in specs
    assert val["estimated_price"] is None


def test_no_price_without_a_verified_market_source():
    v = VehicleIntelligenceService().estimate_market_value("Honda", "Civic", 2015, market_region="India")
    assert v.estimated_price is None and "insufficient verified market data" in v.assumptions[0]
