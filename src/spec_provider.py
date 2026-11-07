"""Model-specific specifications from the US EPA fueleconomy.gov database (official, free, no API key).

Matches make + model + model year exactly. Returns every real configuration the database lists for that
model year; nothing is guessed or filled in. Only covers vehicles sold in the US. Results are cached.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("autovision.specs")
_BASE = "https://www.fueleconomy.gov/ws/rest/vehicle"
_HEADERS = {"Accept": "application/json"}
_cache: Dict[tuple, tuple] = {}
_TTL_OK, _TTL_MISS, _MAX_VARIANTS = 7 * 86400, 3600, 6


def _num(v: Any, cast):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def lookup_epa_variants(make: str, model: str, year: int, timeout: float = 5.0) -> Optional[List[Dict[str, Any]]]:
    """Return the database's configurations for this exact make/model/year, or None when unavailable."""
    try:
        key = (make.strip().lower(), model.strip().lower(), int(year))
    except (AttributeError, TypeError, ValueError):
        return None
    hit = _cache.get(key)
    if hit and time.time() < hit[0]:
        return hit[1]
    try:
        r = httpx.get(f"{_BASE}/menu/options", params={"year": key[2], "make": make.strip(), "model": model.strip()},
                      headers=_HEADERS, timeout=timeout)
        body = r.json() if r.status_code == 200 and r.text.strip() not in ("", "null") else None
        items = (body or {}).get("menuItem")
        items = items if isinstance(items, list) else ([items] if items else [])
        variants = []
        for it in items[:_MAX_VARIANTS]:
            v = httpx.get(f"{_BASE}/{it['value']}", headers=_HEADERS, timeout=timeout).json()
            variants.append({
                "option": it.get("text"), "displacement_l": _num(v.get("displ"), float), "cylinders": _num(v.get("cylinders"), int),
                "drive": v.get("drive") or None, "transmission": v.get("trany") or None,
                "fuel_type": v.get("fuelType1") or v.get("fuelType") or None, "vehicle_class": v.get("VClass") or None,
                "turbo": bool(v.get("turbo")), "supercharged": bool(v.get("sCharger")),
            })
        result = variants or None
        _cache[key] = (time.time() + (_TTL_OK if result else _TTL_MISS), result)
        return result
    except Exception as e:  # network/API failure: specs simply stay unavailable
        logger.warning("EPA spec lookup failed for %s %s %s: %s", make, model, year, type(e).__name__)
        _cache[key] = (time.time() + 60, None)
        return None
