"""Live market prices via Gemini with Google Search grounding (real-time web search, no extra API key).

Gemini is asked to find CURRENT asking prices in the target market and report only listings it actually saw.
A result is accepted only if: search grounding returned web sources, at least two listings were reported,
the prices are in the requested currency and plausible. Anything else yields None (valuation stays
"unavailable"). Prices are never converted from another currency and never interpolated.
Results are cached (default 6 h). Needs GEMINI_API_KEY; set LIVE_MARKET_PRICING=false to disable.
"""

import json
import logging
import re
import statistics
import time
from typing import Any, Dict, Optional

import httpx

from .config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger("autovision.market")
_cache: Dict[tuple, tuple] = {}
_TTL_OK, _TTL_MISS = 6 * 3600, 1800
_MIN_PLAUSIBLE = {"INR": 50_000, "USD": 500, "EUR": 500, "GBP": 500}  # catches "45" meaning 45 lakh, etc.


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def lookup_live_prices(make: str, model: str, year: int, region: str, currency: str,
                       timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """Return {min, median, max, listings, sources, notes} from live listings, or None when unverifiable."""
    if not (GEMINI_API_KEY and make and model and year):
        return None
    key = (make.strip().lower(), model.strip().lower(), int(year), region.lower(), currency)
    hit = _cache.get(key)
    if hit and time.time() < hit[0]:
        return hit[1]

    prompt = (
        f"Use Google Search to find CURRENT used-car asking prices in {region} for a {year} {make} {model}.\n"
        f"Rules: report only prices that appear in the search results you actually used, in {currency} as full numbers "
        f"(for example 4500000, not '45 lakh'). Do NOT convert from another currency, estimate or interpolate. "
        f"If this model is not officially sold in {region}, look for imported / used listings in {region} (dealers, import "
        f"specialists, classifieds) and say so in notes. Nearby model years (within 2 years) are acceptable; record each listing's year.\n"
        'Return ONLY JSON: {"found": true|false, "official_in_region": true|false|null, "listings": '
        '[{"price": number, "year": integer|null, "variant": string|null, "mileage_km": number|null, "source": "site name"}], '
        '"notes": "short"}. If you cannot find at least two real listings, set found=false and listings=[].'
    )
    body = {"contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048}}
    result = None
    short_cached = False
    try:
        r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                       json=body, headers={"x-goog-api-key": GEMINI_API_KEY}, timeout=timeout)
        if r.status_code != 200:
            logger.warning("Live price search failed: HTTP %s%s", r.status_code,
                           " (Google Search grounding needs a paid Gemini plan / billing)" if r.status_code == 429 else "")
            _cache[key] = (time.time() + 300, None)  # retry in 5 minutes, not 30
            short_cached = True
        else:
            cand = r.json()["candidates"][0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
            sources = sorted({(c.get("web") or {}).get("title") for c in chunks if (c.get("web") or {}).get("title")})
            data = _extract_json(text)
            listings = [x for x in (data or {}).get("listings", []) if isinstance(x.get("price"), (int, float)) and x["price"] > 0]
            prices = [float(x["price"]) for x in listings]
            if (data and data.get("found") and sources and len(prices) >= 2
                    and statistics.median(prices) >= _MIN_PLAUSIBLE.get(currency, 0) and max(prices) / min(prices) <= 8):
                result = {"min": min(prices), "median": statistics.median(prices), "max": max(prices),
                          "listings": listings[:8], "sources": sources[:6], "official_in_region": data.get("official_in_region"),
                          "notes": data.get("notes") or ""}
            else:
                logger.info("Live price search for %s %s %s: no verifiable listings (sources=%d, listings=%d)", make, model, year, len(sources), len(prices))
    except Exception as e:  # network / parsing: valuation stays unavailable
        logger.warning("Live price search error: %s", type(e).__name__)
    if not short_cached:
        _cache[key] = (time.time() + (_TTL_OK if result else _TTL_MISS), result)
    return result
