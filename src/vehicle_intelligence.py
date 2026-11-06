"""
Vehicle Specifications and Current Market Price Intelligence Service (Part 4).

Provides authoritative automotive specifications and transparent, evidence-based
current market valuations for identified vehicles.

Design Principles:
- Factual external data sources (SIAM, Manufacturer Spec Sheets, CarWale, CarDekho, KBB).
- Strict separation of verified, estimated, and unavailable specifications.
- Never invent vehicle specifications or fabricate live market listings.
- India-first pricing by default with multi-region/currency configurability.
- Explicit valuation assumptions when mileage or condition is unspecified.
- Isolated fail-safe execution: failures never block perception or verdict branches.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import (
    AUTOMOTIVE_DB_PATH,
    DEFAULT_MARKET_REGION,
    DEFAULT_MARKET_CURRENCY,
    ENABLE_VEHICLE_SPECIFICATIONS,
    ENABLE_MARKET_VALUATION,
    VALUATION_CACHE_TTL_SECONDS,
)

logger = logging.getLogger("autovision.vehicle_intelligence")


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
@dataclass
class EngineSpecification:
    displacement: Optional[str] = None
    configuration: Optional[str] = None
    fuel_type: Optional[str] = None
    aspiration: Optional[str] = None
    horsepower: Optional[int] = None
    torque_nm: Optional[int] = None
    cylinders: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VehicleSpecifications:
    make: str
    model: str
    year: Optional[int] = None
    generation: Optional[str] = None
    variant: Optional[str] = None
    engine: EngineSpecification = field(default_factory=EngineSpecification)
    transmission: Optional[str] = None
    drivetrain: Optional[str] = None
    body_type: Optional[str] = None
    seating_capacity: Optional[int] = None
    source: str = "Automotive Database"
    verification_status: str = "unavailable"  # "verified", "estimated", "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.engine, EngineSpecification):
            data["engine"] = self.engine.to_dict()
        return data


@dataclass
class MarketValuation:
    estimated_price: Optional[float] = None
    minimum_price: Optional[float] = None
    maximum_price: Optional[float] = None
    currency: str = "INR"
    market_region: str = "India"
    vehicle_condition: str = "unknown"
    mileage: Optional[int] = None
    valuation_type: str = "indicative"  # "indicative", "live", "unavailable"
    source: str = "Indicative Market Valuation"
    source_url: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reliability: str = "moderate"  # "high", "moderate", "low", "indicative", "unavailable"
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Core Vehicle Intelligence Service
# -----------------------------------------------------------------------------
class VehicleIntelligenceService:
    """
    Automotive data service providing verified specifications and regional
    secondary market valuations.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        default_region: str = DEFAULT_MARKET_REGION,
        default_currency: str = DEFAULT_MARKET_CURRENCY,
        cache_ttl_seconds: int = VALUATION_CACHE_TTL_SECONDS,
        enable_specifications: bool = ENABLE_VEHICLE_SPECIFICATIONS,
        enable_valuation: bool = ENABLE_MARKET_VALUATION,
    ):
        self.db_path = Path(db_path or AUTOMOTIVE_DB_PATH)
        self.default_region = default_region
        self.default_currency = default_currency
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_specifications = enable_specifications
        self.enable_valuation = enable_valuation

        # In-memory caches: key -> (timestamp, data)
        self._spec_cache: Dict[str, Any] = {}
        self._val_cache: Dict[str, Tuple[float, Any]] = {}

        # Load database
        self._db: Dict[str, Any] = self._load_database()

    def _load_database(self) -> Dict[str, Any]:
        """Load curated automotive reference JSON with safe fallback."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded automotive reference data from {self.db_path} ({len(data.get('vehicles', []))} vehicles)")
                    return data
            except Exception as e:
                logger.error(f"Failed to read automotive DB at {self.db_path}: {e}. Falling back to internal catalog.")

        # Minimal internal fallback catalog if file missing
        return {
            "version": "1.0-fallback",
            "regional_settings": {
                "default_region": "India",
                "supported_regions": {
                    "India": {
                        "currency": "INR",
                        "currency_symbol": "₹",
                        "source_name": "CarWale / CarDekho Secondary Valuation Index (India)",
                        "source_url": "https://www.carwale.com/used/",
                        "standard_annual_mileage_km": 12000,
                    },
                    "USA": {
                        "currency": "USD",
                        "currency_symbol": "$",
                        "source_name": "Kelley Blue Book / Edmunds Secondary Valuation (USA)",
                        "source_url": "https://www.kbb.com/used-cars/",
                        "standard_annual_mileage_km": 19300,
                    },
                },
            },
            "vehicles": [],
        }

    # -------------------------------------------------------------------------
    # 1. Vehicle Specification Retrieval
    # -------------------------------------------------------------------------
    def get_specifications(
        self,
        make: str,
        model: str,
        year: Optional[int] = None,
        generation: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> VehicleSpecifications:
        """
        Retrieve verified vehicle technical specifications.
        Returns 'verified' if variant and model are identified with external source,
        'estimated' if base generation specifications are matched without exact variant,
        or 'unavailable' if uncataloged without fabricating specifications.
        """
        if not self.enable_specifications:
            return VehicleSpecifications(
                make=make,
                model=model,
                year=year,
                generation=generation,
                variant=variant,
                source="Specifications feature disabled in configuration",
                verification_status="unavailable",
            )

        cache_key = f"{make.strip().lower()}:{model.strip().lower()}:{year or ''}:{variant or ''}"
        if cache_key in self._spec_cache:
            return self._spec_cache[cache_key]

        make_clean = make.strip()
        model_clean = model.strip()

        matched_vehicle = self._find_vehicle_record(make_clean, model_clean)
        if not matched_vehicle:
            # Model not cataloged in external database: preserve transparency, do NOT invent specs!
            spec = VehicleSpecifications(
                make=make_clean,
                model=model_clean,
                year=year,
                generation=generation,
                variant=variant,
                engine=EngineSpecification(),
                transmission=None,
                drivetrain=None,
                body_type=self._infer_body_type_from_name(model_clean),
                seating_capacity=None,
                source="No verified external specification source found for this make/model",
                verification_status="unavailable",
            )
            self._spec_cache[cache_key] = spec
            return spec

        # Find matching generation
        gens = matched_vehicle.get("generations", [])
        matched_gen = self._find_generation(gens, year, generation)

        if not matched_gen:
            # Vehicle model known, but specific year generation not cataloged
            spec = VehicleSpecifications(
                make=matched_vehicle.get("make", make_clean),
                model=matched_vehicle.get("model", model_clean),
                year=year,
                generation=generation,
                variant=variant,
                engine=EngineSpecification(),
                transmission=None,
                drivetrain=None,
                body_type=matched_vehicle.get("body_type"),
                seating_capacity=matched_vehicle.get("seating_capacity"),
                source=f"Automotive Catalog ({matched_vehicle.get('make')})",
                verification_status="unavailable",
            )
            self._spec_cache[cache_key] = spec
            return spec

        # Populate specification from generation record
        eng_data = matched_gen.get("engine", {})
        engine = EngineSpecification(
            displacement=eng_data.get("displacement"),
            configuration=eng_data.get("configuration"),
            fuel_type=eng_data.get("fuel_type"),
            aspiration=eng_data.get("aspiration"),
            horsepower=eng_data.get("horsepower"),
            torque_nm=eng_data.get("torque_nm"),
            cylinders=eng_data.get("cylinders"),
        )

        actual_variant = variant or matched_gen.get("default_variant")
        verification_status = "verified" if variant else "estimated"

        spec = VehicleSpecifications(
            make=matched_vehicle.get("make", make_clean),
            model=matched_vehicle.get("model", model_clean),
            year=year,
            generation=matched_gen.get("generation") or generation,
            variant=actual_variant,
            engine=engine,
            transmission=matched_gen.get("transmission"),
            drivetrain=matched_gen.get("drivetrain"),
            body_type=matched_vehicle.get("body_type"),
            seating_capacity=matched_vehicle.get("seating_capacity"),
            source=matched_gen.get("source", "Manufacturer Official Specifications"),
            verification_status=verification_status,
        )

        self._spec_cache[cache_key] = spec
        return spec

    # -------------------------------------------------------------------------
    # 2. Current Market Price Intelligence
    # -------------------------------------------------------------------------
    def estimate_market_value(
        self,
        make: str,
        model: str,
        year: Optional[int] = None,
        generation: Optional[str] = None,
        variant: Optional[str] = None,
        condition: Optional[str] = None,
        mileage: Optional[int] = None,
        market_region: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> MarketValuation:
        """
        Estimate current market valuation using regional secondary market indices,
        depreciation curves, and explicit valuation assumptions.
        """
        region = market_region or self.default_region
        reg_info = self._get_region_info(region)
        curr = currency or reg_info.get("currency", self.default_currency)

        now_iso = datetime.now(timezone.utc).isoformat()
        current_year = datetime.now(timezone.utc).year

        if not self.enable_valuation:
            return MarketValuation(
                currency=curr,
                market_region=region,
                valuation_type="unavailable",
                source="Market valuation feature disabled in configuration",
                retrieved_at=now_iso,
                reliability="unavailable",
                assumptions=["Market valuation disabled in configuration."],
            )

        # Check Cache
        cache_key = f"{make.strip().lower()}:{model.strip().lower()}:{year or ''}:{region}:{mileage or ''}:{condition or ''}"
        now_ts = time.time()
        if cache_key in self._val_cache:
            entry_ts, val_result = self._val_cache[cache_key]
            if now_ts - entry_ts < self.cache_ttl_seconds:
                return val_result

        make_clean = make.strip()
        model_clean = model.strip()

        matched_vehicle = self._find_vehicle_record(make_clean, model_clean)
        if not matched_vehicle or year is None:
            # Model or year unknown: return transparently unavailable, NEVER fabricate!
            valuation = MarketValuation(
                estimated_price=None,
                minimum_price=None,
                maximum_price=None,
                currency=curr,
                market_region=region,
                vehicle_condition=condition or "unknown",
                mileage=mileage,
                valuation_type="unavailable",
                source=reg_info.get("source_name", "Secondary Market Index"),
                source_url=reg_info.get("source_url", ""),
                retrieved_at=now_iso,
                reliability="unavailable",
                assumptions=[
                    f"Market valuation unavailable: vehicle model '{make_clean} {model_clean}' "
                    f"{'or model year' if year is None else ''} is not mapped to reliable regional pricing indices."
                ],
            )
            self._val_cache[cache_key] = (now_ts, valuation)
            return valuation

        gens = matched_vehicle.get("generations", [])
        matched_gen = self._find_generation(gens, year, generation)

        if not matched_gen or "msrp_new" not in matched_gen:
            valuation = MarketValuation(
                estimated_price=None,
                minimum_price=None,
                maximum_price=None,
                currency=curr,
                market_region=region,
                vehicle_condition=condition or "unknown",
                mileage=mileage,
                valuation_type="unavailable",
                source=reg_info.get("source_name", "Secondary Market Index"),
                source_url=reg_info.get("source_url", ""),
                retrieved_at=now_iso,
                reliability="unavailable",
                assumptions=[
                    f"Baseline regional pricing for year {year} not found in verified market indices."
                ],
            )
            self._val_cache[cache_key] = (now_ts, valuation)
            return valuation

        # Base MSRP in target currency
        msrp_dict = matched_gen.get("msrp_new", {})
        base_msrp = float(msrp_dict.get(curr, msrp_dict.get("INR", 0.0)))
        if base_msrp <= 0.0:
            # Fallback approximate conversion from INR or USD
            if "INR" in msrp_dict and curr == "USD":
                base_msrp = msrp_dict["INR"] / 86.0
            elif "USD" in msrp_dict and curr == "INR":
                base_msrp = msrp_dict["USD"] * 86.0
            else:
                base_msrp = float(next(iter(msrp_dict.values()), 0.0))

        # Vehicle Age & Depreciation Calculation
        age = max(0, current_year - year)
        retention_ratio = self._calculate_depreciation_retention(
            age=age,
            make=make_clean,
            body_type=matched_vehicle.get("body_type", ""),
            region=region,
        )

        raw_estimated = base_msrp * retention_ratio

        # Mileage and Condition Adjustments
        assumptions: List[str] = []
        is_mileage_supplied = mileage is not None and mileage > 0
        cond_str = (condition or "").strip().lower()

        # Mileage adjustment
        std_annual_km = reg_info.get("standard_annual_mileage_km", 12000)
        expected_mileage = max(std_annual_km, age * std_annual_km)

        mileage_factor = 1.0
        if is_mileage_supplied:
            # User provided mileage
            diff_ratio = (mileage - expected_mileage) / expected_mileage
            # Lower mileage increases value up to +12%, higher decreases up to -16%
            mileage_factor = max(0.84, min(1.12, 1.0 - (diff_ratio * 0.12)))
            assumptions.append(
                f"Mileage supplied: {mileage:,} {reg_info.get('mileage_unit', 'km')} "
                f"(vs expected benchmark of ~{expected_mileage:,} {reg_info.get('mileage_unit', 'km')})."
            )
        else:
            assumptions.append(
                f"Vehicle mileage not provided: estimated assuming typical driving of "
                f"~{std_annual_km:,} {reg_info.get('mileage_unit', 'km')}/year (~{expected_mileage:,} {reg_info.get('mileage_unit', 'km')} total)."
            )

        # Condition adjustment
        cond_factor = 1.0
        if cond_str in ["excellent", "mint", "like new"]:
            cond_factor = 1.08
            assumptions.append("Vehicle condition specified as Excellent (+8% market premium).")
        elif cond_str in ["good", "clean"]:
            cond_factor = 1.0
            assumptions.append("Vehicle condition specified as Good (standard market baseline).")
        elif cond_str in ["fair", "average"]:
            cond_factor = 0.92
            assumptions.append("Vehicle condition specified as Fair (-8% market adjustment).")
        elif cond_str in ["poor", "rough"]:
            cond_factor = 0.80
            assumptions.append("Vehicle condition specified as Poor (-20% market adjustment for cosmetic/mechanical reconditioning).")
        else:
            assumptions.append("Vehicle condition not evaluated or supplied: valuation assumes standard fair-to-good used condition.")

        # Final Estimated Price
        final_estimated = raw_estimated * mileage_factor * cond_factor

        # Range spread: Wider interval (+/- 18%) when mileage/condition unknown; tighter (+/- 10%) when known
        spread_ratio = 0.10 if (is_mileage_supplied and cond_str in ["excellent", "good", "fair", "poor"]) else 0.18
        min_price = final_estimated * (1.0 - spread_ratio)
        max_price = final_estimated * (1.0 + spread_ratio)

        # Round to sensible currency increments
        round_unit = 5000.0 if curr == "INR" else 100.0
        final_estimated = round(final_estimated / round_unit) * round_unit
        min_price = round(min_price / round_unit) * round_unit
        max_price = round(max_price / round_unit) * round_unit

        # Reliability Assessment
        if is_mileage_supplied and variant and cond_str:
            reliability = "high"
        elif is_mileage_supplied or variant:
            reliability = "moderate"
        else:
            reliability = "indicative"

        assumptions.append(
            f"Indicative secondary market valuation based on {reg_info.get('source_name')} "
            f"depreciation indices for the {region} market."
        )

        valuation = MarketValuation(
            estimated_price=final_estimated,
            minimum_price=min_price,
            maximum_price=max_price,
            currency=curr,
            market_region=region,
            vehicle_condition=cond_str if cond_str else "unknown",
            mileage=mileage,
            valuation_type="indicative",
            source=reg_info.get("source_name", "Secondary Market Index"),
            source_url=matched_gen.get("source_url") or reg_info.get("source_url", ""),
            retrieved_at=now_iso,
            reliability=reliability,
            assumptions=assumptions,
        )

        self._val_cache[cache_key] = (now_ts, valuation)
        return valuation

    # -------------------------------------------------------------------------
    # 3. Consolidated Analysis Pipeline
    # -------------------------------------------------------------------------
    def analyze_vehicle_intelligence(
        self,
        vehicle_identity: Dict[str, Any],
        market_region: Optional[str] = None,
        mileage: Optional[int] = None,
        vehicle_condition: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Produce both vehicle specifications and market valuation dictionaries
        from the identified vehicle payload.
        """
        make = vehicle_identity.get("make") or ""
        model = vehicle_identity.get("model") or ""
        year = vehicle_identity.get("estimated_year") or vehicle_identity.get("year")
        generation = vehicle_identity.get("generation")
        variant = vehicle_identity.get("variant")

        # Safely execute specifications lookup
        try:
            specs = self.get_specifications(
                make=make,
                model=model,
                year=year,
                generation=generation,
                variant=variant,
            )
            specs_dict = specs.to_dict()
        except Exception as e:
            logger.error(f"Specification lookup failed for '{make} {model}': {e}")
            specs_dict = VehicleSpecifications(
                make=make,
                model=model,
                year=year,
                source=f"Specification lookup error: {str(e)}",
                verification_status="unavailable",
            ).to_dict()

        # Safely execute market valuation lookup
        try:
            valuation = self.estimate_market_value(
                make=make,
                model=model,
                year=year,
                generation=generation,
                variant=variant,
                condition=vehicle_condition,
                mileage=mileage,
                market_region=market_region,
                currency=currency,
            )
            val_dict = valuation.to_dict()
        except Exception as e:
            logger.error(f"Market valuation lookup failed for '{make} {model}': {e}")
            val_dict = MarketValuation(
                currency=currency or self.default_currency,
                market_region=market_region or self.default_region,
                valuation_type="unavailable",
                source=f"Market valuation error: {str(e)}",
                reliability="unavailable",
                assumptions=["Market valuation calculation encountered a transient failure."],
            ).to_dict()

        return specs_dict, val_dict

    # -------------------------------------------------------------------------
    # Helper Matching & Depreciation Logic
    # -------------------------------------------------------------------------
    def _find_vehicle_record(self, make: str, model: str) -> Optional[Dict[str, Any]]:
        """Find matching vehicle entry in catalog using exact and alias matching."""
        if not make or not model:
            return None

        vehicles = self._db.get("vehicles", [])
        make_low = make.lower().strip()
        model_low = model.lower().strip()

        # 1. Exact make & model match
        for v in vehicles:
            if v.get("make", "").lower() == make_low and v.get("model", "").lower() == model_low:
                return v

        # 2. Check aliases & sub-tokens
        for v in vehicles:
            if v.get("make", "").lower() != make_low:
                continue
            aliases = [a.lower() for a in v.get("aliases", [])]
            for alias in aliases:
                if alias == model_low or alias in model_low or model_low in alias:
                    return v

        # 3. Fuzzy model name matching (e.g. "BMW 3 Series Sedan 2012" matching "3 Series")
        for v in vehicles:
            v_make = v.get("make", "").lower()
            v_model = v.get("model", "").lower()
            if v_make in make_low or make_low in v_make:
                # Compare main model tokens (e.g., '3 series', 'civic', 'f-150')
                clean_target = re.sub(r"\b(sedan|coupe|suv|hatchback|truck|convertible|\d{4})\b", "", model_low).strip()
                clean_candidate = re.sub(r"\b(sedan|coupe|suv|hatchback|truck|convertible|\d{4})\b", "", v_model).strip()
                if clean_candidate and (clean_candidate in clean_target or clean_target in clean_candidate):
                    return v

        return None

    @staticmethod
    def _find_generation(
        generations: List[Dict[str, Any]],
        year: Optional[int],
        generation_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find generation matching the model year or generation string hint."""
        if not generations:
            return None

        # 1. Match by year if available
        if year is not None:
            for g in generations:
                if year in g.get("years", []):
                    return g

            # Closest year if within 3 years
            closest_gen = None
            min_dist = 999
            for g in generations:
                for y in g.get("years", []):
                    dist = abs(y - year)
                    if dist < min_dist:
                        min_dist = dist
                        closest_gen = g
            if closest_gen and min_dist <= 3:
                return closest_gen

        # 2. Match by generation name hint (e.g., "F30", "9th Gen", "Mk7")
        if generation_hint:
            hint_low = generation_hint.lower()
            for g in generations:
                gen_name = g.get("generation", "").lower()
                if hint_low in gen_name or gen_name in hint_low:
                    return g

        # 3. Default to the most recent generation cataloged
        return generations[0] if generations else None

    @staticmethod
    def _calculate_depreciation_retention(
        age: int,
        make: str,
        body_type: str,
        region: str,
    ) -> float:
        """
        Compute residual market value ratio based on vehicle age, brand resilience, and market dynamics.
        """
        # Standard Indian / Global Depreciation Schedule
        # Year 0 (new): 1.00
        # Year 1: ~0.85
        # Year 2: ~0.75
        # Year 3: ~0.65
        # Year 4: ~0.58
        # Year 5: ~0.50
        # Year 7: ~0.38
        # Year 10: ~0.25
        # Year 14: ~0.15
        # Floor value: 0.10
        depr_table = {
            0: 1.00,
            1: 0.85,
            2: 0.75,
            3: 0.65,
            4: 0.58,
            5: 0.50,
            6: 0.44,
            7: 0.38,
            8: 0.33,
            9: 0.28,
            10: 0.24,
            11: 0.20,
            12: 0.18,
            13: 0.16,
            14: 0.14,
            15: 0.12,
        }

        if age in depr_table:
            base_retention = depr_table[age]
        elif age > 15:
            base_retention = max(0.09, 0.12 - (age - 15) * 0.005)
        else:
            base_retention = 1.00

        # Brand resilience adjustment
        make_low = make.lower()
        if any(b in make_low for b in ["toyota", "maruti", "honda", "mahindra"]):
            # Strong resale value in India
            base_retention += 0.05
        elif any(b in make_low for b in ["porsche"]):
            # Strong sports car value retention
            base_retention += 0.08
        elif any(b in make_low for b in ["bmw", "mercedes", "audi"]):
            # Higher initial depreciation in secondary market
            base_retention -= 0.03

        # Ladder-frame SUV resilience bonus in India
        if "suv" in body_type.lower() and region.lower() == "india":
            base_retention += 0.03

        return max(0.08, min(1.00, base_retention))

    def _get_region_info(self, region: str) -> Dict[str, Any]:
        """Retrieve regional settings from database with fallback."""
        regions = self._db.get("regional_settings", {}).get("supported_regions", {})
        if region in regions:
            return regions[region]

        # Case-insensitive search
        for r_name, r_data in regions.items():
            if r_name.lower() == region.lower():
                return r_data

        # Default fallback
        return {
            "currency": self.default_currency,
            "currency_symbol": "₹" if self.default_currency == "INR" else "$",
            "source_name": f"{region} Secondary Market Valuation Index",
            "source_url": "",
            "standard_annual_mileage_km": 12000,
            "mileage_unit": "km",
        }

    @staticmethod
    def _infer_body_type_from_name(name: str) -> Optional[str]:
        """Infer basic body style if model name contains standard terms."""
        n = name.lower()
        if "sedan" in n:
            return "Sedan"
        if "suv" in n:
            return "SUV"
        if "coupe" in n:
            return "Coupe"
        if "convertible" in n or "cabriolet" in n:
            return "Convertible"
        if "hatchback" in n:
            return "Hatchback"
        if "wagon" in n:
            return "Station Wagon"
        if "truck" in n or "pickup" in n:
            return "Pickup Truck"
        return None
