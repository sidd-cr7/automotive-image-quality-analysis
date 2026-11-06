"""
Comprehensive Test Suite for Part 4: Vehicle Specifications & Current Market Price Intelligence.
Verifies specification retrieval, regional pricing, assumption transparency, and fail-safe operation.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.vehicle_intelligence import (
    VehicleIntelligenceService,
    VehicleSpecifications,
    MarketValuation,
    EngineSpecification,
)
from src.orchestrator import MultiModelOrchestrator
from src.pipeline import VehicleRecognitionPipeline
from api import app

client = TestClient(app)


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Standard synthetic vehicle image."""
    arr = np.full((240, 320, 3), 128, dtype=np.uint8)
    arr[60:180, 50:270] = [20, 80, 160]
    return arr


# -----------------------------------------------------------------------------
# 1. Correct Specification Retrieval
# -----------------------------------------------------------------------------
def test_correct_specification_retrieval():
    """Verify accurate extraction of engine, displacement, power, torque, and drivetrain for cataloged vehicle."""
    service = VehicleIntelligenceService()

    spec = service.get_specifications(
        make="BMW",
        model="3 Series Sedan",
        year=2012,
        variant="328i Luxury Line",
    )

    assert spec.make == "BMW"
    assert spec.model == "3 Series Sedan"
    assert spec.year == 2012
    assert "F30" in str(spec.generation)
    assert spec.verification_status == "verified"
    assert spec.engine.cylinders == 4
    assert spec.engine.horsepower == 241
    assert spec.engine.torque_nm == 350
    assert "Turbocharged" in str(spec.engine.aspiration)
    assert spec.drivetrain == "RWD"
    assert spec.seating_capacity == 5
    assert "BMW Group" in spec.source


# -----------------------------------------------------------------------------
# 2. Unknown Vehicle Variant
# -----------------------------------------------------------------------------
def test_unknown_vehicle_variant():
    """When exact variant is unconfirmed, system provides base generation specs and marks status as estimated."""
    service = VehicleIntelligenceService()

    spec = service.get_specifications(
        make="BMW",
        model="3 Series Sedan",
        year=2013,
        variant=None,  # Variant not provided
    )

    assert spec.verification_status == "estimated"
    assert spec.variant is not None
    assert spec.engine.horsepower is not None
    assert spec.engine.displacement is not None


# -----------------------------------------------------------------------------
# 3. Missing Specification Fields
# -----------------------------------------------------------------------------
def test_missing_specification_fields():
    """Uncataloged or unknown vehicles do not have fabricated specifications."""
    service = VehicleIntelligenceService()

    spec = service.get_specifications(
        make="AcmeMotors",
        model="SolarHover 3000",
        year=2025,
    )

    assert spec.verification_status == "unavailable"
    assert spec.engine.horsepower is None
    assert spec.engine.displacement is None
    assert spec.engine.cylinders is None
    assert spec.seating_capacity is None
    assert "No verified external specification source" in spec.source


# -----------------------------------------------------------------------------
# 4. Market Price Retrieval
# -----------------------------------------------------------------------------
def test_market_price_retrieval():
    """Verify calculated indicative secondary market price range for cataloged vehicle."""
    service = VehicleIntelligenceService()

    val = service.estimate_market_value(
        make="BMW",
        model="3 Series Sedan",
        year=2014,
        market_region="India",
        currency="INR",
    )

    assert val.valuation_type == "indicative"
    assert val.currency == "INR"
    assert val.market_region == "India"
    assert val.estimated_price is not None
    assert val.minimum_price is not None
    assert val.maximum_price is not None
    assert val.minimum_price < val.estimated_price < val.maximum_price
    assert "CarWale" in val.source or "CarDekho" in val.source


# -----------------------------------------------------------------------------
# 5. Unavailable Pricing Sources
# -----------------------------------------------------------------------------
def test_unavailable_pricing_sources():
    """When model or pricing indices are unavailable, transparently mark as unavailable without fabricating."""
    service = VehicleIntelligenceService()

    val = service.estimate_market_value(
        make="UnknownMake",
        model="MysteryWagon",
        year=1995,
        market_region="India",
    )

    assert val.valuation_type == "unavailable"
    assert val.estimated_price is None
    assert val.minimum_price is None
    assert val.maximum_price is None
    assert val.reliability == "unavailable"
    assert any("unavailable" in a.lower() for a in val.assumptions)


# -----------------------------------------------------------------------------
# 6. Unknown Mileage and Condition
# -----------------------------------------------------------------------------
def test_unknown_mileage_and_condition():
    """Explicitly list assumptions and widen price range when mileage or condition are unknown."""
    service = VehicleIntelligenceService()

    # Unknown mileage & condition
    val_unknown = service.estimate_market_value(
        make="Honda",
        model="Civic",
        year=2015,
        mileage=None,
        condition=None,
    )

    assert any("mileage not provided" in a.lower() for a in val_unknown.assumptions)
    assert any("condition not evaluated" in a.lower() for a in val_unknown.assumptions)

    # Known mileage & condition
    val_known = service.estimate_market_value(
        make="Honda",
        model="Civic",
        year=2015,
        mileage=65000,
        condition="good",
    )

    assert any("65,000" in a for a in val_known.assumptions)
    # The spread between min and max should be tighter when mileage/condition are known
    spread_unknown = (val_unknown.maximum_price - val_unknown.minimum_price) / val_unknown.estimated_price
    spread_known = (val_known.maximum_price - val_known.minimum_price) / val_known.estimated_price
    assert spread_known < spread_unknown


# -----------------------------------------------------------------------------
# 7. Regional Currency Handling
# -----------------------------------------------------------------------------
def test_regional_currency_handling():
    """Verify multi-region support with correct currencies (INR, USD, GBP, EUR)."""
    service = VehicleIntelligenceService()

    # India
    val_in = service.estimate_market_value("Toyota", "Camry Sedan", 2018, market_region="India")
    assert val_in.currency == "INR"
    assert val_in.market_region == "India"
    assert val_in.estimated_price > 100000.0  # In Lakhs INR

    # USA
    val_us = service.estimate_market_value("Toyota", "Camry Sedan", 2018, market_region="USA")
    assert val_us.currency == "USD"
    assert val_us.market_region == "USA"
    assert "Kelley Blue Book" in val_us.source

    # UK
    val_uk = service.estimate_market_value("Toyota", "Camry Sedan", 2018, market_region="UK")
    assert val_uk.currency == "GBP"
    assert "AutoTrader UK" in val_uk.source

    # Europe
    val_eu = service.estimate_market_value("Toyota", "Camry Sedan", 2018, market_region="Europe")
    assert val_eu.currency == "EUR"


# -----------------------------------------------------------------------------
# 8. External Source Failure Graceful Fallback
# -----------------------------------------------------------------------------
def test_external_source_failure_graceful_fallback():
    """Ensure service handles corrupt or missing database files with safe internal fallback."""
    service = VehicleIntelligenceService(db_path=Path("non_existent_automotive_db.json"))

    # Service must not raise an unhandled exception
    specs, val = service.analyze_vehicle_intelligence({
        "make": "Ford",
        "model": "Mustang",
        "estimated_year": 2016,
    })

    assert isinstance(specs, dict)
    assert isinstance(val, dict)
    assert specs.get("make") == "Ford"
    assert val.get("valuation_type") in ["indicative", "unavailable"]


# -----------------------------------------------------------------------------
# 9. API Response Integration
# -----------------------------------------------------------------------------
def test_api_response_integration():
    """Verify POST /predict returns vehicle_specifications and market_valuation schemas."""
    img = Image.new("RGB", (200, 200), color=(100, 120, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    files = {"file": ("test_part4.jpg", img_bytes, "image/jpeg")}
    response = client.post(
        "/predict?market_region=India&mileage=45000&vehicle_condition=good&enable_groq_inspection=false&enable_final_verdict=false",
        files=files,
    )

    assert response.status_code == 200
    data = response.json()

    assert "vehicle_specifications" in data
    assert "market_valuation" in data

    specs = data["vehicle_specifications"]
    assert "make" in specs
    assert "model" in specs
    assert "engine" in specs
    assert "verification_status" in specs

    val = data["market_valuation"]
    assert "currency" in val
    assert "market_region" in val
    assert "valuation_type" in val
    assert "assumptions" in val
    assert isinstance(val["assumptions"], list)


# -----------------------------------------------------------------------------
# 10. Existing Pipeline Functioning When Service Fails
# -----------------------------------------------------------------------------
def test_existing_pipeline_functioning_when_service_fails(dummy_image):
    """If vehicle intelligence raises an exception, the pipeline must NOT crash."""
    mock_service = MagicMock()
    mock_service.analyze_vehicle_intelligence.side_effect = RuntimeError("External database crashed")

    orchestrator = MultiModelOrchestrator(
        vehicle_intelligence_service=mock_service,
    )

    # analyze_sync must complete safely without raising RuntimeError
    report = orchestrator.analyze_sync(
        dummy_image,
        enable_groq_inspection=False,
        enable_final_verdict=False,
    )

    assert report["status"] in ["success", "partial"]
    assert "vehicle" in report
    assert "exterior_analysis" in report
    assert "vehicle_specifications" in report
    assert "market_valuation" in report
