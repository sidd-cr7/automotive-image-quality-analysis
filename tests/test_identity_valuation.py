"""Identity resolution, specification and valuation honesty tests."""

import numpy as np
import pytest

from src.orchestrator import MultiModelOrchestrator
from src.vehicle_intelligence import VehicleIntelligenceService
from src.vision_service import VisionService

svc = VehicleIntelligenceService()


# ---- model matching: a make or a body style never becomes a model ----
@pytest.mark.parametrize("make,model", [("Dodge", ""), ("Dodge", "Coupe"), ("Dodge", "Unknown"), ("McLaren", "MP4-12C Coupe"), ("Dodge", "Sedan 2012")])
def test_make_only_or_body_style_matches_nothing(make, model):
    assert svc._find_vehicle_record(make, model) is None


def test_exact_model_still_matches():
    assert svc._find_vehicle_record("Dodge", "Challenger")["model"] == "Challenger"
    assert svc._find_vehicle_record("BMW", "3 Series Sedan 2012")["make"] == "BMW"


def test_dodge_without_model_has_unavailable_specs():
    spec = svc.get_specifications("Dodge", "", None)
    assert spec.verification_status == "unavailable" and spec.engine.displacement is None


def test_specs_never_use_a_default_variant_engine():
    spec = svc.get_specifications("Dodge", "Challenger", 2011)   # model known, variant NOT known
    assert spec.verification_status == "model_level_only"
    assert spec.engine.displacement is None and spec.engine.horsepower is None
    assert spec.transmission is None and spec.drivetrain is None and spec.variant is None


# ---- valuation ----
def test_unofficial_india_import_gets_no_price():
    v = svc.estimate_market_value("Dodge", "Challenger", 2011, market_region="India")
    assert v.estimated_price is None and v.valuation_type == "unavailable"
    assert "not officially sold in India" in v.assumptions[0]


def test_missing_year_gets_no_price_not_an_assumed_age():
    v = svc.estimate_market_value("Honda", "Civic", None, market_region="India")
    assert v.estimated_price is None and "model year" in v.assumptions[0]


def test_unknown_model_gets_no_price():
    assert svc.estimate_market_value("McLaren", "MP4-12C Coupe", 2012).estimated_price is None


# ---- identity resolution ----
CLF = {"make": "McLaren", "model": "MP4-12C Coupe", "label": "McLaren MP4-12C Coupe 2012", "year": 2012, "is_uncertain": True}


def test_gemini_make_only_leaves_model_unknown_and_does_not_borrow_classifier_model():
    vs = VisionService(primary_api_key="", backup_api_key="")
    vs.analyze_frame = lambda image: {"ok": True, "provider": "gemini", "cached": False, "attempts": [], "error": None,
                                      "result": {"vehicle": {"make": "Dodge", "model": None, "year_estimate": 2011, "visible_evidence": ["Dodge badge"]}, "summary": "Dodge"}}
    v = vs.generate_final_verdict({"predictions": [CLF]}, {"predictions": {}}, {}, np.zeros((8, 8, 3), np.uint8))
    assert v["final_make"] == "Dodge" and v["final_model"] is None and v["final_estimated_year"] is None


def _block(make, model, year, verdict, clf=CLF, conf=0.057):
    return MultiModelOrchestrator._vehicle_identity_block(make, model, year, None, clf, conf, verdict)


def test_disagreement_is_flagged_uncertain_with_no_confidence_and_no_valuation():
    b = _block("Dodge", None, None, {"verdict_status": "provider_assessed", "verdict_provider": "gemini"})
    assert b["identity_conflict"] is True and b["model_known"] is False and b["model"] is None
    assert b["confidence"] is None and b["is_uncertain"] is True
    assert b["valuation_eligible"] is False and "manufacturer" in b["valuation_blocker"]
    assert b["classifier_top1"] == "McLaren MP4-12C Coupe"


def test_full_agreement_with_confident_classifier_is_eligible_for_valuation():
    clf = {"make": "Honda", "model": "Civic Sedan", "label": "Honda Civic Sedan 2012", "year": 2012, "is_uncertain": False}
    b = _block("Honda", "Civic Sedan", 2012, {"verdict_status": "provider_assessed", "verdict_provider": "gemini"}, clf, 0.8)
    assert b["identity_conflict"] is False and b["valuation_eligible"] is True and b["confidence"] == 0.8


def test_provider_exact_model_with_weak_classifier_noise_can_be_priced_from_live_listings():
    b = _block("Dodge", "Challenger", 2018, {"verdict_status": "provider_assessed", "verdict_provider": "gemini"})
    assert b["model_known"] is True and b["identity_conflict"] is True  # still flagged and shown
    assert b["valuation_eligible"] is True and b["is_uncertain"] is True  # priced, but labelled AI-identified / unverified


def test_provider_model_contradicting_a_confident_classifier_is_not_priced():
    clf = {"make": "McLaren", "model": "MP4-12C Coupe", "label": "McLaren MP4-12C Coupe 2012", "year": 2012, "is_uncertain": False}
    b = _block("Dodge", "Challenger", 2018, {"verdict_status": "provider_assessed", "verdict_provider": "gemini"}, clf, 0.6)
    assert b["valuation_eligible"] is False and "confident classifier" in b["valuation_blocker"]
