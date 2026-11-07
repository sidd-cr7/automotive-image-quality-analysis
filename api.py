"""FastAPI backend application for AutoVision AI Vehicle Recognition & Multi-Model Intelligence."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import logging
import gc
import threading
from typing import TYPE_CHECKING, Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    DEVICE,
    NUM_CLASSES,
    DEFAULT_DETECTION_CONF,
    DEFAULT_EXTERIOR_CONF,
    ENABLE_PREDICTION_COMPARISON,
    COMPARE_GROQ_VERSIONS,
    PREDICTION_COMPARISON_MODE,
)


# Pydantic Schemas for Documentation and Serialization
class QualityMetricsModel(BaseModel):
    brightness: float
    contrast: float
    blur_score: float


class PreprocessingModel(BaseModel):
    applied_transforms: List[str]
    processing_time_ms: float
    metrics_before: QualityMetricsModel
    metrics_after: QualityMetricsModel


class ClassicalCVModel(BaseModel):
    edge_density: float
    sobel_gradient_mean: float
    orb_keypoint_count: int
    dominant_hue: int
    mean_saturation: float
    mean_value: float
    feature_dim: int


class PredictionItemModel(BaseModel):
    class_id: int
    make: str
    model: str
    year: int
    label: str
    confidence: float


class VehiclePredictionModel(BaseModel):
    vehicle_index: int
    box: List[int] = Field(description="Bounding box [x1, y1, x2, y2]")
    detection_confidence: float
    detection_class: str
    is_fallback: bool
    make: str
    model: str
    year: int
    label: str
    classification_confidence: float
    is_uncertain: Optional[bool] = None
    top5: List[PredictionItemModel]
    preprocessing: Optional[PreprocessingModel] = None
    classical_cv: Optional[ClassicalCVModel] = None
    visual_ai_reasoning: Optional[Dict[str, Any]] = None


# Part 2: Image Quality Assessment & Adaptive Enhancement Schemas
class ImageQualityModel(BaseModel):
    overall_score: float
    detected_issues: List[str]
    quality_metrics: Dict[str, Any]
    enhancements_applied: List[Dict[str, Any]]
    enhancement_success: bool
    quality_gain: Optional[float] = None
    recommendations: Optional[List[str]] = None


class ImageVersionsModel(BaseModel):
    original_preserved: bool
    enhanced_available: bool
    selected_for_vehicle_identification: str
    selected_for_exterior_analysis: str
    selected_for_interior_analysis: str


# Part 3: Prediction Comparison & Intelligent Image Selection Schemas
class BranchComparisonModel(BaseModel):
    branch_name: Optional[str] = None
    selected_version: str
    comparison_score: float
    selection_reason: str
    predictions_agree: Optional[bool] = None
    original_prediction: Optional[Any] = None
    enhanced_prediction: Optional[Any] = None
    metrics: Optional[Dict[str, Any]] = None


class PredictionComparisonModel(BaseModel):
    vehicle_identification: Optional[BranchComparisonModel] = None
    exterior_analysis: Optional[BranchComparisonModel] = None
    interior_engine_analysis: Optional[BranchComparisonModel] = None
    comparison_enabled: Optional[bool] = True
    comparison_mode: Optional[str] = "full"


# Part 4: Vehicle Specifications & Current Market Price Intelligence Schemas
class EngineSpecificationModel(BaseModel):
    displacement: Optional[str] = None
    configuration: Optional[str] = None
    fuel_type: Optional[str] = None
    aspiration: Optional[str] = None
    horsepower: Optional[int] = None
    torque_nm: Optional[int] = None
    cylinders: Optional[int] = None


class VehicleSpecificationsModel(BaseModel):
    make: str
    model: str
    year: Optional[int] = None
    generation: Optional[str] = None
    variant: Optional[str] = None
    engine: Optional[EngineSpecificationModel] = None
    transmission: Optional[str] = None
    drivetrain: Optional[str] = None
    body_type: Optional[str] = None
    seating_capacity: Optional[int] = None
    source: str
    verification_status: str  # database_verified | database_multiple_variants | model_level_only | verified | unavailable
    variants: List[Dict[str, Any]] = []  # every real configuration the database lists for this model year


class MarketValuationModel(BaseModel):
    estimated_price: Optional[float] = None
    minimum_price: Optional[float] = None
    maximum_price: Optional[float] = None
    currency: str = "INR"
    market_region: str = "India"
    vehicle_condition: str = "unknown"
    mileage: Optional[int] = None
    valuation_type: str = "indicative"  # "indicative", "live", "unavailable"
    source: str
    source_url: str = ""
    retrieved_at: str
    reliability: str
    assumptions: List[str] = []


# Multi-Model Perception Intelligence Schemas
class VehicleIdentityModel(BaseModel):
    make: str
    model: Optional[str] = None  # None: only the manufacturer is known
    model_known: bool = True
    identity_conflict: bool = False
    valuation_eligible: bool = False
    valuation_blocker: str = ""
    estimated_year: Optional[int] = None
    generation: Optional[str] = None
    # None when the Groq vision model changed the identification (it has no calibrated confidence).
    confidence: Optional[float] = None
    classifier_confidence: Optional[float] = None
    classifier_top1: Optional[str] = None
    identification_source: str = "classifier_only"  # classifier_only | classifier_confirmed_by_<provider> | <provider>_vision
    identification_note: Optional[str] = None
    vision_provider: Optional[str] = None
    is_uncertain: bool = False


class ExteriorComponentItemModel(BaseModel):
    class_id: int
    name: str
    confidence: float
    box: List[int] = Field(description="Bounding box [x1, y1, x2, y2]")
    area_pixels: int
    area_ratio: float
    polygon_points_count: int


class ExteriorAnalysisModel(BaseModel):
    status: str
    total_components_detected: int
    component_counts: Dict[str, int]
    detected_classes: List[str]
    components: List[ExteriorComponentItemModel]
    possible_modifications: List[Dict[str, Any]]
    orientation_evidence: List[str]


class InteriorAnalysisModel(BaseModel):
    status: str
    is_interior_visible: bool
    view_perspective: str
    interior_components: List[Dict[str, Any]]
    identity_cues: List[str]
    notes: str


class VisibleEngineModel(BaseModel):
    status: str
    is_engine_visible: bool
    engine_components: List[Dict[str, Any]]
    notes: str


class ConfidenceUncertaintyModel(BaseModel):
    vehicle_identification_confidence: float
    exterior_segmentation_confidence: float
    interior_engine_confidence: float
    overall_uncertainty_assessment: str


class FinalAIVerdictModel(BaseModel):
    verdict_status: str
    verdict_summary: str
    confirmed_facts: List[str]
    visual_inferences: List[str]
    corrections_made: List[Dict[str, Any]]
    # Which vision provider produced the analysis (None = local models only) and what was tried.
    provider: Optional[str] = None
    provider_cached: bool = False
    provider_attempts: List[Dict[str, Any]] = []


class PredictResponse(BaseModel):
    status: str
    # Real original / enhanced frames from the existing enhancer (small JPEG data URLs), or None.
    enhancement_visuals: Optional[Dict[str, Any]] = None
    image_dimensions: Dict[str, int]
    vehicles_detected: int
    predictions_count: int
    processing_time_ms: float
    predictions: List[VehiclePredictionModel]

    # Part 2: Intelligent Image Quality Assessment & Adaptive Enhancement
    image_quality: Optional[ImageQualityModel] = None
    image_versions: Optional[ImageVersionsModel] = None

    # Part 3: Prediction Comparison & Intelligent Image Selection
    prediction_comparison: Optional[PredictionComparisonModel] = None
    selected_predictions: Optional[Dict[str, Any]] = None
    original_predictions: Optional[Dict[str, Any]] = None
    enhanced_predictions: Optional[Dict[str, Any]] = None

    # Part 4: Vehicle Specifications & Current Market Price Intelligence
    vehicle_specifications: Optional[VehicleSpecificationsModel] = None
    market_valuation: Optional[MarketValuationModel] = None

    # Consolidated Multi-Model Intelligence Fields
    vehicle: Optional[VehicleIdentityModel] = None
    exterior_analysis: Optional[ExteriorAnalysisModel] = None
    interior_analysis: Optional[InteriorAnalysisModel] = None
    visible_engine_info: Optional[VisibleEngineModel] = None
    confidence_and_uncertainty: Optional[ConfidenceUncertaintyModel] = None
    cross_model_disagreements: Optional[List[str]] = None
    final_ai_verdict: Optional[FinalAIVerdictModel] = None
    missing_or_unavailable_information: Optional[List[str]] = None
    original_model_outputs: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    device: str
    models: Dict[str, Any]


# Lazy/Global Pipeline Singleton
if TYPE_CHECKING:  # not imported at runtime: src.pipeline pulls in torch / ultralytics (hundreds of MB)
    from src.pipeline import VehicleRecognitionPipeline

pipeline: Optional["VehicleRecognitionPipeline"] = None
_pipeline_lock = threading.Lock()
# Memory safety: run one inference at a time by default (raise on bigger instances).
_inference_gate = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_INFERENCES", "1")))


def get_pipeline() -> "VehicleRecognitionPipeline":
    """Create the pipeline on first use (single-flight). The heavy models inside it load lazily too."""
    global pipeline
    if pipeline is None:
        with _pipeline_lock:
            if pipeline is None:
                from src.pipeline import VehicleRecognitionPipeline
                pipeline = VehicleRecognitionPipeline()
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start instantly: no models are loaded here (Render's 512 MB tier cannot hold them at boot)."""
    yield


# Initialize FastAPI app
app = FastAPI(
    title="AutoVision AI - Multi-Model Vehicle Intelligence API",
    description=(
        "Upgraded Multi-Model Vehicle Intelligence Platform combining:\n"
        "1. Part 2: Intelligent Image Quality Assessment & Adaptive CV Enhancement Engine\n"
        "2. Part 3: Original vs Enhanced Prediction Comparison & Intelligent Selection Engine\n"
        "3. Branch 1: Vehicle Detection (YOLOv8) & Fine-Grained Classification (ResNet-50) with Visual AI fallback\n"
        "4. Branch 2: Exterior Component Detection & Instance Segmentation (YOLO11n-seg 23 classes) with modification detection\n"
        "5. Branch 3: Interior & Engine Cabin Analysis (Groq Vision with Primary + Backup failover)\n"
        "6. Final Groq Verdict Engine: Cross-branch synthesis, verification, contradiction checking, and factual report generation."
    ),
    version="2.2.0",
    lifespan=lifespan,
)

# Enable CORS
# Production serves the UI from this same origin, so no CORS is needed there. For the Vite dev server
# (or another trusted frontend) set CORS_ORIGINS="https://a.example,https://b.example".
_cors_origins = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Lightweight health check. Never builds the pipeline or loads any model."""
    from src.config import GROQ_API_KEY, GROQ_API_KEY_BACKUP, GEMINI_API_KEY, GROQ_VISION_MODEL, VISION_PROVIDER_ORDER
    pipe = pipeline  # may be None until the first /predict
    loaded = lambda m: bool(getattr(m, "loaded", m is not None)) if pipe is not None else False
    vision = pipe.groq_service.status() if pipe is not None and hasattr(pipe.groq_service, "status") else {
        "order": [p.strip() for p in VISION_PROVIDER_ORDER.split(",") if p.strip()],
        "providers": {"gemini": {"configured": bool(GEMINI_API_KEY)}, "groq": {"configured": bool(GROQ_API_KEY or GROQ_API_KEY_BACKUP)}},
    }
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        device=DEVICE,
        models={
            "detector": "YOLOv8n",
            "classifier": "ResNet50",
            "exterior_segmentation": "YOLO11n-seg",
            "image_quality_engine": "Active (Part 2 Assessor + Adaptive Enhancer)",
            "prediction_comparison_engine": "Active (Part 3 Comparator + Intelligent Selection)",
            "vehicle_intelligence_engine": "Active (Part 4 Specifications & Market Valuation Engine)",
            "num_classes": NUM_CLASSES,
            "classes_mapping": "Stanford Cars (196 classes)",
            "exterior_classes_count": 23,
            "models_loaded": pipe is not None,
            "detector_ready": loaded(pipe.detector) if pipe is not None else False,
            "classifier_ready": loaded(pipe.classifier) if pipe is not None else False,
            "exterior_ready": loaded(pipe.exterior_detector) if pipe is not None else False,
            "groq_primary_configured": bool(GROQ_API_KEY),
            "groq_backup_configured": bool(GROQ_API_KEY_BACKUP),
            "groq_vision_model": GROQ_VISION_MODEL,
            "vision_providers": vision,
        },
    )


@app.get("/progress/{job_id}", include_in_schema=False)
async def get_progress(job_id: str):
    """Real stage progress for an in-flight /predict request (see src/progress.py)."""
    from src import progress as _progress
    state = _progress.get(job_id)
    if state is None:
        # Not registered yet (the client polls while the upload / first model import is still running).
        return {"current": None, "completed": [], "done": False, "known": False, "elapsed_s": 0.0}
    return {**state, "known": True}


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Vehicle Detection, Quality Enhancement & Prediction Comparison",
    tags=["Inference"],
)
async def predict_vehicle(
    file: UploadFile = File(..., description="Vehicle image file (JPEG, PNG, WEBP, BMP)"),
    conf_threshold: float = Query(
        DEFAULT_DETECTION_CONF, ge=0.01, le=1.0, description="Detection confidence threshold"
    ),
    apply_preprocessing: bool = Query(
        True, description="Enable adaptive enhancement (CLAHE, Gamma, Bilateral, Sharpen)"
    ),
    include_classical_cv: bool = Query(
        True, description="Extract Canny, Sobel, ORB, and HSV classical CV features"
    ),
    fallback_to_full_image: bool = Query(
        True, description="Classify the whole image if no vehicle is detected by YOLO"
    ),
    enable_multimodal: bool = Query(
        True, description="Execute full multi-model architecture (Branch 1, 2, 3 + Verdict)"
    ),
    enable_quality_enhancement: bool = Query(
        True, description="Enable Part 2 Intelligent Image Quality Assessment and Adaptive Enhancement"
    ),
    enable_prediction_comparison: bool = Query(
        ENABLE_PREDICTION_COMPARISON, description="Enable Part 3 Original vs Enhanced prediction comparison"
    ),
    compare_groq_versions: bool = Query(
        COMPARE_GROQ_VERSIONS, description="Enable dual Groq vision analysis for original and enhanced images"
    ),
    comparison_mode: str = Query(
        PREDICTION_COMPARISON_MODE, description="Comparison mode: 'full', 'fast', 'off'"
    ),
    enable_groq_inspection: bool = Query(
        True, description="Enable Groq inspection for exterior modifications & interior/engine"
    ),
    enable_final_verdict: bool = Query(
        True, description="Enable final Groq cross-branch verification stage"
    ),
    exterior_conf_threshold: float = Query(
        DEFAULT_EXTERIOR_CONF, ge=0.01, le=1.0, description="Exterior component detection threshold"
    ),
    market_region: str = Query(
        "India", description="Regional automotive market for valuation (India, USA, UK, Europe)"
    ),
    mileage: Optional[int] = Query(
        None, ge=0, le=1000000, description="Optional vehicle odometer reading in km or miles"
    ),
    vehicle_condition: Optional[str] = Query(
        None, description="Optional vehicle condition (excellent, good, fair, poor)"
    ),
    currency: Optional[str] = Query(
        None, description="Optional target currency code (INR, USD, EUR, GBP)"
    ),
    job_id: Optional[str] = Query(
        None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
        description="Optional client-generated id for polling GET /progress/{job_id}",
    ),
) -> PredictResponse:
    """
    Accepts an uploaded vehicle image and executes:
    - Pre-inference: Intelligent Image Quality Assessment & Adaptive Enhancement (Part 2)
    - Part 3: Original vs Enhanced Prediction Comparison & Intelligent Selection
    - Branch 1: Vehicle identification & classification
    - Branch 2: Exterior component detection & segmentation (23 Carparts-Seg classes)
    - Branch 3: Interior & engine cabin analysis (Groq Vision with failover)
    - Final Verdict: Synthesized verification and fact/inference distinction
    """
    if job_id:
        from src import progress as _progress
        _progress.start(job_id)

    # 1. Validate File Existence and MIME/Filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload.",
        )

    content_type = file.content_type or ""
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"]
    if content_type and content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WEBP, BMP.",
        )

    # 2. Read File Bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {str(e)}",
        )

    from src.config import MAX_UPLOAD_BYTES
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image is too large ({len(image_bytes) / 1048576:.1f} MB). Maximum is {MAX_UPLOAD_BYTES // 1048576} MB.",
        )

    if not image_bytes or len(image_bytes) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or too small to be a valid image.",
        )

    # 3. Execute Pipeline (pipeline creation imports torch on first use, so keep it off the event loop)
    pipe = await asyncio.to_thread(get_pipeline)
    try:
      async with _inference_gate:
        result = await pipe.process_async(
            image_input=image_bytes,
            conf_threshold=conf_threshold,
            apply_preprocessing=apply_preprocessing,
            include_classical_cv=include_classical_cv,
            fallback_to_full_image=fallback_to_full_image,
            enable_multimodal=enable_multimodal,
            enable_quality_enhancement=enable_quality_enhancement,
            enable_prediction_comparison=enable_prediction_comparison,
            compare_groq_versions=compare_groq_versions,
            comparison_mode=comparison_mode,
            enable_groq_inspection=enable_groq_inspection,
            enable_final_verdict=enable_final_verdict,
            exterior_conf_threshold=exterior_conf_threshold,
            market_region=market_region,
            mileage=mileage,
            vehicle_condition=vehicle_condition,
            currency=currency,
        )
    except ValueError as val_err:
        logging.getLogger("autovision.api").warning("predict rejected input: %s", val_err)
        if job_id:
            _progress.finish()
        raise HTTPException(
            status_code=422,
            detail="Image validation error: the file could not be decoded as a valid image (JPEG, PNG, WEBP or BMP).",
        )
    except Exception as exc:
        logging.getLogger("autovision.api").exception("predict failed (%s: %s)", type(exc).__name__, exc)
        if job_id:
            _progress.finish()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(exc)}",
        )

    if job_id:
        _progress.finish()
    response = PredictResponse(**result)
    del result, image_bytes  # do not retain the upload or raw prediction data
    gc.collect()
    return response


@app.post(
    "/analyze",
    response_model=PredictResponse,
    summary="Comprehensive Multi-Model Vehicle Intelligence Analysis",
    tags=["Inference"],
)
async def analyze_vehicle_comprehensive(
    file: UploadFile = File(..., description="Vehicle image file (JPEG, PNG, WEBP, BMP)"),
    conf_threshold: float = Query(DEFAULT_DETECTION_CONF, ge=0.01, le=1.0),
    exterior_conf_threshold: float = Query(DEFAULT_EXTERIOR_CONF, ge=0.01, le=1.0),
    market_region: str = Query("India", description="Regional automotive market"),
    mileage: Optional[int] = Query(None, ge=0),
    vehicle_condition: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
) -> PredictResponse:
    """Dedicated endpoint for full multi-model vehicle intelligence analysis with prediction comparison."""
    return await predict_vehicle(
        file=file,
        conf_threshold=conf_threshold,
        exterior_conf_threshold=exterior_conf_threshold,
        enable_multimodal=True,
        enable_quality_enhancement=True,
        enable_prediction_comparison=True,
        compare_groq_versions=False,
        comparison_mode="full",
        enable_groq_inspection=True,
        enable_final_verdict=True,
        market_region=market_region,
        mileage=mileage,
        vehicle_condition=vehicle_condition,
        currency=currency,
    )


# -----------------------------------------------------------------------------
# Mount Built Frontend (Editorial Surrealism UI)
# -----------------------------------------------------------------------------
frontend_dist = Path(__file__).resolve().parent / "frontend" / "dist"
if frontend_dist.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

