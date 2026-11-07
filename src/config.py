"""Configuration for the Vehicle Recognition Backend."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DETECTION_MODEL_PATH = Path(os.getenv("DETECTION_MODEL_PATH", str(MODELS_DIR / "detection" / "yolov8n.pt")))
CLASSIFICATION_MODEL_PATH = Path(os.getenv("CLASSIFICATION_MODEL_PATH", str(MODELS_DIR / "resnet50_best.pth")))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", str(MODELS_DIR / "classes.json")))

# Exterior Segmentation Model (YOLO11n-seg trained on Carparts-Seg)
_default_exterior = PROJECT_ROOT / "exterior_best.pt"
if not _default_exterior.exists():
    _default_exterior = MODELS_DIR / "detection" / "exterior_best.pt"
EXTERIOR_MODEL_PATH = Path(os.getenv("EXTERIOR_MODEL_PATH", str(_default_exterior)))
EXTERIOR_CLASSES_PATH = Path(os.getenv("EXTERIOR_CLASSES_PATH", str(PROJECT_ROOT / "classes.txt")))

# Device selection
# CPU by default: importing torch here would load ~250 MB at startup. Set TORCH_DEVICE=cuda to use a GPU.
DEVICE = os.getenv("TORCH_DEVICE", "cpu")

# YOLO vehicle detection parameters
# COCO class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
DEFAULT_DETECTION_CONF = float(os.getenv("DEFAULT_DETECTION_CONF", "0.25"))
DEFAULT_IOU_THRESH = float(os.getenv("DEFAULT_IOU_THRESH", "0.45"))

# Exterior detection & segmentation parameters (23 Carparts-Seg classes)
DEFAULT_EXTERIOR_CONF = float(os.getenv("DEFAULT_EXTERIOR_CONF", "0.20"))
DEFAULT_EXTERIOR_IOU = float(os.getenv("DEFAULT_EXTERIOR_IOU", "0.45"))

# Classifier parameters & uncertainty threshold
NUM_CLASSES = 196
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSIFIER_UNCERTAINTY_THRESHOLD = float(os.getenv("CLASSIFIER_UNCERTAINTY_THRESHOLD", "0.40"))
# The classifiers were trained on WHOLE frames resized to 224x224, not tight vehicle crops.
# Classifying the YOLO crop drops accuracy from ~33% to ~8% top-1 on Stanford Cars; classify the full frame.
CLASSIFY_FULL_FRAME = os.getenv("CLASSIFY_FULL_FRAME", "true").lower() in ("true", "1", "yes")
# Average ResNet-50 and EfficientNet-B0 probabilities (both trained on the same 196 classes).
EFFICIENTNET_MODEL_PATH = Path(os.getenv("EFFICIENTNET_MODEL_PATH", str(MODELS_DIR / "efficientnet_b0_best.pth")))
CLASSIFIER_ENSEMBLE = os.getenv("CLASSIFIER_ENSEMBLE", "true").lower() in ("true", "1", "yes")

# Groq API configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP", os.getenv("GROQ_BACKUP_API_KEY", ""))
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GROQ_VERDICT_MODEL = os.getenv("GROQ_VERDICT_MODEL", "qwen/qwen3.8-27b")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "20.0"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))
# Vision providers (tried in VISION_PROVIDER_ORDER; the local classifier is the final fallback).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Fast Flash-tier alias that follows Google's current model (measured 2.7-3.9 s). Note: "gemini-2.5-flash" is
# retired for new keys and "gemini-flash-latest" timed out (>60 s) in testing.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "25"))
VISION_PROVIDER_ORDER = os.getenv("VISION_PROVIDER_ORDER", "gemini,groq")
VISION_CACHE_SIZE = int(os.getenv("VISION_CACHE_SIZE", "64"))
VISION_CACHE_TTL = float(os.getenv("VISION_CACHE_TTL", "3600"))

# Image optimisation for Groq vision payloads (longest edge in px, JPEG quality 1-100).
# This only affects images sent to Groq; YOLO / classifier inputs are untouched.
GROQ_IMAGE_MAX_DIM = int(os.getenv("GROQ_IMAGE_MAX_DIM", "1024"))
GROQ_IMAGE_JPEG_QUALITY = int(os.getenv("GROQ_IMAGE_JPEG_QUALITY", "78"))
# Reject uploads larger than this (bytes) before decoding.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
# Memory safety (512 MB hosts): photos are decoded at reduced size and never processed above MAX_IMAGE_DIM px.
# The models only look at 224-640 px inputs, so this does not reduce accuracy. Absurd resolutions are rejected.
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1024"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(50_000_000)))
# Heavy inference jobs that may run at the same time inside one request (1 = sequential, lowest peak RAM).
INFERENCE_WORKERS = max(1, int(os.getenv("INFERENCE_WORKERS", "1")))
TORCH_THREADS = max(1, int(os.getenv("TORCH_THREADS", "1")))
# Release the ~100 MB classifier right after Branch 1 uses it; it is re-mapped from disk on the next use (about 1 s).
UNLOAD_CLASSIFIER_AFTER_USE = os.getenv("UNLOAD_CLASSIFIER_AFTER_USE", "true").lower() in ("true", "1", "yes")

# Part 3: Original vs Enhanced Prediction Comparison Configuration
ENABLE_PREDICTION_COMPARISON = os.getenv("ENABLE_PREDICTION_COMPARISON", "true").lower() in ("true", "1", "yes")
COMPARE_GROQ_VERSIONS = os.getenv("COMPARE_GROQ_VERSIONS", "false").lower() in ("true", "1", "yes")
PREDICTION_COMPARISON_MODE = os.getenv("PREDICTION_COMPARISON_MODE", "full")  # "full", "fast", "off"

# Part 4: Vehicle Specifications & Current Market Price Intelligence
DEFAULT_MARKET_REGION = os.getenv("DEFAULT_MARKET_REGION", "India")
DEFAULT_MARKET_CURRENCY = os.getenv("DEFAULT_MARKET_CURRENCY", "INR")
ENABLE_VEHICLE_SPECIFICATIONS = os.getenv("ENABLE_VEHICLE_SPECIFICATIONS", "true").lower() in ("true", "1", "yes")
ENABLE_MARKET_VALUATION = os.getenv("ENABLE_MARKET_VALUATION", "true").lower() in ("true", "1", "yes")
AUTOMOTIVE_DB_PATH = Path(os.getenv("AUTOMOTIVE_DB_PATH", str(PROJECT_ROOT / "data" / "automotive_reference_data.json")))
VALUATION_CACHE_TTL_SECONDS = int(os.getenv("VALUATION_CACHE_TTL_SECONDS", "3600"))
