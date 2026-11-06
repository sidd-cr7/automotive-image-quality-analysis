"""Configuration for the Vehicle Recognition Backend."""

import os
from pathlib import Path
import torch
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
DEVICE = os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

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

# Groq API configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP", os.getenv("GROQ_BACKUP_API_KEY", ""))
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
GROQ_VERDICT_MODEL = os.getenv("GROQ_VERDICT_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "20.0"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))

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
