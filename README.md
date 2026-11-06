# ENGINE INTELLIGENCE (AUTOVISION AI)

> **Multi-Model Automotive Perception, Image Quality Assessment & Adaptive Computer Vision**  
> Built for CV Hackathon '26 | Aesthetic: *Editorial Surrealism*

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%20|%2011n--seg-00FFFF.svg)](https://ultralytics.com/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://react.dev/)
[![Tests](https://img.shields.io/badge/Tests-69%20Passing-brightgreen.svg)]()

---

## 🏛️ Overview

**ENGINE INTELLIGENCE** is an end-to-end multi-model computer vision platform for automotive recognition, component segmentation, quality defect restoration, and powertrain intelligence. 

Instead of relying on a single monolithic model or generic SaaS interfaces, the system merges **Editorial Surrealism** design principles with a **four-stage AI/CV pipeline**:

1. **Part 1: Three-Branch Parallel Perception Engine**
   - **Branch 1 (Vehicle ID)**: YOLOv8 vehicle detection + ResNet-50 feature extraction across Stanford Cars dataset.
   - **Branch 2 (Exterior Intelligence)**: Custom-trained YOLO11n-seg (`exterior_best.pt`) segmenting 23 exterior component classes and detecting aftermarket modifications.
   - **Branch 3 (Interior & Engine Cognition)**: Multimodal Groq Llama-Vision with automated primary-to-backup API failover.
   - **Consensus & Final Verdict**: Cross-branch verification identifying contradictions and synthesizing unified perceptions.

2. **Part 2: Intelligent Image Quality Assessment & Adaptive CV Enhancement**
   - Quantifies exposure (underexposure/overexposure), motion blur (Laplacian variance), sensor noise, and specular highlights.
   - Dynamically selects and tunes restoration filters (CLAHE lighting balance, Bilateral edge-preserving smoothing, Unsharp masking).
   - Preserves original imagery while generating branch-specific enhanced streams.

3. **Part 3: Comparative Confidence & Information Entropy Selection**
   - Runs parallel inference on both original and restored images.
   - Evaluates bounding-box stability, classification entropy, mask confidence, and multimodal visual reasoning.
   - Intelligently routes the optimal image version per branch rather than assuming enhancement is always superior.

4. **Part 4: Verified Technical Specifications & Market Valuation**
   - Maps predicted make, model, and year to authentic powertrain specifications (displacement, configuration, horsepower, torque, drivetrain, transmission).
   - Generates regional market valuations (INR, USD, EUR, GBP) taking into account mileage and depreciation.

---

## 🎨 Visual Identity: Editorial Surrealism

The interface rejects generic AI dashboard tropes in favor of an experimental editorial aesthetic inspired by classical sculpture, mechanical engineering diagrams, and alpine landscape photography:

- **Warm Chalk (`#F1EBDD`)**: 70% primary canvas background with subtle organic paper-grain texture.
- **Ink Black (`#20201E`)**: 15% structural typography and high-contrast editorial hierarchy.
- **Petrol Teal (`#176B70`)**: 10% primary signature accent for interactive elements and verified statuses.
- **Tangerine (`#F28B45`)**: 5% contrasting highlight accent for warnings, indicators, and kinetic details.
- **Warm Stone (`#DED5C5`)**: Secondary surfaces, frosted cards, and subtle dividers.

### Interactive Features
- **Custom Precision Cursor**: Smooth lerp tracking with interactive element expansion; disabled on touch devices.
- **Kinetic Multi-Layer Parallax**: Separated layers (etched mountains, rising sun, marble bust, mechanical visor, winding road) with an animated miniature vehicle traveling along the road during scroll.
- **Cinematic Wireframe Preloader**: Self-drawing SVG car outline sequence shown once per session.
- **Dual Experience**: Immersive Landing Manifesto separated from the dedicated Detection Laboratory Workspace (with zero fabricated results).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (Tested up to Python 3.14)
- Node.js 18+ and npm (for frontend development)
- Optional: Groq API Key (for Branch 3 visual reasoning)

### 1. Clone & Set Up Backend

```bash
git clone https://github.com/sidd-cr7/automotive-image-quality-analysis.git
cd automotive-image-quality-analysis

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

Copy `.env.example` to `.env` and set your Groq API keys if you wish to run Branch 3 visual reasoning:

```bash
cp .env.example .env
```

```ini
GROQ_API_KEY=your_primary_groq_api_key
GROQ_BACKUP_API_KEY=your_backup_groq_api_key
DEFAULT_MARKET_REGION=India
DEFAULT_CURRENCY=INR
```

### 3. Frontend Setup

The built frontend is pre-compiled inside `frontend/dist` and served automatically by FastAPI at `/`. To build from source:

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Run the Server

```bash
python run_server.py
# or: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`** to experience the full application.

---

## 🧪 Testing

The test suite covers the entire pipeline across 69 comprehensive test cases:

```bash
pytest
```

```
================= 69 passed in ~60s =================
- tests/test_api.py (Endpoints, health, predict)
- tests/test_components.py (Detector, classifier, feature extraction)
- tests/test_image_quality_engine.py (Part 2 Assessor + Adaptive Enhancer)
- tests/test_multimodal_architecture.py (Part 1 3-Branch Concurrent Pipeline)
- tests/test_prediction_comparator.py (Part 3 Dual-Image Route Comparator)
- tests/test_vehicle_intelligence.py (Part 4 Specs & Market Valuation)
```

---

## 📡 API Reference

### Health Check
`GET /health`
```json
{
  "status": "healthy",
  "device": "cpu",
  "models": {
    "detector": "YOLOv8n",
    "classifier": "ResNet50",
    "exterior_segmentation": "YOLO11n-seg",
    "image_quality_engine": "Active (Part 2 Assessor + Adaptive Enhancer)",
    "prediction_comparison_engine": "Active (Part 3 Comparator + Intelligent Selection)",
    "vehicle_intelligence_engine": "Active (Part 4 Specifications & Market Valuation Engine)"
  }
}
```

### Inference & Perception
`POST /predict` or `POST /analyze`
- **Body**: `multipart/form-data` with `file: UploadFile`
- **Parameters**:
  - `conf_threshold` (float, default: 0.25)
  - `apply_preprocessing` (bool, default: true)
  - `enable_multimodal` (bool, default: true)
  - `enable_quality_enhancement` (bool, default: true)
  - `enable_prediction_comparison` (bool, default: true)
  - `market_region` (str, default: "India")
  - `currency` (str, default: "INR")

---

## 📦 Model Weights Included

- **`models/resnet50_best.pth`**: Stanford Cars vehicle make/model/year classifier (~91.5 MB).
- **`models/detection/yolov8n.pt`**: YOLOv8 vehicle detector (~6.5 MB).
- **`exterior_best.pt`**: YOLO11n-seg trained on Carparts-Seg 23 exterior classes (~6.0 MB).
- **`models/classes.json`**: Stanford Cars 196 class definitions.

---

## 👥 Authors & Attribution

Developed for **CV Hackathon '26** by [@sidd-cr7](https://github.com/sidd-cr7).  
Architecture: Multi-Model Computer Vision & Editorial Surrealism.
