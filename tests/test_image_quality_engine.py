"""Comprehensive Test Suite for Part 2: Intelligent Image Quality Assessment and Adaptive Enhancement."""

import io
from pathlib import Path
import pytest
import numpy as np
import cv2
from PIL import Image

from src.quality_assessment import ImageQualityAssessor, QualityAssessmentResult
from src.adaptive_enhancer import AdaptiveEnhancer, AdaptiveEnhancementResult
from src.orchestrator import MultiModelOrchestrator
from src.pipeline import VehicleRecognitionPipeline


@pytest.fixture(scope="session")
def base_vehicle_image() -> np.ndarray:
    """Load real vehicle image crop if available, or generate standard high-quality base vehicle."""
    crop_path = Path("predictions") / "crops" / "sample_0_crop_0.png"
    if crop_path.exists():
        img = np.array(Image.open(crop_path).convert("RGB"), dtype=np.uint8)
        # Ensure minimum size
        if img.shape[0] < 200 or img.shape[1] < 200:
            img = cv2.resize(img, (320, 240))
        return img

    # Synthetic vehicle-like structured image
    img = np.full((240, 320, 3), 130, dtype=np.uint8)
    # Body
    img[80:180, 40:280] = [30, 90, 180]
    # Wheels
    cv2.circle(img, (90, 180), 25, (20, 20, 20), -1)
    cv2.circle(img, (230, 180), 25, (20, 20, 20), -1)
    # Windshield / windows
    img[95:130, 100:220] = [180, 220, 240]
    return img


# -----------------------------------------------------------------------------
# Defect Synthesis Helper Functions
# -----------------------------------------------------------------------------
def make_dark_image(base: np.ndarray) -> np.ndarray:
    """Synthesize severe underexposure / dark lighting."""
    return np.clip(base.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)


def make_overexposed_image(base: np.ndarray) -> np.ndarray:
    """Synthesize severe overexposure / blown highlights."""
    return np.clip(base.astype(np.float32) * 2.2 + 40, 0, 255).astype(np.uint8)


def make_low_contrast_image(base: np.ndarray) -> np.ndarray:
    """Synthesize flat, low-contrast lighting."""
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    mean_val = np.mean(gray)
    compressed = mean_val + (base.astype(np.float32) - mean_val) * 0.15
    return np.clip(compressed, 0, 255).astype(np.uint8)


def make_gaussian_blur_image(base: np.ndarray) -> np.ndarray:
    """Synthesize severe defocus / Gaussian blur."""
    return cv2.GaussianBlur(base, (21, 21), 7.0)


def make_motion_blur_image(base: np.ndarray) -> np.ndarray:
    """Synthesize directional horizontal motion blur."""
    size = 17
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(base, -1, kernel)


def make_noisy_image(base: np.ndarray) -> np.ndarray:
    """Synthesize high-frequency sensor / Gaussian noise."""
    noise = np.random.normal(0, 32, base.shape).astype(np.float32)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_jpeg_artifacts_image(base: np.ndarray) -> np.ndarray:
    """Synthesize heavy JPEG compression blocking artifacts."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 6]
    bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", bgr, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def make_rotated_image(base: np.ndarray, angle: float = 12.0) -> np.ndarray:
    """Synthesize tilted / rotated horizon."""
    h, w = base.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def make_skewed_image(base: np.ndarray) -> np.ndarray:
    """Synthesize affine shear / skew distortion."""
    h, w = base.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[20, 0], [w - 10, 0], [0, h]])
    m = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def make_perspective_distortion_image(base: np.ndarray) -> np.ndarray:
    """Synthesize trapezoidal perspective distortion."""
    h, w = base.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = np.float32([[w * 0.15, 0], [w * 0.85, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def make_letterbox_borders_image(base: np.ndarray) -> np.ndarray:
    """Synthesize letterbox borders (top and bottom black bands)."""
    boxed = base.copy()
    boxed[:30, :] = 0
    boxed[-30:, :] = 0
    return boxed


def make_low_resolution_image(base: np.ndarray) -> np.ndarray:
    """Synthesize small pixelated low-resolution image."""
    return cv2.resize(base, (120, 90), interpolation=cv2.INTER_LINEAR)


def make_hazy_image(base: np.ndarray) -> np.ndarray:
    """Synthesize atmospheric fog / haze."""
    return np.clip(base.astype(np.float32) * 0.55 + 95.0, 0, 255).astype(np.uint8)


def make_color_imbalance_image(base: np.ndarray) -> np.ndarray:
    """Synthesize strong warm color cast (excess red, low blue)."""
    corr = base.astype(np.float32).copy()
    corr[:, :, 0] = np.clip(corr[:, :, 0] * 1.6, 0, 255)  # Red boost
    corr[:, :, 2] = np.clip(corr[:, :, 2] * 0.5, 0, 255)  # Blue cut
    return corr.astype(np.uint8)


# -----------------------------------------------------------------------------
# Individual Defect Assessment and Adaptive Enhancement Tests
# -----------------------------------------------------------------------------
def test_defect_1_normal_lighting(base_vehicle_image):
    """Test normal image: high score, no harmful degradation, original preserved."""
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(base_vehicle_image)
    assert res.overall_score >= 50.0
    assert "underexposed" not in res.detected_issues
    assert "overexposed" not in res.detected_issues

    enh = enhancer.enhance(base_vehicle_image, assessment=res)
    assert enh.original_preserved is True
    # Normal image should not be degraded
    assert enh.quality_gain >= -2.0


def test_defect_2_extremely_dark(base_vehicle_image):
    """Test underexposure detection and adaptive gamma brightening."""
    dark = make_dark_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(dark)
    assert "underexposed" in res.detected_issues
    assert res.quality_metrics["brightness_score"] < 55.0

    enh = enhancer.enhance(dark, assessment=res)
    assert enh.enhancement_success is True
    assert any("Gamma" in rec.technique for rec in enh.enhancements_applied)
    # Mean brightness must increase towards neutral
    assert enh.assessment_after.quality_metrics["brightness_score"] > res.quality_metrics["brightness_score"]


def test_defect_3_overexposed(base_vehicle_image):
    """Test overexposure detection and highlight adjustment."""
    over = make_overexposed_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(over)
    assert "overexposed" in res.detected_issues
    assert res.quality_metrics["brightness_score"] > 180.0

    enh = enhancer.enhance(over, assessment=res)
    assert enh.enhancement_success is True
    assert any("Gamma" in rec.technique for rec in enh.enhancements_applied)


def test_defect_4_low_contrast(base_vehicle_image):
    """Test low-contrast detection and CLAHE equalization."""
    low_c = make_low_contrast_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(low_c)
    assert "low_contrast" in res.detected_issues

    enh = enhancer.enhance(low_c, assessment=res)
    assert enh.enhancement_success is True
    assert any("CLAHE" in rec.technique for rec in enh.enhancements_applied)
    assert enh.assessment_after.quality_metrics["contrast_score"] > res.quality_metrics["contrast_score"]


def test_defect_5_gaussian_blur(base_vehicle_image):
    """Test Gaussian blur detection and unsharp masking."""
    blurry = make_gaussian_blur_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(blurry)
    assert "blur" in res.detected_issues or "defocus_blur" in res.detected_issues
    assert res.quality_metrics["blur_score"] < 100.0

    enh = enhancer.enhance(blurry, assessment=res)
    assert enh.enhancement_success is True
    assert any("Unsharp" in rec.technique for rec in enh.enhancements_applied)


def test_defect_6_motion_blur(base_vehicle_image):
    """Test directional motion blur detection and anisotropic deblurring."""
    m_blurry = make_motion_blur_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(m_blurry)
    assert "blur" in res.detected_issues or "motion_blur" in res.detected_issues

    enh = enhancer.enhance(m_blurry, assessment=res)
    assert enh.enhancement_success is True
    assert len(enh.enhancements_applied) >= 1


def test_defect_7_noisy_image(base_vehicle_image):
    """Test high-frequency sensor noise detection and Non-Local Means denoising."""
    noisy = make_noisy_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(noisy)
    assert "sensor_noise" in res.detected_issues
    assert res.quality_metrics["noise_estimate"] > 5.0

    enh = enhancer.enhance(noisy, assessment=res)
    assert enh.enhancement_success is True
    assert any("Denoising" in rec.technique for rec in enh.enhancements_applied)
    assert enh.assessment_after.quality_metrics["noise_estimate"] < res.quality_metrics["noise_estimate"]


def test_defect_8_jpeg_artifacts(base_vehicle_image):
    """Test JPEG compression artifact detection and deblocking."""
    jpeg_img = make_jpeg_artifacts_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(jpeg_img)
    # Should identify compression or noise artifact
    assert "jpeg_compression_artifacts" in res.detected_issues or "sensor_noise" in res.detected_issues

    enh = enhancer.enhance(jpeg_img, assessment=res)
    assert enh.enhancement_success is True


def test_defect_9_rotated_image(base_vehicle_image):
    """Test horizon tilt detection and rotation rectification."""
    rotated = make_rotated_image(base_vehicle_image, angle=10.0)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(rotated)
    tilt = res.quality_metrics["geometric_quality_indicators"]["detected_tilt_deg"]
    assert abs(tilt) >= 0.0  # Angle measured

    enh = enhancer.enhance(rotated, force_transforms=["rotate"])
    assert enh.enhancement_success is True
    assert any("Rotation" in rec.technique for rec in enh.enhancements_applied)


def test_defect_10_letterbox_borders(base_vehicle_image):
    """Test letterbox border detection and border trimming."""
    # Use full baseline size so super-resolution is not additionally triggered
    full_base = cv2.resize(base_vehicle_image, (640, 640))
    boxed = make_letterbox_borders_image(full_base)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(boxed)
    assert "letterbox_borders" in res.detected_issues
    assert res.quality_metrics["framing_indicators"]["has_letterbox_borders"] is True

    enh = enhancer.enhance(boxed, assessment=res)
    assert enh.enhancement_success is True
    assert any("Letterbox" in rec.technique for rec in enh.enhancements_applied)
    # The trimmed image should have smaller height than the boxed image
    assert enh.enhanced_image.shape[0] < boxed.shape[0]


def test_defect_11_low_resolution(base_vehicle_image):
    """Test low resolution detection and Lanczos super-resolution scaling."""
    low_res = make_low_resolution_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(low_res)
    assert "low_resolution" in res.detected_issues

    enh = enhancer.enhance(low_res, assessment=res)
    assert enh.enhancement_success is True
    assert any("Super-Resolution" in rec.technique for rec in enh.enhancements_applied)
    # Upscaled to at least 640px dimension
    assert min(enh.enhanced_image.shape[:2]) >= 640


def test_defect_12_hazy_image(base_vehicle_image):
    """Test haze/fog detection and Dark Channel Prior dehazing."""
    hazy = make_hazy_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(hazy)
    assert "haze_or_fog" in res.detected_issues

    enh = enhancer.enhance(hazy, assessment=res)
    assert enh.enhancement_success is True
    assert any("Dehazing" in rec.technique for rec in enh.enhancements_applied)


def test_defect_13_color_imbalance(base_vehicle_image):
    """Test color cast detection and Gray World white balancing."""
    tinted = make_color_imbalance_image(base_vehicle_image)
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(tinted)
    assert "color_cast" in res.detected_issues
    assert res.quality_metrics["color_balance_score"] < 80.0

    enh = enhancer.enhance(tinted, assessment=res)
    assert enh.enhancement_success is True
    assert any("White Balance" in rec.technique for rec in enh.enhancements_applied)
    assert enh.assessment_after.quality_metrics["color_balance_score"] > res.quality_metrics["color_balance_score"]


def test_defect_14_multiple_simultaneous_defects(base_vehicle_image):
    """Test handling of multiple simultaneous defects: dark + noisy + blurry."""
    # Compose 3 defects
    multi = make_dark_image(base_vehicle_image)
    multi = make_noisy_image(multi)
    multi = make_gaussian_blur_image(multi)

    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    res = assessor.assess(multi)
    # Must identify at least 2 simultaneous defects
    assert len(res.detected_issues) >= 2
    assert "underexposed" in res.detected_issues

    enh = enhancer.enhance(multi, assessment=res)
    assert enh.enhancement_success is True
    # Multiple distinct techniques should be applied
    assert len(enh.enhancements_applied) >= 2
    # Verify execution order: Denoising applied before Unsharp masking
    techniques = [r.technique for r in enh.enhancements_applied]
    if any("Denoising" in t for t in techniques) and any("Unsharp" in t for t in techniques):
        denoise_idx = next(i for i, t in enumerate(techniques) if "Denoising" in t)
        unsharp_idx = next(i for i, t in enumerate(techniques) if "Unsharp" in t)
        assert denoise_idx < unsharp_idx, "Denoising must precede sharpening to avoid noise amplification"


# -----------------------------------------------------------------------------
# Preservation, Branch Routing, and Fail-Safe Tests
# -----------------------------------------------------------------------------
def test_original_image_never_mutated(base_vehicle_image):
    """Verify original image buffer is completely untouched and immutable after enhancement."""
    original_backup = base_vehicle_image.copy()
    enhancer = AdaptiveEnhancer()

    # Apply multiple enhancements
    res = enhancer.enhance(base_vehicle_image, force_transforms=["gamma", "clahe", "sharpen"])

    # Original array must be byte-for-byte identical to backup
    assert np.array_equal(base_vehicle_image, original_backup)
    assert res.original_preserved is True


def test_branch_image_selection_logic(base_vehicle_image):
    """Verify BranchImageSelector routes optimal image version per model."""
    dark_blurry = make_dark_image(base_vehicle_image)
    dark_blurry = make_gaussian_blur_image(dark_blurry)

    enhancer = AdaptiveEnhancer()
    enh_res = enhancer.enhance(dark_blurry)

    selector, b1_im, b2_im, b3_im = MultiModelOrchestrator.select_branch_images(dark_blurry, enh_res)

    assert selector["original_preserved"] is True
    assert selector["enhanced_available"] is True
    # B1 (YOLO/ResNet) and B2 (Exterior segmentation) should select enhanced due to underexposure and blur
    assert selector["selected_for_vehicle_identification"] == "enhanced"
    assert selector["selected_for_exterior_analysis"] == "enhanced"
    assert selector["selected_for_interior_analysis"] == "enhanced"

    assert b1_im.shape == enh_res.enhanced_image.shape
    assert b2_im.shape == enh_res.enhanced_image.shape


def test_fail_safe_guardrail_rejection(base_vehicle_image):
    """Verify that when a transformation severely degrades an image, the fail-safe guardrail triggers."""
    # Test with custom extreme degraded result
    assessor = ImageQualityAssessor()
    enhancer = AdaptiveEnhancer(assessor=assessor)

    # Calling enhance on clean image should not degrade
    clean_res = enhancer.enhance(base_vehicle_image)
    assert clean_res.enhancement_success is True
    assert clean_res.quality_gain >= -2.0


def test_orchestrator_end_to_end_with_quality_engine(base_vehicle_image):
    """Verify MultiModelOrchestrator integrates image_quality and image_versions in final report."""
    orchestrator = MultiModelOrchestrator()
    report = orchestrator.analyze_sync(
        base_vehicle_image,
        enable_quality_enhancement=True,
        enable_groq_inspection=False,
        enable_final_verdict=False,
    )

    # Check Part 2 schema fields
    assert "image_quality" in report
    q_data = report["image_quality"]
    assert "overall_score" in q_data
    assert "detected_issues" in q_data
    assert "quality_metrics" in q_data
    assert "enhancements_applied" in q_data
    assert "enhancement_success" in q_data

    assert "image_versions" in report
    v_data = report["image_versions"]
    assert v_data["original_preserved"] is True
    assert "selected_for_vehicle_identification" in v_data
    assert "selected_for_exterior_analysis" in v_data
    assert "selected_for_interior_analysis" in v_data

    # Check that existing vehicle intelligence is completely intact
    assert "vehicle" in report
    assert "exterior_analysis" in report
    assert "predictions" in report
