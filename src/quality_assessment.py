"""Intelligent Image Quality Assessment Engine for AutoVision AI.

Calculates measurable, interpretable quality metrics across:
- Exposure and Lighting (underexposure, overexposure, low contrast, uneven lighting)
- Blur and Sharpness (Laplacian variance, Tenengrad, directional motion blur)
- Noise and Compression (Immerkaer noise variance, JPEG blockiness, chromatic noise)
- Geometric Quality (tilt angle, aspect ratio abnormality, perspective skew)
- Framing and Composition (subject coverage, letterbox/pillarbox borders)
- Resolution and Color Balance (resolution score, color cast, haze)
"""

import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

logger = logging.getLogger("autovision.quality_assessment")


@dataclass
class QualityIndicators:
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    blur_score: float
    noise_estimate: float
    resolution_score: float
    exposure_score: float
    color_balance_score: float
    geometric_quality_indicators: Dict[str, Any]
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityAssessmentResult:
    overall_score: float
    detected_issues: List[str]
    quality_metrics: Dict[str, Any]
    is_acceptable: bool
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 1),
            "detected_issues": self.detected_issues,
            "quality_metrics": self.quality_metrics,
            "is_acceptable": self.is_acceptable,
            "recommendations": self.recommendations,
        }


class ImageQualityAssessor:
    """
    Evaluates image quality using classical Computer Vision and statistical signal processing.
    Produces interpretable numerical metrics and defect classifications without heuristic probabilities.
    """

    def __init__(
        self,
        target_resolution: Tuple[int, int] = (640, 640),
        dark_threshold: float = 60.0,
        overexposed_threshold: float = 195.0,
        low_contrast_threshold: float = 40.0,
        blur_laplacian_threshold: float = 100.0,
        noise_threshold: float = 5.0,
        max_tilt_threshold_deg: float = 1.5,
    ):
        self.target_resolution = target_resolution
        self.dark_threshold = dark_threshold
        self.overexposed_threshold = overexposed_threshold
        self.low_contrast_threshold = low_contrast_threshold
        self.blur_laplacian_threshold = blur_laplacian_threshold
        self.noise_threshold = noise_threshold
        self.max_tilt_threshold_deg = max_tilt_threshold_deg

    # -------------------------------------------------------------------------
    # A. Exposure & Lighting Assessment
    # -------------------------------------------------------------------------
    @staticmethod
    def assess_exposure(gray: np.ndarray) -> Dict[str, Any]:
        """Compute luminance distribution, dynamic range, and highlight/shadow clipping."""
        h, w = gray.shape[:2]
        total_pixels = h * w

        mean_luma = float(np.mean(gray))
        std_luma = float(np.std(gray))
        p1, p5, p50, p95, p99 = [float(v) for v in np.percentile(gray, [1, 5, 50, 95, 99])]
        dynamic_range = p99 - p1

        # Clipping ratios
        under_ratio = float(np.count_nonzero(gray < 25) / total_pixels)
        over_ratio = float(np.count_nonzero(gray > 235) / total_pixels)

        # Shannon Entropy
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist_prob = hist / total_pixels
        hist_prob = hist_prob[hist_prob > 0]
        entropy = float(-np.sum(hist_prob * np.log2(hist_prob)))

        # Sub-grid illumination uniformity (divide into 4x4 blocks)
        grid_means = []
        gh, gw = max(1, h // 4), max(1, w // 4)
        for i in range(4):
            for j in range(4):
                block = gray[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
                if block.size > 0:
                    grid_means.append(float(np.mean(block)))
        illum_uniformity = float(np.std(grid_means)) if grid_means else 0.0

        # Exposure score calculation (0 - 100 scale)
        exp_score = 100.0
        # Penalize mean luminance deviations from ideal midtone (~115 - 145)
        if mean_luma < 90:
            exp_score -= min(50.0, (90.0 - mean_luma) * 0.9)
        elif mean_luma > 165:
            exp_score -= min(50.0, (mean_luma - 165.0) * 0.8)

        # Penalize excessive clipping
        exp_score -= min(30.0, under_ratio * 100.0 * 0.6)
        exp_score -= min(30.0, over_ratio * 100.0 * 0.8)
        exp_score = max(0.0, min(100.0, exp_score))

        # Contrast score (std_luma ideal between 45 and 75)
        if std_luma >= 45.0:
            contrast_score = min(100.0, 70.0 + (std_luma - 45.0) * 0.8)
        else:
            contrast_score = max(0.0, (std_luma / 45.0) * 70.0)

        return {
            "mean_luminance": round(mean_luma, 2),
            "std_luminance": round(std_luma, 2),
            "dynamic_range": round(dynamic_range, 2),
            "underexposed_ratio": round(under_ratio, 4),
            "overexposed_ratio": round(over_ratio, 4),
            "entropy": round(entropy, 2),
            "illumination_variance": round(illum_uniformity, 2),
            "exposure_score": round(exp_score, 1),
            "contrast_score": round(contrast_score, 1),
        }

    # -------------------------------------------------------------------------
    # B. Blur & Sharpness Assessment
    # -------------------------------------------------------------------------
    @staticmethod
    def assess_blur_and_sharpness(gray: np.ndarray) -> Dict[str, Any]:
        """Compute Laplacian variance, Tenengrad gradient energy, and directional motion blur."""
        # 1. Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        # 2. Tenengrad Gradient Energy
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        tenengrad = float(np.mean(grad_mag**2))

        # 3. Directional Motion Blur Anisotropy
        mean_gx = float(np.mean(np.abs(sobelx)))
        mean_gy = float(np.mean(np.abs(sobely)))
        min_g = max(0.01, min(mean_gx, mean_gy))
        max_g = max(mean_gx, mean_gy)
        motion_anisotropy = float(max_g / min_g)

        # Logarithmic normalization of sharpness score to 0 - 100
        # Typical Laplacian variance: < 50 = very blurry, 100-300 = acceptable, > 800 = very sharp
        if blur_score <= 1.0:
            sharpness_score = 5.0
        else:
            sharpness_score = min(100.0, math.log10(blur_score + 1.0) * 32.0)
        sharpness_score = max(0.0, min(100.0, sharpness_score))

        return {
            "blur_score": round(blur_score, 2),
            "tenengrad_score": round(tenengrad, 2),
            "motion_anisotropy": round(motion_anisotropy, 2),
            "sharpness_score": round(sharpness_score, 1),
        }

    # -------------------------------------------------------------------------
    # C. Noise & Compression Artifact Assessment
    # -------------------------------------------------------------------------
    @staticmethod
    def assess_noise_and_compression(image_rgb: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Estimate Gaussian/sensor noise using Immerkaer's method and JPEG blockiness."""
        h, w = gray.shape[:2]

        # 1. Immerkaer Fast Noise Estimator (Laplacian-like mask)
        # N = [1 -2 1; -2 4 -2; 1 -2 1]
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ], dtype=np.float32)

        if h >= 8 and w >= 8:
            sigma_conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            noise_est = float(np.sum(np.abs(sigma_conv)) * math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)))
        else:
            noise_est = 0.0

        # Signal-to-Noise Ratio (dB)
        signal_power = float(np.mean(gray**2))
        noise_power = max(1e-4, noise_est**2)
        snr_db = float(10.0 * math.log10(max(1.0, signal_power / noise_power)))

        # 2. Chromatic Noise (Color channel standard deviations in YCrCb)
        ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        cr_blur = cv2.GaussianBlur(ycrcb[:, :, 1], (5, 5), 0)
        cb_blur = cv2.GaussianBlur(ycrcb[:, :, 2], (5, 5), 0)
        chroma_noise = float(np.mean(np.abs(ycrcb[:, :, 1].astype(float) - cr_blur)) +
                            np.mean(np.abs(ycrcb[:, :, 2].astype(float) - cb_blur)))

        # 3. JPEG Blockiness Metric (Discontinuity along 8x8 block boundaries)
        blockiness = 0.0
        if h >= 32 and w >= 32:
            diff_h = np.abs(gray[7:-1:8, :] - gray[8::8, :])
            diff_non_h = np.abs(gray[6:-2:8, :] - gray[7:-1:8, :])
            if diff_non_h.size > 0 and np.mean(diff_non_h) > 0:
                blockiness = float(np.mean(diff_h) / (np.mean(diff_non_h) + 1e-4))

        noise_score = max(0.0, min(100.0, 100.0 - (noise_est * 6.5) - max(0.0, (blockiness - 1.2) * 20.0)))

        return {
            "noise_estimate": round(noise_est, 2),
            "snr_db": round(snr_db, 2),
            "chroma_noise": round(chroma_noise, 2),
            "jpeg_blockiness": round(blockiness, 2),
            "noise_score": round(noise_score, 1),
        }

    # -------------------------------------------------------------------------
    # D. Geometric Distortion Assessment
    # -------------------------------------------------------------------------
    @staticmethod
    def assess_geometry(gray: np.ndarray) -> Dict[str, Any]:
        """Detect tilt/skew angle via Hough Lines and evaluate aspect ratio."""
        h, w = gray.shape[:2]
        aspect_ratio = float(w / h) if h > 0 else 1.0

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=max(30, min(h, w) // 8), maxLineGap=10)

        detected_tilt_deg = 0.0
        tilt_confidence = 0.0
        skew_score = 0.0

        if lines is not None and len(lines) >= 4:
            angles = []
            lengths = []
            # HoughLinesP returns (N,1,4) on some OpenCV builds and (N,4) on others; flatten to (N,4).
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4).astype(float):
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx**2 + dy**2)
                angle_deg = math.degrees(math.atan2(dy, dx))

                # Normalize near horizontal lines [-45, 45]
                if abs(angle_deg) <= 35.0:
                    angles.append(angle_deg)
                    lengths.append(length)

            if len(angles) >= 3:
                # Weighted median tilt
                sorted_indices = np.argsort(angles)
                median_idx = sorted_indices[len(sorted_indices) // 2]
                detected_tilt_deg = float(angles[median_idx])
                tilt_confidence = min(1.0, len(angles) / 15.0)

        # Aspect ratio abnormality check
        aspect_abnormal = aspect_ratio < 0.6 or aspect_ratio > 2.4

        return {
            "detected_tilt_deg": round(detected_tilt_deg, 2),
            "tilt_confidence": round(tilt_confidence, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "aspect_abnormal": aspect_abnormal,
            "perspective_skew_score": round(abs(detected_tilt_deg), 2),
        }

    # -------------------------------------------------------------------------
    # E. Framing & Composition Assessment
    # -------------------------------------------------------------------------
    @staticmethod
    def assess_framing(image_rgb: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Detect letterbox/pillarbox borders and estimate subject framing."""
        h, w = gray.shape[:2]

        # 1. Border detection (check outer 5% on each edge)
        top_border = int(np.std(gray[:max(1, int(h * 0.04)), :]) < 5.0 and np.mean(gray[:max(1, int(h * 0.04)), :]) in [0, 255])
        bottom_border = int(np.std(gray[-max(1, int(h * 0.04)):, :]) < 5.0 and np.mean(gray[-max(1, int(h * 0.04)):, :]) in [0, 255])
        left_border = int(np.std(gray[:, :max(1, int(w * 0.04))]) < 5.0 and np.mean(gray[:, :max(1, int(w * 0.04))]) in [0, 255])
        right_border = int(np.std(gray[:, -max(1, int(w * 0.04)):]) < 5.0 and np.mean(gray[:, -max(1, int(w * 0.04)):]) in [0, 255])

        letterbox_detected = bool(top_border or bottom_border or left_border or right_border)

        # 2. Saliency / Foreground coverage estimation via Otsu threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        active_pixels = np.count_nonzero(thresh)
        coverage_ratio = float(active_pixels / (h * w))

        # Framing score
        framing_score = 90.0
        if letterbox_detected:
            framing_score -= 25.0
        if coverage_ratio < 0.10:
            framing_score -= 20.0  # Vehicle potentially too small / distant

        return {
            "has_letterbox_borders": letterbox_detected,
            "top_border": bool(top_border),
            "bottom_border": bool(bottom_border),
            "left_border": bool(left_border),
            "right_border": bool(right_border),
            "foreground_coverage_ratio": round(coverage_ratio, 3),
            "framing_score": round(max(0.0, min(100.0, framing_score)), 1),
        }

    # -------------------------------------------------------------------------
    # F. Resolution & Color Quality Assessment
    # -------------------------------------------------------------------------
    @staticmethod
    def assess_resolution_and_color(image_rgb: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Assess resolution, white balance / color cast, and atmospheric haze."""
        h, w = image_rgb.shape[:2]
        megapixels = float((h * w) / 1_000_000.0)
        min_dim = min(h, w)

        # Resolution score (640x640 is 100 baseline)
        if min_dim >= 640:
            res_score = min(100.0, 90.0 + (min_dim - 640) * 0.02)
        else:
            res_score = max(10.0, (min_dim / 640.0) * 85.0)

        # Color cast via LAB color space (neutral a*=128, b*=128)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        mean_a = float(np.mean(lab[:, :, 1]))
        mean_b = float(np.mean(lab[:, :, 2]))
        # In OpenCV LAB, a and b range from 0 to 255 with 128 as neutral gray
        chromaticity_dist = math.sqrt((mean_a - 128.0)**2 + (mean_b - 128.0)**2)
        color_balance_score = max(0.0, min(100.0, 100.0 - chromaticity_dist * 2.2))

        # Haze estimation via Dark Channel Prior (DCP)
        # Minimum of RGB channels across image
        min_channel = np.min(image_rgb, axis=2)
        haze_score = float(np.mean(min_channel))  # Higher means hazier / milky

        return {
            "megapixels": round(megapixels, 3),
            "dimensions": {"width": w, "height": h},
            "min_dimension": min_dim,
            "resolution_score": round(res_score, 1),
            "color_cast_distance": round(chromaticity_dist, 2),
            "color_balance_score": round(color_balance_score, 1),
            "dark_channel_haze_score": round(haze_score, 2),
        }

    # -------------------------------------------------------------------------
    # Master Quality Evaluation
    # -------------------------------------------------------------------------
    def assess(self, image_rgb: np.ndarray) -> QualityAssessmentResult:
        """
        Execute full image quality assessment and diagnose all defects.

        Args:
            image_rgb: RGB image array (H, W, 3) uint8.

        Returns:
            QualityAssessmentResult with scores, detected issues, and metrics.
        """
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
            raise ValueError("Input image must be a 3D numpy array (H, W, 3)")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Run All Sub-Assessments
        exp_metrics = self.assess_exposure(gray)
        blur_metrics = self.assess_blur_and_sharpness(gray)
        noise_metrics = self.assess_noise_and_compression(image_rgb, gray)
        geom_metrics = self.assess_geometry(gray)
        framing_metrics = self.assess_framing(image_rgb, gray)
        res_color_metrics = self.assess_resolution_and_color(image_rgb, gray)

        # 2. Defect Classification Rules
        detected_issues: List[str] = []
        recommendations: List[str] = []

        # Exposure issues
        if exp_metrics["mean_luminance"] < self.dark_threshold:
            detected_issues.append("underexposed")
            recommendations.append("Apply adaptive gamma brightening and shadow illumination correction.")
        elif exp_metrics["mean_luminance"] > self.overexposed_threshold:
            detected_issues.append("overexposed")
            recommendations.append("Apply highlight compression and exposure normalization.")

        if exp_metrics["std_luminance"] < self.low_contrast_threshold:
            detected_issues.append("low_contrast")
            recommendations.append("Apply CLAHE contrast equalization.")

        if exp_metrics["illumination_variance"] > 35.0:
            detected_issues.append("uneven_illumination")
            recommendations.append("Apply background illumination normalization.")

        # Blur issues
        if blur_metrics["blur_score"] < self.blur_laplacian_threshold:
            detected_issues.append("blur")
            if blur_metrics["motion_anisotropy"] > 2.2:
                detected_issues.append("motion_blur")
                recommendations.append("Apply directional motion deblurring and unsharp masking.")
            else:
                detected_issues.append("defocus_blur")
                recommendations.append("Apply edge-preserving unsharp masking.")

        # Noise issues
        if noise_metrics["noise_estimate"] > self.noise_threshold:
            detected_issues.append("sensor_noise")
            recommendations.append("Apply edge-preserving Non-Local Means / Bilateral denoising.")

        if noise_metrics["jpeg_blockiness"] > 1.35:
            detected_issues.append("jpeg_compression_artifacts")
            recommendations.append("Apply JPEG deblocking and artifact suppression filter.")

        # Geometric issues
        if abs(geom_metrics["detected_tilt_deg"]) >= self.max_tilt_threshold_deg and geom_metrics["tilt_confidence"] >= 0.35:
            detected_issues.append("tilted_image")
            recommendations.append(f"Apply {geom_metrics['detected_tilt_deg']:.1f}° rotation correction.")

        if geom_metrics["aspect_abnormal"]:
            detected_issues.append("unusual_aspect_ratio")

        # Framing issues
        if framing_metrics["has_letterbox_borders"]:
            detected_issues.append("letterbox_borders")
            recommendations.append("Trim extraneous black/white letterbox borders.")

        # Resolution & Color issues
        if res_color_metrics["min_dimension"] < 480:
            detected_issues.append("low_resolution")
            recommendations.append("Apply edge-preserving super-resolution interpolation.")

        if res_color_metrics["color_cast_distance"] > 14.0:
            detected_issues.append("color_cast")
            recommendations.append("Apply Gray World white balance correction.")

        if res_color_metrics["dark_channel_haze_score"] > 65.0:
            detected_issues.append("haze_or_fog")
            recommendations.append("Apply Dark Channel Prior contrast dehazing.")

        # 3. Overall Image Quality Composite Score (Weighted 0 - 100)
        # Weights: Exposure (25%), Sharpness (25%), Noise (15%), Contrast (15%), Color Balance (10%), Resolution (10%)
        overall = (
            exp_metrics["exposure_score"] * 0.25 +
            blur_metrics["sharpness_score"] * 0.25 +
            noise_metrics["noise_score"] * 0.15 +
            exp_metrics["contrast_score"] * 0.15 +
            res_color_metrics["color_balance_score"] * 0.10 +
            res_color_metrics["resolution_score"] * 0.10
        )
        overall = round(max(0.0, min(100.0, overall)), 1)
        is_acceptable = (overall >= 45.0) and ("blur" not in detected_issues or blur_metrics["blur_score"] >= 40.0)

        # Consolidate all metrics
        quality_metrics = {
            "brightness_score": exp_metrics["mean_luminance"],
            "contrast_score": exp_metrics["contrast_score"],
            "sharpness_score": blur_metrics["sharpness_score"],
            "blur_score": blur_metrics["blur_score"],
            "noise_estimate": noise_metrics["noise_estimate"],
            "snr_db": noise_metrics["snr_db"],
            "resolution_score": res_color_metrics["resolution_score"],
            "exposure_score": exp_metrics["exposure_score"],
            "color_balance_score": res_color_metrics["color_balance_score"],
            "geometric_quality_indicators": geom_metrics,
            "framing_indicators": framing_metrics,
            "overall_score": overall,
        }

        return QualityAssessmentResult(
            overall_score=overall,
            detected_issues=detected_issues,
            quality_metrics=quality_metrics,
            is_acceptable=is_acceptable,
            recommendations=recommendations,
        )
