"""Adaptive Preprocessing and Image Enhancement module with Part 2 Quality Engine integration."""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2

# Import upgraded Part 2 Quality Assessment & Adaptive Enhancement Engines
from .quality_assessment import ImageQualityAssessor, QualityAssessmentResult, QualityIndicators
from .adaptive_enhancer import AdaptiveEnhancer, AdaptiveEnhancementResult, EnhancementRecord


@dataclass
class QualityMetrics:
    brightness: float
    contrast: float
    blur_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "blur_score": round(self.blur_score, 2),
        }


@dataclass
class PreprocessingReport:
    applied_transforms: List[str]
    processing_time_ms: float
    metrics_before: QualityMetrics
    metrics_after: QualityMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied_transforms": self.applied_transforms,
            "processing_time_ms": round(self.processing_time_ms, 3),
            "metrics_before": self.metrics_before.to_dict(),
            "metrics_after": self.metrics_after.to_dict(),
        }


class AdaptivePreprocessor:
    """Adaptive image enhancement based on classical quality metrics with backward compatibility."""

    def __init__(
        self,
        dark_threshold: float = 65.0,
        low_contrast_threshold: float = 45.0,
        blur_threshold: float = 120.0,
    ):
        self.dark_threshold = dark_threshold
        self.low_contrast_threshold = low_contrast_threshold
        self.blur_threshold = blur_threshold
        self.enhancer = AdaptiveEnhancer()
        self.assessor = ImageQualityAssessor()

    @staticmethod
    def compute_metrics(image_rgb: np.ndarray) -> QualityMetrics:
        """Compute brightness, contrast, and Laplacian blur score."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return QualityMetrics(
            brightness=brightness,
            contrast=contrast,
            blur_score=blur_score,
        )

    @staticmethod
    def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE to L-channel in LAB color space."""
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def apply_gamma(image_rgb: np.ndarray, gamma: float = 1.3) -> np.ndarray:
        """Apply gamma correction for brightening/darkening."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image_rgb, table)

    @staticmethod
    def apply_bilateral_filter(image_rgb: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
        """Apply bilateral filtering for edge-preserving denoising."""
        return cv2.bilateralFilter(image_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    @staticmethod
    def apply_sharpening(image_rgb: np.ndarray) -> np.ndarray:
        """Apply subtle unsharp masking / sharpening filter."""
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(image_rgb, -1, kernel)

    def enhance(self, image_rgb: np.ndarray, force_transforms: Optional[List[str]] = None) -> Tuple[np.ndarray, PreprocessingReport]:
        """
        Adaptively enhance input vehicle crop or image.

        Args:
            image_rgb: Input RGB image (H, W, 3).
            force_transforms: Optional explicit list of transforms to apply.

        Returns:
            Tuple of (enhanced_rgb, PreprocessingReport).
        """
        start_time = time.perf_counter()
        metrics_before = self.compute_metrics(image_rgb)
        enhanced = image_rgb.copy()
        applied: List[str] = []

        if force_transforms is not None:
            for t in force_transforms:
                name = t.lower()
                if "clahe" in name:
                    enhanced = self.apply_clahe(enhanced)
                    applied.append("CLAHE")
                elif "gamma" in name:
                    enhanced = self.apply_gamma(enhanced, gamma=1.3)
                    applied.append("Gamma Correction (Brighten)")
                elif "bilateral" in name:
                    enhanced = self.apply_bilateral_filter(enhanced)
                    applied.append("Bilateral Filter")
                elif "sharpen" in name:
                    enhanced = self.apply_sharpening(enhanced)
                    applied.append("Sharpening Filter")
        else:
            # Adaptive decision rules based on quality metrics
            if metrics_before.brightness < self.dark_threshold:
                enhanced = self.apply_gamma(enhanced, gamma=1.35)
                applied.append("Gamma Correction (Brighten)")

            if metrics_before.contrast < self.low_contrast_threshold:
                enhanced = self.apply_clahe(enhanced)
                applied.append("CLAHE")

            if metrics_before.blur_score < self.blur_threshold and "Sharpening Filter" not in applied:
                enhanced = self.apply_sharpening(enhanced)
                applied.append("Sharpening Filter")

            if len(applied) == 0:
                applied.append("Original Retained")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        metrics_after = self.compute_metrics(enhanced)

        report = PreprocessingReport(
            applied_transforms=applied,
            processing_time_ms=elapsed_ms,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )
        return enhanced, report
