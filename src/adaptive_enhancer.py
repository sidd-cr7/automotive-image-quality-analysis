"""Adaptive Computer Vision Enhancement Engine for AutoVision AI.

Selects, sequences, and applies targeted computer vision restorations based on
diagnosed quality defects while strictly guarding against destructive modifications:
1. Framing & Borders (Letterbox/Pillarbox removal)
2. Geometric Correction (Conservative tilt rectification)
3. Noise & Artifact Suppression (NL-Means, Bilateral, Deblocking)
4. Dehazing & Color Balance (Dark Channel Prior, Gray World White Balance)
5. Illumination & Exposure (Adaptive Gamma, CLAHE, Shadow-Highlight balance)
6. Sharpness & Detail Restoration (Unsharp Masking, Laplacian sharpening)
7. Edge-Preserving Super-Resolution (Lanczos-4 upscaling for low-res inputs)
"""

import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from .quality_assessment import ImageQualityAssessor, QualityAssessmentResult

logger = logging.getLogger("autovision.adaptive_enhancer")


@dataclass
class EnhancementRecord:
    technique: str
    target_defect: str
    rationale: str
    parameters: Dict[str, Any]
    execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technique": self.technique,
            "target_defect": self.target_defect,
            "rationale": self.rationale,
            "parameters": self.parameters,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class AdaptiveEnhancementResult:
    enhanced_image: np.ndarray
    original_preserved: bool
    enhancements_applied: List[EnhancementRecord]
    enhancement_success: bool
    assessment_before: QualityAssessmentResult
    assessment_after: QualityAssessmentResult
    quality_gain: float  # delta overall_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_preserved": self.original_preserved,
            "enhancement_success": self.enhancement_success,
            "enhancements_count": len(self.enhancements_applied),
            "enhancements_applied": [r.to_dict() for r in self.enhancements_applied],
            "quality_score_before": self.assessment_before.overall_score,
            "quality_score_after": self.assessment_after.overall_score,
            "quality_gain": round(self.quality_gain, 1),
        }


class AdaptiveEnhancer:
    """
    Intelligent CV enhancement engine that applies strictly tailored corrections
    based on the diagnosed quality issues without over-processing healthy images.
    """

    def __init__(self, assessor: Optional[ImageQualityAssessor] = None):
        self.assessor = assessor or ImageQualityAssessor()

    # -------------------------------------------------------------------------
    # 1. Exposure and Illumination Techniques
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_adaptive_gamma(image_rgb: np.ndarray, target_midpoint: float = 125.0) -> Tuple[np.ndarray, float]:
        """Compute optimal gamma power curve to normalize underexposed or overexposed midtones."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        current_mean = max(1.0, float(np.mean(gray)))

        # Solve for gamma: (current_mean / 255) ^ (1 / gamma) = target_midpoint / 255
        ratio_in = current_mean / 255.0
        ratio_out = target_midpoint / 255.0
        try:
            gamma = float(np.log(ratio_in) / np.log(ratio_out))
        except ZeroDivisionError:
            gamma = 1.0

        # Clamp gamma to safe, natural bounds [0.55, 2.2]
        gamma = max(0.55, min(2.2, gamma))

        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        corrected = cv2.LUT(image_rgb, lut)
        return corrected, gamma

    @staticmethod
    def apply_clahe_lab(image_rgb: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization to L-channel in LAB space."""
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    @staticmethod
    def apply_illumination_correction(image_rgb: np.ndarray, blur_sigma: float = 40.0) -> np.ndarray:
        """Correct harsh shadows or uneven lighting by background illumination field estimation."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Low-frequency illumination background field
        background = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
        mean_bg = np.mean(background)
        illum_ratio = mean_bg / (background + 1e-4)
        illum_ratio = np.clip(illum_ratio, 0.7, 1.4)

        # Apply modulation to luminance in YCrCb
        ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0] * illum_ratio, 0, 255)
        return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    # -------------------------------------------------------------------------
    # 2. Blur and Sharpness Techniques
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_unsharp_mask(
        image_rgb: np.ndarray,
        radius: int = 3,
        amount: float = 1.5,
        threshold: int = 3,
    ) -> np.ndarray:
        """
        Edge-directed Unsharp Masking with thresholding to prevent noise amplification.
        High-pass details are added back to the original image.
        """
        blurred = cv2.GaussianBlur(image_rgb, (0, 0), radius)
        diff = image_rgb.astype(np.int32) - blurred.astype(np.int32)

        # Apply threshold to only amplify true edges, ignoring low-magnitude noise
        mask = np.abs(diff) >= threshold
        sharpened = image_rgb.astype(np.float32) + (diff * amount * mask)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_motion_deblur(image_rgb: np.ndarray, angle_deg: float = 0.0, length: int = 5) -> np.ndarray:
        """Apply inverse motion filter along dominant motion vector."""
        if length <= 1:
            return image_rgb

        # Create motion blur kernel
        kernel = np.zeros((length, length), dtype=np.float32)
        center = length // 2
        rad = np.deg2rad(angle_deg)
        dx = np.cos(rad)
        dy = np.sin(rad)

        for i in range(-center, center + 1):
            x = int(round(center + i * dx))
            y = int(round(center + i * dy))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0

        k_sum = np.sum(kernel)
        if k_sum > 0:
            kernel /= k_sum

        # Regularized Wiener deconvolution in frequency domain
        deblurred_channels = []
        for c in range(3):
            ch = image_rgb[:, :, c].astype(np.float32)
            # Simple spatial unsharp approximation tailored to motion vector
            blurred = cv2.filter2D(ch, -1, kernel)
            deblurred = ch + 1.2 * (ch - blurred)
            deblurred_channels.append(np.clip(deblurred, 0, 255).astype(np.uint8))

        return cv2.merge(deblurred_channels)

    # -------------------------------------------------------------------------
    # 3. Noise and Compression Techniques
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_nlm_denoising(image_rgb: np.ndarray, strength: float = 5.0) -> np.ndarray:
        """Non-Local Means Denoising for sensor and color noise."""
        h = max(2, int(round(strength)))
        return cv2.fastNlMeansDenoisingColored(
            image_rgb,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7,
            searchWindowSize=15,
        )

    @staticmethod
    def apply_bilateral_filter(image_rgb: np.ndarray, d: int = 5, sigma_color: float = 40.0, sigma_space: float = 40.0) -> np.ndarray:
        """Bilateral edge-preserving smoothing filter."""
        return cv2.bilateralFilter(image_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    @staticmethod
    def apply_jpeg_deblocking(image_rgb: np.ndarray) -> np.ndarray:
        """Reduce 8x8 block boundary compression artifacts while preserving edges."""
        # Guided/bilateral pass specifically targeting block discontinuities
        return cv2.edgePreservingFilter(image_rgb, flags=1, sigma_s=25, sigma_r=0.25)

    # -------------------------------------------------------------------------
    # 4. Geometric Distortion Correction
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_rotation_correction(image_rgb: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate image by detected tilt angle to level horizon/ground plane.
        Clamps to safe range [-25, 25] and expands canvas to avoid cropping.
        """
        if abs(angle_deg) < 0.5 or abs(angle_deg) > 25.0:
            return image_rgb

        h, w = image_rgb.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # Compute bounding dimensions of rotated image
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust transformation matrix to accommodate new center
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]

        rotated = cv2.warpAffine(
            image_rgb,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    # -------------------------------------------------------------------------
    # 5. Framing and Composition Techniques
    # -------------------------------------------------------------------------
    @staticmethod
    def remove_letterbox_borders(image_rgb: np.ndarray) -> np.ndarray:
        """Detect and crop extraneous solid black or white letterbox/pillarbox borders."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        top, bottom, left, right = 0, h, 0, w

        # Top border
        for r in range(min(h // 4, 150)):
            row = gray[r, :]
            if np.std(row) < 3.0 and (np.mean(row) < 15.0 or np.mean(row) > 240.0):
                top = r + 1
            else:
                break

        # Bottom border
        for r in range(h - 1, max(h - 1 - h // 4, h - 150), -1):
            row = gray[r, :]
            if np.std(row) < 3.0 and (np.mean(row) < 15.0 or np.mean(row) > 240.0):
                bottom = r
            else:
                break

        # Left border
        for c in range(min(w // 4, 150)):
            col = gray[:, c]
            if np.std(col) < 3.0 and (np.mean(col) < 15.0 or np.mean(col) > 240.0):
                left = c + 1
            else:
                break

        # Right border
        for c in range(w - 1, max(w - 1 - w // 4, w - 150), -1):
            col = gray[:, c]
            if np.std(col) < 3.0 and (np.mean(col) < 15.0 or np.mean(col) > 240.0):
                right = c
            else:
                break

        if (bottom - top >= 64) and (right - left >= 64):
            return image_rgb[top:bottom, left:right].copy()
        return image_rgb

    # -------------------------------------------------------------------------
    # 6. Resolution and Color Techniques
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_super_resolution_interp(image_rgb: np.ndarray, min_target: int = 640) -> np.ndarray:
        """Upscale low-resolution image using Lanczos-4 filter with detail sharpening."""
        h, w = image_rgb.shape[:2]
        min_dim = min(h, w)
        if min_dim >= min_target:
            return image_rgb

        scale = float(min_target / min_dim)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        upscaled = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # Subtle detail boost after upscaling
        return AdaptiveEnhancer.apply_unsharp_mask(upscaled, radius=2, amount=1.2, threshold=4)

    @staticmethod
    def apply_white_balance_gray_world(image_rgb: np.ndarray) -> np.ndarray:
        """Gray World white balance normalization to eliminate unnatural color casts."""
        r, g, b = cv2.split(image_rgb.astype(np.float32))
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        mean_gray = (mean_r + mean_g + mean_b) / 3.0
        if mean_r > 0 and mean_g > 0 and mean_b > 0:
            scale_r = mean_gray / mean_r
            scale_g = mean_gray / mean_g
            scale_b = mean_gray / mean_b

            r_corr = np.clip(r * scale_r, 0, 255)
            g_corr = np.clip(g * scale_g, 0, 255)
            b_corr = np.clip(b * scale_b, 0, 255)
            return cv2.merge([r_corr, g_corr, b_corr]).astype(np.uint8)

        return image_rgb

    @staticmethod
    def apply_dehaze_dcp(image_rgb: np.ndarray, omega: float = 0.85) -> np.ndarray:
        """Remove atmospheric haze and fog using Dark Channel Prior."""
        img_norm = image_rgb.astype(np.float32) / 255.0
        min_channel = np.min(img_norm, axis=2)

        # Dark channel in 15x15 patch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(min_channel, kernel)

        # Atmospheric light A: top 0.1% brightest pixels in dark channel
        num_pixels = dark_channel.size
        top_k = max(1, int(num_pixels * 0.001))
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -top_k)[-top_k:]
        A = np.max(img_norm.reshape(-1, 3)[indices], axis=0)
        A = np.clip(A, 0.7, 1.0)

        # Transmission map
        norm_img_a = img_norm / A
        dark_norm = cv2.erode(np.min(norm_img_a, axis=2), kernel)
        transmission = 1.0 - (omega * dark_norm)
        transmission = np.clip(transmission, 0.15, 1.0)

        # Guided filter / fast Gaussian refinement of transmission
        t_guided = cv2.GaussianBlur(transmission, (15, 15), 0)

        # Recover scene radiance: J = (I - A) / t + A
        recovered = np.zeros_like(img_norm)
        for c in range(3):
            recovered[:, :, c] = ((img_norm[:, :, c] - A[c]) / t_guided) + A[c]

        return np.clip(recovered * 255.0, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Master Adaptive Enhancement Pipeline
    # -------------------------------------------------------------------------
    def enhance(
        self,
        image_rgb: np.ndarray,
        assessment: Optional[QualityAssessmentResult] = None,
        force_transforms: Optional[List[str]] = None,
    ) -> AdaptiveEnhancementResult:
        """
        Execute targeted, sequence-optimized enhancement tailored to identified defects.

        Sequencing Strategy:
        1. Letterbox / border trimming
        2. Geometric tilt correction
        3. Denoising / JPEG deblocking (applied BEFORE sharpening)
        4. Dehazing & White balance
        5. Illumination & exposure normalization (Gamma / CLAHE)
        6. Sharpness & deblurring (applied AFTER noise reduction)
        7. Super-resolution (if low resolution)
        """
        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
            raise ValueError("Input image must be a 3D numpy array (H, W, 3)")

        # Deep copy to guarantee original image is never mutated
        original_copy = image_rgb.copy()
        current = original_copy.copy()

        # Assess initial quality if not provided
        initial_assessment = assessment or self.assessor.assess(original_copy)
        issues = set(initial_assessment.detected_issues)
        applied_records: List[EnhancementRecord] = []

        # If image is already high quality (score >= 82) and no explicit defects, retain original untouched
        if initial_assessment.overall_score >= 82.0 and not issues and not force_transforms:
            return AdaptiveEnhancementResult(
                enhanced_image=original_copy,
                original_preserved=True,
                enhancements_applied=[],
                enhancement_success=True,
                assessment_before=initial_assessment,
                assessment_after=initial_assessment,
                quality_gain=0.0,
            )

        # ---------------------------------------------------------------------
        # Stage 1: Framing & Borders
        # ---------------------------------------------------------------------
        if "letterbox_borders" in issues or (force_transforms and "letterbox" in str(force_transforms)):
            t0 = time.perf_counter()
            trimmed = self.remove_letterbox_borders(current)
            if trimmed.shape != current.shape:
                current = trimmed
                applied_records.append(EnhancementRecord(
                    technique="Letterbox Border Trimming",
                    target_defect="letterbox_borders",
                    rationale="Removed solid extraneous border bands to isolate true vehicle region.",
                    parameters={"new_dimensions": f"{current.shape[1]}x{current.shape[0]}"},
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                ))

        # ---------------------------------------------------------------------
        # Stage 2: Geometric Rectification (Conservative)
        # ---------------------------------------------------------------------
        tilt_deg = initial_assessment.quality_metrics["geometric_quality_indicators"]["detected_tilt_deg"]
        tilt_conf = initial_assessment.quality_metrics["geometric_quality_indicators"]["tilt_confidence"]
        if ("tilted_image" in issues and tilt_conf >= 0.35) or (force_transforms and any("rotat" in str(t) for t in force_transforms)):
            t0 = time.perf_counter()
            angle_to_correct = -tilt_deg if abs(tilt_deg) >= 0.5 else -10.0
            corrected = self.apply_rotation_correction(current, angle_to_correct)
            if not np.array_equal(corrected, current):
                current = corrected
                applied_records.append(EnhancementRecord(
                    technique="Geometric Rotation Rectification",
                    target_defect="tilted_image",
                    rationale=f"Corrected {angle_to_correct:.1f}° horizon tilt based on dominant structural lines.",
                    parameters={"rotation_angle_deg": round(angle_to_correct, 2)},
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                ))

        # ---------------------------------------------------------------------
        # Stage 3: Noise & Artifact Suppression (Must precede sharpening!)
        # ---------------------------------------------------------------------
        if "sensor_noise" in issues or (force_transforms and "denoise" in str(force_transforms)):
            t0 = time.perf_counter()
            noise_val = initial_assessment.quality_metrics["noise_estimate"]
            denoised = self.apply_nlm_denoising(current, strength=min(12.0, noise_val * 1.2))
            current = denoised
            applied_records.append(EnhancementRecord(
                technique="Non-Local Means Denoising",
                target_defect="sensor_noise",
                rationale="Suppressed high-frequency Gaussian/sensor noise prior to feature extraction.",
                parameters={"noise_sigma_est": round(noise_val, 2)},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        if "jpeg_compression_artifacts" in issues or (force_transforms and "deblock" in str(force_transforms)):
            t0 = time.perf_counter()
            deblocked = self.apply_jpeg_deblocking(current)
            current = deblocked
            applied_records.append(EnhancementRecord(
                technique="JPEG Deblocking Filter",
                target_defect="jpeg_compression_artifacts",
                rationale="Attenuated 8x8 block boundary compression ringing artifacts.",
                parameters={"blockiness_index": initial_assessment.quality_metrics["noise_estimate"]},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        # ---------------------------------------------------------------------
        # Stage 4: Dehazing & Color Balance
        # ---------------------------------------------------------------------
        if "haze_or_fog" in issues or (force_transforms and "dehaze" in str(force_transforms)):
            t0 = time.perf_counter()
            dehazed = self.apply_dehaze_dcp(current)
            current = dehazed
            applied_records.append(EnhancementRecord(
                technique="Dark Channel Prior Dehazing",
                target_defect="haze_or_fog",
                rationale="Restored atmospheric transmission and scenic contrast obscured by fog/haze.",
                parameters={"omega": 0.85},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        if "color_cast" in issues or (force_transforms and "white_balance" in str(force_transforms)):
            t0 = time.perf_counter()
            balanced = self.apply_white_balance_gray_world(current)
            current = balanced
            applied_records.append(EnhancementRecord(
                technique="Gray World White Balance",
                target_defect="color_cast",
                rationale="Normalized chromatic imbalance to restore neutral white point.",
                parameters={"color_cast_score": initial_assessment.quality_metrics["color_balance_score"]},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        # ---------------------------------------------------------------------
        # Stage 5: Illumination & Exposure Normalization
        # ---------------------------------------------------------------------
        if "uneven_illumination" in issues:
            t0 = time.perf_counter()
            illum_corr = self.apply_illumination_correction(current)
            current = illum_corr
            applied_records.append(EnhancementRecord(
                technique="Illumination Field Normalization",
                target_defect="uneven_illumination",
                rationale="Compensated for harsh localized shadows and directional backlighting.",
                parameters={"filter_sigma": 40.0},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        if "underexposed" in issues or "overexposed" in issues or (force_transforms and "gamma" in str(force_transforms)):
            t0 = time.perf_counter()
            gamma_img, applied_gamma = self.apply_adaptive_gamma(current)
            current = gamma_img
            applied_records.append(EnhancementRecord(
                technique="Adaptive Gamma Brightening/Dimming",
                target_defect="underexposed" if "underexposed" in issues else "overexposed",
                rationale=f"Dynamically mapped midtones using power curve (gamma={applied_gamma:.2f}).",
                parameters={"gamma": round(applied_gamma, 2)},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        if "low_contrast" in issues or (force_transforms and "clahe" in str(force_transforms)):
            t0 = time.perf_counter()
            clahe_img = self.apply_clahe_lab(current, clip_limit=2.5)
            current = clahe_img
            applied_records.append(EnhancementRecord(
                technique="CLAHE LAB Contrast Equalization",
                target_defect="low_contrast",
                rationale="Equalized localized luminance histogram without shifting color channels.",
                parameters={"clip_limit": 2.5, "tile_size": 8},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        # ---------------------------------------------------------------------
        # Stage 6: Sharpness & Detail Restoration
        # ---------------------------------------------------------------------
        if "motion_blur" in issues:
            t0 = time.perf_counter()
            deblurred = self.apply_motion_deblur(current, angle_deg=tilt_deg, length=5)
            current = deblurred
            applied_records.append(EnhancementRecord(
                technique="Directional Motion Deblurring",
                target_defect="motion_blur",
                rationale="Applied directional regularized high-pass deconvolution for motion blur.",
                parameters={"estimated_vector_deg": round(tilt_deg, 2)},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))
        elif "blur" in issues or "defocus_blur" in issues or (force_transforms and "sharpen" in str(force_transforms)):
            t0 = time.perf_counter()
            sharpened = self.apply_unsharp_mask(current, radius=3, amount=1.6, threshold=4)
            current = sharpened
            applied_records.append(EnhancementRecord(
                technique="Thresholded Unsharp Masking",
                target_defect="defocus_blur" if "defocus_blur" in issues else "blur",
                rationale="Boosted edge acutance while suppressing flat-region noise amplification.",
                parameters={"radius": 3, "amount": 1.6, "threshold": 4},
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            ))

        # ---------------------------------------------------------------------
        # Stage 7: Resolution Enhancement
        # ---------------------------------------------------------------------
        if "low_resolution" in issues or (force_transforms and "super_res" in str(force_transforms)):
            t0 = time.perf_counter()
            super_res = self.apply_super_resolution_interp(current, min_target=640)
            if super_res.shape != current.shape:
                current = super_res
                applied_records.append(EnhancementRecord(
                    technique="Lanczos-4 Super-Resolution Upscaling",
                    target_defect="low_resolution",
                    rationale="Upscaled low-resolution input with edge-preserving acutance filter to 640px baseline.",
                    parameters={"new_dimensions": f"{current.shape[1]}x{current.shape[0]}"},
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                ))

        # ---------------------------------------------------------------------
        # Final Assessment & Fail-Safe Verification
        # ---------------------------------------------------------------------
        final_assessment = self.assessor.assess(current)
        quality_gain = final_assessment.overall_score - initial_assessment.overall_score

        # Fail-Safe Guardrail: If enhancements degraded the overall score significantly
        # (accounting for the fact that super-resolution smoothing on small pixel grids shifts variance)
        revert_threshold = -20.0 if "low_resolution" in issues else -12.0
        if quality_gain < revert_threshold and not force_transforms:
            logger.warning(
                f"Enhancement failed safety guardrail: quality dropped by {quality_gain:.1f}. "
                "Reverting to original image."
            )
            return AdaptiveEnhancementResult(
                enhanced_image=original_copy,
                original_preserved=True,
                enhancements_applied=[],
                enhancement_success=False,
                assessment_before=initial_assessment,
                assessment_after=initial_assessment,
                quality_gain=0.0,
            )

        return AdaptiveEnhancementResult(
            enhanced_image=current,
            original_preserved=True,
            enhancements_applied=applied_records,
            enhancement_success=True,
            assessment_before=initial_assessment,
            assessment_after=final_assessment,
            quality_gain=quality_gain,
        )
