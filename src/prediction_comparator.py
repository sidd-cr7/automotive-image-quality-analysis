"""Intelligent Prediction Comparator and Image Selection Engine (Part 3).

Evaluates and compares computer vision and multimodal perception outputs obtained
from both the Original image and the Part 2 Enhanced image:
- Branch 1: Vehicle Detection (YOLOv8) & Fine-Grained Classification (ResNet-50)
- Branch 2: Exterior Component Detection & Instance Segmentation (YOLO11n-seg)
- Branch 3: Interior and Engine Cabin Analysis (Groq Vision)

Core Principle: A higher confidence score alone does NOT imply higher accuracy.
If evidence is inconclusive or enhancement introduces artifact risks, the engine
conservatively selects the original image.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger("autovision.prediction_comparator")


def calculate_box_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


@dataclass
class BranchComparisonResult:
    branch_name: str
    selected_version: str  # "original" or "enhanced"
    comparison_score: float  # [-100, +100] scale, positive favors enhanced
    selection_reason: str
    original_prediction: Dict[str, Any]
    enhanced_prediction: Dict[str, Any]
    predictions_agree: Optional[bool]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_name": self.branch_name,
            "selected_version": self.selected_version,
            "comparison_score": round(self.comparison_score, 1),
            "selection_reason": self.selection_reason,
            "predictions_agree": self.predictions_agree,
            "original_prediction": self.original_prediction,
            "enhanced_prediction": self.enhanced_prediction,
            "metrics": self.metrics,
        }


@dataclass
class ComprehensiveComparisonReport:
    vehicle_identification: BranchComparisonResult
    exterior_analysis: BranchComparisonResult
    interior_engine_analysis: BranchComparisonResult
    comparison_enabled: bool
    comparison_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_identification": self.vehicle_identification.to_dict(),
            "exterior_analysis": self.exterior_analysis.to_dict(),
            "interior_engine_analysis": self.interior_engine_analysis.to_dict(),
            "comparison_enabled": self.comparison_enabled,
            "comparison_mode": self.comparison_mode,
        }


class PredictionComparator:
    """
    Evaluates branch predictions from original vs enhanced images, detects
    enhancement-induced artifacts and class flips, and makes independent,
    evidence-backed image selections per branch.
    """

    def __init__(
        self,
        b1_agreement_weight: float = 20.0,
        b1_conf_delta_weight: float = 25.0,
        b1_disagreement_penalty: float = 35.0,
        b2_mask_consistency_weight: float = 30.0,
        b2_new_class_reward: float = 12.0,
        b2_spurious_edge_penalty: float = 30.0,
    ):
        self.b1_agreement_weight = b1_agreement_weight
        self.b1_conf_delta_weight = b1_conf_delta_weight
        self.b1_disagreement_penalty = b1_disagreement_penalty
        self.b2_mask_consistency_weight = b2_mask_consistency_weight
        self.b2_new_class_reward = b2_new_class_reward
        self.b2_spurious_edge_penalty = b2_spurious_edge_penalty

    # -------------------------------------------------------------------------
    # 1. Branch 1 Comparison: Vehicle Identification
    # -------------------------------------------------------------------------
    def compare_vehicle_identification(
        self,
        original_b1: Dict[str, Any],
        enhanced_b1: Dict[str, Any],
        quality_score_orig: float = 70.0,
        quality_gain: float = 0.0,
        enhancements_applied: Optional[List[Dict[str, Any]]] = None,
    ) -> BranchComparisonResult:
        """
        Compare YOLOv8 detection and ResNet-50 classification across original and enhanced inputs.
        Does NOT blindly prefer higher confidence.
        """
        enhancements = enhancements_applied or []
        preds_orig = original_b1.get("predictions", [])
        preds_enh = enhanced_b1.get("predictions", [])

        p1_orig = preds_orig[0] if preds_orig else {}
        p1_enh = preds_enh[0] if preds_enh else {}

        # 1. Detection Stability & Localization
        det_count_orig = len(preds_orig)
        det_count_enh = len(preds_enh)
        box_orig = p1_orig.get("box", [])
        box_enh = p1_enh.get("box", [])
        box_iou = calculate_box_iou(box_orig, box_enh) if (box_orig and box_enh) else 0.0
        det_conf_orig = float(p1_orig.get("detection_confidence", 0.0))
        det_conf_enh = float(p1_enh.get("detection_confidence", 0.0))

        # 2. Classification Stability & Consistency
        make_orig = p1_orig.get("make", "")
        model_orig = p1_orig.get("model", "")
        year_orig = p1_orig.get("year", 0)
        conf_orig = float(p1_orig.get("classification_confidence", 0.0))

        make_enh = p1_enh.get("make", "")
        model_enh = p1_enh.get("model", "")
        year_enh = p1_enh.get("year", 0)
        conf_enh = float(p1_enh.get("classification_confidence", 0.0))

        predictions_agree = bool(
            make_orig and make_enh and
            make_orig.lower() == make_enh.lower() and
            model_orig.lower() == model_enh.lower()
        )

        conf_delta = conf_enh - conf_orig

        # Top-5 Overlap
        top5_orig_labels = {item.get("label") for item in p1_orig.get("top5", [])}
        top5_enh_labels = {item.get("label") for item in p1_enh.get("top5", [])}
        top5_shared = len(top5_orig_labels & top5_enh_labels)

        # Margin over Top-2 (Entropy / decisiveness check)
        margin_orig = 0.0
        if len(p1_orig.get("top5", [])) >= 2:
            margin_orig = conf_orig - float(p1_orig["top5"][1].get("confidence", 0.0))

        margin_enh = 0.0
        if len(p1_enh.get("top5", [])) >= 2:
            margin_enh = conf_enh - float(p1_enh["top5"][1].get("confidence", 0.0))

        # 3. Artifact & Class-Flip Investigation
        # If prediction flipped, investigate whether enhancement caused color/edge distortion
        color_shift_suspected = any(
            "white balance" in str(e.get("technique", "")).lower() or
            "gamma" in str(e.get("technique", "")).lower()
            for e in enhancements
        )
        edge_artifact_suspected = any(
            "unsharp" in str(e.get("technique", "")).lower() or
            "sharpen" in str(e.get("technique", "")).lower()
            for e in enhancements
        )

        # 4. Compute Comparison Score (-100 to +100)
        # Positive favors enhanced, negative favors original
        score = 0.0

        if predictions_agree:
            score += self.b1_agreement_weight  # Base consistency bonus
            score += self.b1_conf_delta_weight * conf_delta  # Benefit from confidence gain
            if box_iou >= 0.80:
                score += 15.0  # Stable localization
            elif box_iou < 0.40 and box_orig and box_enh:
                score -= 20.0  # Localization drift
            score += max(-10.0, min(15.0, (margin_enh - margin_orig) * 20.0))
            score += quality_gain * 0.4
        else:
            # Predictions disagree (Class flip)
            # If original image was already good quality (score >= 65), class flip is penalized heavily
            if quality_score_orig >= 65.0:
                score -= self.b1_disagreement_penalty
                if color_shift_suspected:
                    score -= 15.0  # Suspected color shift caused class change
                if edge_artifact_suspected:
                    score -= 10.0
                # Do not reward higher confidence when class flipped on an already good image!
                if conf_enh > conf_orig:
                    score -= 10.0  # Overconfidence on potentially distorted image
            else:
                # Original image was poor/underexposed; enhancement may have legitimately rescued classification
                if conf_enh >= 0.50 and margin_enh >= 0.20 and quality_gain > 5.0:
                    score += 15.0  # Likely legitimate recovery from degraded input
                else:
                    score -= 15.0  # Inconclusive class flip

        score = max(-100.0, min(100.0, score))

        # Check whether enhancement provides meaningful improvement
        has_enhancements = (enhancements_applied is None) or bool(enhancements_applied)
        meaningful_gain = (
            (conf_delta >= 0.04) or
            (quality_gain > 5.0 and conf_delta >= 0.01) or
            (det_count_orig == 0 and det_count_enh > 0) or
            (not predictions_agree and quality_score_orig < 60.0 and conf_enh >= 0.50 and margin_enh >= 0.20)
        )

        # 5. Selection Decision (Conservative threshold: favors original unless enhanced is clearly superior)
        if score >= 12.0 and det_count_enh >= 1 and meaningful_gain and has_enhancements and (quality_gain > 0.0 or conf_delta > 0.0):
            selected_version = "enhanced"
            if predictions_agree:
                reason = (
                    f"Selected enhanced image: Predictions agree on '{make_enh} {model_enh}'. "
                    f"Confidence increased by {conf_delta * 100:+.1f}% with stable localization (IoU: {box_iou:.2f})."
                )
            else:
                reason = (
                    f"Selected enhanced image: Original input was degraded (quality: {quality_score_orig:.1f}). "
                    f"Enhancement recovered clear vehicle features yielding decisive prediction '{make_enh} {model_enh}' "
                    f"with confidence {conf_enh * 100:.1f}% (margin: {margin_enh:.2f})."
                )
        else:
            selected_version = "original"
            if not predictions_agree and preds_orig and preds_enh:
                reason = (
                    f"Selected original image: Enhancement caused prediction disagreement "
                    f"(Original: '{make_orig} {model_orig}' [{conf_orig*100:.1f}%] vs "
                    f"Enhanced: '{make_enh} {model_enh}' [{conf_enh*100:.1f}%]). "
                    f"Retained original to prevent artifact-induced misclassification."
                )
            elif box_iou < 0.40 and box_orig and box_enh:
                reason = (
                    f"Selected original image: Enhanced image showed bounding-box localization drift "
                    f"(IoU: {box_iou:.2f})."
                )
            else:
                reason = (
                    f"Selected original image: Prediction consistency maintained; "
                    f"enhancement offered insufficient reliability gain (score: {score:.1f})."
                )

        metrics = {
            "detection_count_orig": det_count_orig,
            "detection_count_enh": det_count_enh,
            "primary_bbox_iou": round(box_iou, 3),
            "detection_conf_delta": round(det_conf_enh - det_conf_orig, 4),
            "classification_conf_orig": round(conf_orig, 4),
            "classification_conf_enh": round(conf_enh, 4),
            "classification_conf_delta": round(conf_delta, 4),
            "top5_shared_count": top5_shared,
            "margin_orig": round(margin_orig, 4),
            "margin_enh": round(margin_enh, 4),
            "color_shift_suspected": color_shift_suspected,
        }

        return BranchComparisonResult(
            branch_name="vehicle_identification",
            selected_version=selected_version,
            comparison_score=score,
            selection_reason=reason,
            original_prediction=p1_orig,
            enhanced_prediction=p1_enh,
            predictions_agree=predictions_agree if (preds_orig and preds_enh) else None,
            metrics=metrics,
        )

    # -------------------------------------------------------------------------
    # 2. Branch 2 Comparison: Exterior Segmentation
    # -------------------------------------------------------------------------
    def compare_exterior_segmentation(
        self,
        original_b2: Dict[str, Any],
        enhanced_b2: Dict[str, Any],
    ) -> BranchComparisonResult:
        """
        Compare YOLO11n-seg instance segmentations across original and enhanced inputs.
        Does NOT assume more detections equals better performance.
        Penalizes false edges and fragmented mask artifacts.
        """
        p_orig = original_b2.get("predictions", {})
        p_enh = enhanced_b2.get("predictions", {})

        comps_orig = p_orig.get("components", [])
        comps_enh = p_enh.get("components", [])

        classes_orig = set(p_orig.get("detected_classes", []))
        classes_enh = set(p_enh.get("detected_classes", []))

        total_orig = len(comps_orig)
        total_enh = len(comps_enh)

        newly_revealed = list(classes_enh - classes_orig)
        lost_classes = list(classes_orig - classes_enh)
        shared_classes = list(classes_orig & classes_enh)

        mean_conf_orig = float(original_b2.get("confidence", 0.0))
        mean_conf_enh = float(enhanced_b2.get("confidence", 0.0))
        conf_delta = mean_conf_enh - mean_conf_orig

        # Mask Consistency (Mean IoU across matched components of same class)
        ious = []
        for cls_name in shared_classes:
            matched_orig = [c for c in comps_orig if c.get("name") == cls_name]
            matched_enh = [c for c in comps_enh if c.get("name") == cls_name]
            for co in matched_orig:
                best_iou = 0.0
                for ce in matched_enh:
                    iou = calculate_box_iou(co.get("box", []), ce.get("box", []))
                    if iou > best_iou:
                        best_iou = iou
                if best_iou > 0.0:
                    ious.append(best_iou)

        mask_consistency = float(np.mean(ious)) if ious else (1.0 if not comps_orig and not comps_enh else 0.0)

        # Artifact Risk Detection: Spurious noisy edges from over-sharpening
        # Check if new components in enhanced have very low confidence (< 0.25) or tiny area
        spurious_classes = {
            c.get("name")
            for c in comps_enh
            if c.get("name") in newly_revealed and (c.get("confidence", 0.0) < 0.25 or c.get("area_ratio", 0.0) < 0.005)
        }
        spurious_artifact_detected = len(spurious_classes) >= 2
        spurious_count = len(spurious_classes)

        # Lost high-confidence detections in enhanced
        lost_high_conf = sum(
            1 for c in comps_orig
            if c.get("name") in lost_classes and c.get("confidence", 0.0) >= 0.55
        )

        # Compute Comparison Score (-100 to +100)
        score = 0.0
        score += self.b2_mask_consistency_weight * mask_consistency
        score += 20.0 * conf_delta

        # Reward legitimate component revelation
        legitimate_new = [c for c in newly_revealed if c not in spurious_classes]
        score += self.b2_new_class_reward * len(legitimate_new)

        # Penalties
        if spurious_artifact_detected:
            score -= self.b2_spurious_edge_penalty
        if lost_high_conf > 0:
            score -= 25.0 * lost_high_conf

        score = max(-100.0, min(100.0, score))

        # Meaningful improvement check
        meaningful_improvement_b2 = (
            (len(legitimate_new) > 0 and not spurious_artifact_detected) or
            (conf_delta >= 0.04 and mask_consistency >= 0.60)
        )

        # Selection Decision
        if score >= 10.0 and not spurious_artifact_detected and total_enh >= total_orig and meaningful_improvement_b2 and total_enh > 0:
            selected_version = "enhanced"
            reason = (
                f"Selected enhanced image: Exterior segmentation achieved high mask consistency "
                f"(IoU: {mask_consistency:.2f}) and mean confidence gain {conf_delta * 100:+.1f}%. "
                f"Components: {total_enh} detected (vs {total_orig} in original)."
            )
            if newly_revealed:
                reason += f" Recovered components: {', '.join(newly_revealed)}."
        else:
            selected_version = "original"
            if spurious_artifact_detected:
                reason = (
                    f"Selected original image: Enhanced image introduced {spurious_count} spurious/noisy "
                    f"segmentation fragments with low confidence, indicating sharpening artifacts."
                )
            elif lost_high_conf > 0:
                reason = (
                    f"Selected original image: Enhanced version dropped {lost_high_conf} high-confidence "
                    f"component detection(s) present in the original."
                )
            else:
                reason = (
                    f"Selected original image: Segmentation boundaries stable in original "
                    f"(score: {score:.1f}); enhancement did not demonstrate clear boundary improvement."
                )

        metrics = {
            "total_components_orig": total_orig,
            "total_components_enh": total_enh,
            "mask_consistency_iou": round(mask_consistency, 3),
            "mean_conf_delta": round(conf_delta, 4),
            "newly_revealed_classes": newly_revealed,
            "lost_classes": lost_classes,
            "spurious_artifacts_flag": spurious_artifact_detected,
        }

        return BranchComparisonResult(
            branch_name="exterior_analysis",
            selected_version=selected_version,
            comparison_score=score,
            selection_reason=reason,
            original_prediction=p_orig,
            enhanced_prediction=p_enh,
            predictions_agree=bool(classes_orig == classes_enh),
            metrics=metrics,
        )

    # -------------------------------------------------------------------------
    # 3. Branch 3 Comparison: Interior and Engine Intelligence
    # -------------------------------------------------------------------------
    def compare_interior_engine(
        self,
        original_b3: Dict[str, Any],
        enhanced_b3: Dict[str, Any],
        quality_gain: float = 0.0,
    ) -> BranchComparisonResult:
        """
        Compare Groq Vision interior and engine assessments across original and enhanced inputs.
        Does NOT treat Groq agreement as ground-truth verification.
        """
        p_orig = original_b3.get("predictions", {})
        p_enh = enhanced_b3.get("predictions", {})

        int_vis_orig = bool(p_orig.get("is_interior_visible", False))
        int_vis_enh = bool(p_enh.get("is_interior_visible", False))

        eng_vis_orig = bool(p_orig.get("is_engine_visible", False))
        eng_vis_enh = bool(p_enh.get("is_engine_visible", False))

        int_comps_orig = len(p_orig.get("interior_components", []))
        int_comps_enh = len(p_enh.get("interior_components", []))

        # Check semantic agreement
        views_agree = (p_orig.get("view_perspective") == p_enh.get("view_perspective"))
        vis_agree = (int_vis_orig == int_vis_enh and eng_vis_orig == eng_vis_enh)
        predictions_agree = bool(views_agree and vis_agree)

        score = 0.0
        if predictions_agree:
            score += 15.0

        # If enhancement brought dark interior or engine components into clear view
        if (not int_vis_orig and int_vis_enh) or (not eng_vis_orig and eng_vis_enh):
            score += 30.0  # Significant visibility gain
        elif int_comps_enh > int_comps_orig:
            score += 15.0

        score += quality_gain * 0.3
        score = max(-100.0, min(100.0, score))

        # Meaningful visibility gain check
        meaningful_b3_gain = (
            ((not int_vis_orig and int_vis_enh) or (not eng_vis_orig and eng_vis_enh)) or
            (int_comps_enh > int_comps_orig) or
            (quality_gain > 5.0 and (int_vis_enh or eng_vis_enh))
        )

        # Selection Decision
        if score >= 15.0 and meaningful_b3_gain:
            selected_version = "enhanced"
            reason = (
                f"Selected enhanced image: Illumination normalization provided clearer visibility "
                f"for visual reasoning (gain: +{quality_gain:.1f}). "
                f"Interior visible: {int_vis_enh}, Engine visible: {eng_vis_enh}."
            )
        else:
            selected_version = "original"
            reason = (
                f"Selected original image: Interpretations consistent ({'Agreed' if predictions_agree else 'Disagreed'}). "
                f"Original image provided sufficient clarity for visual interpretation."
            )

        metrics = {
            "interior_visible_orig": int_vis_orig,
            "interior_visible_enh": int_vis_enh,
            "engine_visible_orig": eng_vis_orig,
            "engine_visible_enh": eng_vis_enh,
            "interior_components_count_orig": int_comps_orig,
            "interior_components_count_enh": int_comps_enh,
            "quality_gain": round(quality_gain, 1),
        }

        return BranchComparisonResult(
            branch_name="interior_engine_analysis",
            selected_version=selected_version,
            comparison_score=score,
            selection_reason=reason,
            original_prediction=p_orig,
            enhanced_prediction=p_enh,
            predictions_agree=predictions_agree,
            metrics=metrics,
        )

    # -------------------------------------------------------------------------
    # Master Comprehensive Comparison Orchestration
    # -------------------------------------------------------------------------
    def compare_all(
        self,
        b1_original: Dict[str, Any],
        b1_enhanced: Dict[str, Any],
        b2_original: Dict[str, Any],
        b2_enhanced: Dict[str, Any],
        b3_original: Dict[str, Any],
        b3_enhanced: Dict[str, Any],
        quality_score_orig: float = 70.0,
        quality_gain: float = 0.0,
        enhancements_applied: Optional[List[Dict[str, Any]]] = None,
        comparison_mode: str = "full",
    ) -> ComprehensiveComparisonReport:
        """
        Execute full cross-branch original vs enhanced prediction comparison.
        Produces independent image selections and comparison metrics for all three branches.
        """
        # Branch 1 Comparison
        b1_comp = self.compare_vehicle_identification(
            original_b1=b1_original,
            enhanced_b1=b1_enhanced,
            quality_score_orig=quality_score_orig,
            quality_gain=quality_gain,
            enhancements_applied=enhancements_applied,
        )

        # Branch 2 Comparison
        b2_comp = self.compare_exterior_segmentation(
            original_b2=b2_original,
            enhanced_b2=b2_enhanced,
        )

        # Branch 3 Comparison
        b3_comp = self.compare_interior_engine(
            original_b3=b3_original,
            enhanced_b3=b3_enhanced,
            quality_gain=quality_gain,
        )

        if comparison_mode == "fast_identical":
            b1_comp.selected_version = "original"
            b1_comp.selection_reason = "Identical image input (fast mode); original selected by default."
            b2_comp.selected_version = "original"
            b2_comp.selection_reason = "Identical image input (fast mode); original selected by default."
            b3_comp.selected_version = "original"
            b3_comp.selection_reason = "Identical image input (fast mode); original selected by default."

        return ComprehensiveComparisonReport(
            vehicle_identification=b1_comp,
            exterior_analysis=b2_comp,
            interior_engine_analysis=b3_comp,
            comparison_enabled=True,
            comparison_mode=comparison_mode,
        )
