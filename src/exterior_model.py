"""YOLO11n-seg Exterior Component Detection and Instance Segmentation module."""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from ultralytics import YOLO

from .config import (
    EXTERIOR_MODEL_PATH,
    EXTERIOR_CLASSES_PATH,
    DEFAULT_EXTERIOR_CONF,
    DEFAULT_EXTERIOR_IOU,
    DEVICE,
)

logger = logging.getLogger("autovision.exterior")


@dataclass
class ExteriorComponentItem:
    class_id: int
    name: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2]
    area_pixels: int
    area_ratio: float  # Fraction of image area
    polygon_points_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExteriorSegmentationResult:
    components: List[ExteriorComponentItem]
    detected_classes: List[str]
    component_counts: Dict[str, int]
    mean_confidence: float
    image_shape: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "detected_classes": self.detected_classes,
            "component_counts": self.component_counts,
            "mean_confidence": round(self.mean_confidence, 4),
            "total_detected": len(self.components),
            "image_dimensions": {"height": self.image_shape[0], "width": self.image_shape[1]},
        }


class ExteriorComponentDetector:
    """
    Wrapper for YOLO11n-seg trained on Carparts-Seg (23 classes).
    Strictly reports only the 23 supported exterior component classes.
    """

    SUPPORTED_CLASSES_23 = [
        "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light",
        "back_light", "back_right_door", "back_right_light", "front_bumper", "front_door",
        "front_glass", "front_left_door", "front_left_light", "front_light", "front_right_door",
        "front_right_light", "hood", "left_mirror", "object", "right_mirror",
        "tailgate", "trunk", "wheel"
    ]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        classes_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path or EXTERIOR_MODEL_PATH)
        self.classes_path = Path(classes_path or EXTERIOR_CLASSES_PATH)
        self.device = device or DEVICE

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Exterior segmentation model not found at '{self.model_path}'. "
                f"Please ensure exterior_best.pt exists."
            )

        # Load classes from classes.txt if present, otherwise use built-in 23 classes
        self.classes_map: Dict[int, str] = {}
        if self.classes_path.exists():
            try:
                with open(self.classes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and ":" in line:
                            idx_str, name = line.split(":", 1)
                            self.classes_map[int(idx_str.strip())] = name.strip()
            except Exception as e:
                logger.warning(f"Could not parse classes.txt at {self.classes_path}: {e}")

        if not self.classes_map:
            self.classes_map = {i: name for i, name in enumerate(self.SUPPORTED_CLASSES_23)}

        # Load Ultralytics YOLO model
        self.model = YOLO(str(self.model_path))
        logger.info(f"Loaded YOLO11n-seg exterior model from {self.model_path} with {len(self.classes_map)} classes.")

    def detect_and_segment(
        self,
        image: np.ndarray,
        conf_threshold: float = DEFAULT_EXTERIOR_CONF,
        iou_threshold: float = DEFAULT_EXTERIOR_IOU,
    ) -> ExteriorSegmentationResult:
        """
        Detect and segment exterior components in an RGB image.

        Args:
            image: RGB numpy uint8 array (H, W, 3).
            conf_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.

        Returns:
            ExteriorSegmentationResult dataclass containing detected components and mask summaries.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

        height, width = image.shape[:2]
        total_pixels = height * width

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False,
        )

        components: List[ExteriorComponentItem] = []
        counts: Dict[str, int] = {}

        if results and len(results) > 0:
            first_res = results[0]
            boxes = first_res.boxes
            masks = first_res.masks

            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    name = self.classes_map.get(cls_id, self.model.names.get(cls_id, f"component_{cls_id}"))

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1 = max(0, int(round(xyxy[0])))
                    y1 = max(0, int(round(xyxy[1])))
                    x2 = min(width, int(round(xyxy[2])))
                    y2 = min(height, int(round(xyxy[3])))

                    # Extract mask metadata
                    polygon_count = 0
                    area_pixels = (x2 - x1) * (y2 - y1)
                    if masks is not None and masks.xy is not None and idx < len(masks.xy):
                        poly = masks.xy[idx]
                        if poly is not None and len(poly) > 0:
                            polygon_count = len(poly)

                    area_ratio = float(area_pixels / total_pixels) if total_pixels > 0 else 0.0

                    item = ExteriorComponentItem(
                        class_id=cls_id,
                        name=name,
                        confidence=conf,
                        box=[x1, y1, x2, y2],
                        area_pixels=area_pixels,
                        area_ratio=round(area_ratio, 4),
                        polygon_points_count=polygon_count,
                    )
                    components.append(item)
                    counts[name] = counts.get(name, 0) + 1

        # Sort components by confidence descending
        components.sort(key=lambda c: c.confidence, reverse=True)
        mean_conf = float(np.mean([c.confidence for c in components])) if components else 0.0

        return ExteriorSegmentationResult(
            components=components,
            detected_classes=sorted(list(counts.keys())),
            component_counts=counts,
            mean_confidence=mean_conf,
            image_shape=(height, width),
        )
