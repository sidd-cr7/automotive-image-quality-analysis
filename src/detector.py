"""YOLOv8 Vehicle Detector module."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

from .config import (
    DETECTION_MODEL_PATH,
    VEHICLE_CLASS_IDS,
    DEFAULT_DETECTION_CONF,
    DEFAULT_IOU_THRESH,
    DEVICE,
)


@dataclass
class DetectionResult:
    box: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    crop: np.ndarray  # RGB image crop


class VehicleDetector:
    """Wrapper for the YOLOv8 vehicle detection model."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        vehicle_class_ids: Optional[List[int]] = None,
    ):
        self.model_path = Path(model_path or DETECTION_MODEL_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Detection model not found at {self.model_path}")

        self.device = device or DEVICE
        self.vehicle_class_ids = set(vehicle_class_ids or VEHICLE_CLASS_IDS)
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = DEFAULT_DETECTION_CONF,
        iou_threshold: float = DEFAULT_IOU_THRESH,
        agnostic_nms: bool = True,
    ) -> List[DetectionResult]:
        """
        Detect vehicles in an RGB image and extract vehicle crops.

        Args:
            image: Input image in RGB format as a numpy uint8 array (H, W, C).
            conf_threshold: Minimum confidence score.
            iou_threshold: NMS IoU threshold.
            agnostic_nms: Whether to apply class-agnostic NMS to eliminate overlapping vehicle classes.

        Returns:
            List of DetectionResult dataclass objects.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

        height, width = image.shape[:2]

        # Run inference filtering for vehicle class IDs directly
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=list(self.vehicle_class_ids),
            agnostic_nms=agnostic_nms,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        for box in first_res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # Filter for vehicle classes
            if cls_id in self.vehicle_class_ids:
                xyxy = box.xyxy[0].cpu().numpy()
                x1 = max(0, int(round(xyxy[0])))
                y1 = max(0, int(round(xyxy[1])))
                x2 = min(width, int(round(xyxy[2])))
                y2 = min(height, int(round(xyxy[3])))

                # Ensure non-empty crop
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2].copy()
                    class_name = self.names.get(cls_id, f"vehicle_{cls_id}")
                    detections.append(
                        DetectionResult(
                            box=[x1, y1, x2, y2],
                            confidence=conf,
                            class_id=cls_id,
                            class_name=class_name,
                            crop=crop,
                        )
                    )

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
