"""Classical Computer Vision Analysis module (Canny, Sobel, HOG, ORB, HSV, Saliency)."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
import torch


@dataclass
class ClassicalCVResult:
    edge_density: float
    sobel_gradient_mean: float
    orb_keypoint_count: int
    dominant_hue: int
    mean_saturation: float
    mean_value: float
    feature_vector_99d: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_density": round(self.edge_density, 4),
            "sobel_gradient_mean": round(self.sobel_gradient_mean, 2),
            "orb_keypoint_count": self.orb_keypoint_count,
            "dominant_hue": self.dominant_hue,
            "mean_saturation": round(self.mean_saturation, 2),
            "mean_value": round(self.mean_value, 2),
            "feature_dim": len(self.feature_vector_99d),
        }


class ClassicalCVAnalyzer:
    """Analyzer implementing classical computer vision feature extraction."""

    def __init__(self, orb_max_features: int = 200):
        self.orb = cv2.ORB_create(nfeatures=orb_max_features)

    def extract_features(self, image_rgb: np.ndarray) -> ClassicalCVResult:
        """
        Extract Canny, Sobel, HOG, ORB, and HSV features from vehicle crop.

        Args:
            image_rgb: RGB image crop as numpy array (H, W, 3).

        Returns:
            ClassicalCVResult dataclass.
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. Canny Edges
        canny_edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(canny_edges) / (h * w))

        # 2. Sobel Gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sobel_mean = float(np.mean(sobel_mag))

        # 3. ORB Keypoints
        keypoints, _ = self.orb.detectAndCompute(gray, None)
        orb_count = len(keypoints) if keypoints is not None else 0

        # 4. HSV Color Histogram (32 bins per channel, normalized)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        sum_h = float(np.sum(hist_h))
        sum_s = float(np.sum(hist_s))
        sum_v = float(np.sum(hist_v))

        norm_h = (hist_h / sum_h) if sum_h > 0 else np.zeros(32)
        norm_s = (hist_s / sum_s) if sum_s > 0 else np.zeros(32)
        norm_v = (hist_v / sum_v) if sum_v > 0 else np.zeros(32)

        dominant_hue = int(np.argmax(norm_h) * (180 // 32))
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))

        # 5. Build 99-dimensional classical feature vector
        # 32 (H) + 32 (S) + 32 (V) + 3 geometric/texture stats = 99
        aspect_ratio = float(w / h) if h > 0 else 1.0
        contrast = float(np.std(gray))
        norm_edge_stat = float(np.log1p(sobel_mean))

        feat_vector = np.concatenate([
            norm_h,
            norm_s,
            norm_v,
            [aspect_ratio, norm_edge_stat, contrast]
        ]).astype(float).tolist()

        return ClassicalCVResult(
            edge_density=edge_density,
            sobel_gradient_mean=sobel_mean,
            orb_keypoint_count=orb_count,
            dominant_hue=dominant_hue,
            mean_saturation=mean_sat,
            mean_value=mean_val,
            feature_vector_99d=feat_vector,
        )

    @staticmethod
    def generate_saliency_map(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate vanilla gradient saliency map for model interpretability.

        Args:
            model: PyTorch classification model in eval mode.
            input_tensor: Tensor of shape (1, 3, 224, 224) with requires_grad=True.
            target_class: Target class index.

        Returns:
            Normalized 2D saliency heatmap (224, 224) in range [0, 1].
        """
        tensor = input_tensor.clone().detach().requires_grad_(True)
        output = model(tensor)
        score = output[0, target_class]
        score.backward()

        # Maximum magnitude across color channels
        saliency, _ = torch.max(tensor.grad.data.abs(), dim=1)
        saliency = saliency[0].cpu().numpy()

        # Normalize to [0, 1]
        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency
