"""ResNet-50 Vehicle Classifier module."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .config import (
    CLASSIFICATION_MODEL_PATH,
    CLASSES_PATH,
    EFFICIENTNET_MODEL_PATH,
    CLASSIFIER_ENSEMBLE,
    NUM_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
)


@dataclass
class PredictionItem:
    class_id: int
    label: str
    make: str
    model: str
    year: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    top1: PredictionItem
    top5: List[PredictionItem]
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "make": self.top1.make,
            "model": self.top1.model,
            "year": self.top1.year,
            "label": self.top1.label,
            "confidence": self.top1.confidence,
            "top5": [item.to_dict() for item in self.top5],
            "embedding": self.embedding,
        }


class VehicleClassifier:
    """Wrapper for the fine-grained ResNet-50 vehicle classifier."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        classes_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path or CLASSIFICATION_MODEL_PATH)
        self.classes_path = Path(classes_path or CLASSES_PATH)
        self.device = torch.device(device or DEVICE)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found at {self.model_path}")
        if not self.classes_path.exists():
            raise FileNotFoundError(f"Classes mapping file not found at {self.classes_path}")

        # Load classes mapping
        with open(self.classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.classes_info = data.get("classes", [])

        if len(self.classes_info) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} classes in {self.classes_path}, got {len(self.classes_info)}"
            )

        # Build model architecture
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, NUM_CLASSES)

        # Load state dictionary
        state_dict = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Optional second model (EfficientNet-B0) averaged with ResNet-50 at the probability level.
        self.secondary = None
        if CLASSIFIER_ENSEMBLE and Path(EFFICIENTNET_MODEL_PATH).exists():
            try:
                eff = models.efficientnet_b0(weights=None)
                eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, NUM_CLASSES)
                eff.load_state_dict(torch.load(str(EFFICIENTNET_MODEL_PATH), map_location=self.device, weights_only=False))
                self.secondary = eff.to(self.device).eval()
            except Exception as e:  # keep ResNet-50 alone if the second model cannot load
                import logging
                logging.getLogger("autovision.classifier").warning(f"EfficientNet-B0 ensemble disabled: {e}")

        # Input preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def predict(
        self,
        image_crop: np.ndarray,
        top_k: int = 5,
        extract_embedding: bool = False,
    ) -> ClassificationResult:
        """
        Classify a vehicle crop and return top predictions and details.

        Args:
            image_crop: RGB crop as numpy array (H, W, 3) or PIL Image.
            top_k: Number of top predictions to return.
            extract_embedding: Whether to extract 2048-dim feature embedding.

        Returns:
            ClassificationResult dataclass.
        """
        if isinstance(image_crop, np.ndarray):
            pil_img = Image.fromarray(image_crop)
        elif isinstance(image_crop, Image.Image):
            pil_img = image_crop
        else:
            raise TypeError("image_crop must be a numpy ndarray or PIL Image")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        embedding = None
        with torch.no_grad():
            if extract_embedding:
                # Forward through backbone up to avgpool
                x = self.model.conv1(tensor)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                x = self.model.avgpool(x)
                emb_tensor = torch.flatten(x, 1)
                logits = self.model.fc(emb_tensor)
                embedding = emb_tensor[0].cpu().numpy().tolist()
            else:
                logits = self.model(tensor)

            probs = torch.softmax(logits, dim=1)[0]
            if self.secondary is not None:
                probs = (probs + torch.softmax(self.secondary(tensor), dim=1)[0]) / 2
            top_probs, top_indices = torch.topk(probs, min(top_k, NUM_CLASSES))

        predictions: List[PredictionItem] = []
        for p, idx in zip(top_probs, top_indices):
            cls_idx = idx.item()
            conf = float(p.item())
            info = self.classes_info[cls_idx]
            predictions.append(
                PredictionItem(
                    class_id=cls_idx,
                    label=info["label"],
                    make=info["make"],
                    model=info["model"],
                    year=info["year"],
                    confidence=conf,
                )
            )

        return ClassificationResult(
            top1=predictions[0],
            top5=predictions,
            embedding=embedding,
        )
