AUTOVISION AI - EXTERIOR COMPONENT MODEL

Architecture: YOLO11n-seg
Dataset: Carparts-Seg
Number of classes: 23
Input size: 640
Checkpoint: exterior_best.pt

Usage:
from ultralytics import YOLO
model = YOLO("exterior_best.pt")
results = model.predict("car.jpg", conf=0.25)
