# -*- coding: utf-8 -*-
"""config.py
by fcascan 2025
"""
import os
import configparser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_ini = os.path.join(BASE_DIR, "config.ini")
parser = configparser.ConfigParser()
parser.read(config_ini)


# INFERENCE DEVICE
INFERENCE_DEVICE = parser.get("INFERENCE", "device", fallback="NPU").strip().upper()
print(f"[config] INFERENCE_DEVICE: {INFERENCE_DEVICE}")


# PATHS
model_rknn_cfg = parser.get("PATHS", "model_rknn", fallback="assets/models/Crime_Detection_1-640-640-yolov11n.rknn")
MODEL_PATH = os.path.join(BASE_DIR, model_rknn_cfg)
print(f"[config] MODEL_PATH: {MODEL_PATH}")

model_onnx_cfg = parser.get("PATHS", "model_onnx", fallback="assets/models/Crime_Detection_1-640-640-yolov11n.onnx")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, model_onnx_cfg)
print(f"[config] ONNX_MODEL_PATH: {ONNX_MODEL_PATH}")


# IMAGE
img_width = parser.getint("IMAGE", "img_width", fallback=640)
img_height = parser.getint("IMAGE", "img_height", fallback=640)
IMG_SIZE = (img_width, img_height)
print(f"[config] IMG_SIZE: {IMG_SIZE}")

FPS_TEXT_SIZE = parser.getfloat("IMAGE", "fps_text_size", fallback=0.5)
print(f"[config] FPS_TEXT_SIZE: {FPS_TEXT_SIZE}")
LABEL_TEXT_SIZE = parser.getfloat("IMAGE", "label_text_size", fallback=0.4)
print(f"[config] LABEL_TEXT_SIZE: {LABEL_TEXT_SIZE}")


# CAMERA
MAX_CAMERAS_TO_SCAN = parser.getint("CAMERA", "max_cameras_to_scan", fallback=6)
print(f"[config] MAX_CAMERAS_TO_SCAN: {MAX_CAMERAS_TO_SCAN}")


# CLASSES
MODEL_LABELS_PATH = parser.get("PATHS", "model_labels", fallback=None)
print(f"[config] MODEL_LABELS_PATH: {MODEL_LABELS_PATH}")
labels = None
if MODEL_LABELS_PATH and os.path.exists(os.path.join(BASE_DIR, MODEL_LABELS_PATH)):
    with open(os.path.join(BASE_DIR, MODEL_LABELS_PATH), "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"[config] Loaded labels from file: {labels}")
else:
    default_labels_cfg = parser.get("CLASSES", "default_labels", fallback="person")
    labels = [name.strip() for name in default_labels_cfg.split(",")]
    print(f"[config] Loaded default_labels from config: {labels}")
CLASSES = tuple(labels)
print(f"[config] CLASSES: {CLASSES}")