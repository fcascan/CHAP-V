# -*- coding: utf-8 -*-
"""config.py
by fcascan 2025
"""
import os
import configparser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_ini = os.path.join(BASE_DIR, "config.ini")
parser = configparser.ConfigParser()

def load_config(is_reload=False):
    """Load or reload configuration from config.ini file"""
    global parser, BENCHMARK_MODE, INFERENCE_DEVICE, MODEL_PATH, ONNX_MODEL_PATH
    global VIDEO_FILE_PATH, IMG_SIZE, FPS_TEXT_SIZE, LABEL_TEXT_SIZE, MAX_CAMERAS_TO_SCAN, CLASSES
    
    # Clear and re-read the config file
    if is_reload:
        parser.clear()
    parser.read(config_ini)
    
    # Load all configuration values
    BENCHMARK_MODE = parser.getboolean("MODE", "benchmark_mode", fallback=False)
    INFERENCE_DEVICE = parser.get("INFERENCE", "device", fallback="NPU").strip().upper()
    
    # Load paths
    model_rknn_cfg = parser.get("PATHS", "model_rknn", fallback="assets/models/yolov11n.rknn")
    MODEL_PATH = os.path.join(BASE_DIR, model_rknn_cfg)
    
    model_onnx_cfg = parser.get("PATHS", "model_onnx", fallback="assets/models/yolov11n.onnx")
    ONNX_MODEL_PATH = os.path.join(BASE_DIR, model_onnx_cfg)
    
    benchmark_video_cfg = parser.get("PATHS", "benchmark_video", fallback="assets/videos/benchmark.mp4")
    VIDEO_FILE_PATH = os.path.join(BASE_DIR, benchmark_video_cfg)
    
    # Load image settings
    img_width = parser.getint("IMAGE", "img_width", fallback=640)
    img_height = parser.getint("IMAGE", "img_height", fallback=640)
    IMG_SIZE = (img_width, img_height)
    
    FPS_TEXT_SIZE = parser.getfloat("IMAGE", "fps_text_size", fallback=0.5)
    LABEL_TEXT_SIZE = parser.getfloat("IMAGE", "label_text_size", fallback=0.4)
    
    # Load camera settings
    MAX_CAMERAS_TO_SCAN = parser.getint("CAMERA", "max_cameras_to_scan", fallback=6)
    
    # Load classes
    MODEL_LABELS_PATH = parser.get("PATHS", "model_labels", fallback=None)
    labels = None
    if MODEL_LABELS_PATH and os.path.exists(os.path.join(BASE_DIR, MODEL_LABELS_PATH)):
        with open(os.path.join(BASE_DIR, MODEL_LABELS_PATH), "r") as f:
            labels = [line.strip() for line in f if line.strip()]
    else:
        default_labels_cfg = parser.get("CLASSES", "default_labels", fallback="person")
        labels = [name.strip() for name in default_labels_cfg.split(",")]
    CLASSES = tuple(labels)
    
    action = "reloaded" if is_reload else "loaded"
    print(f"[CONFIG] Configuration {action}: benchmark_mode={BENCHMARK_MODE}, device={INFERENCE_DEVICE}, max_cameras={MAX_CAMERAS_TO_SCAN}")
    
    # Return updated config for convenience
    return {
        'benchmark_mode': BENCHMARK_MODE,
        'inference_device': INFERENCE_DEVICE,
        'max_cameras': MAX_CAMERAS_TO_SCAN,
        'img_size': IMG_SIZE,
        'classes': CLASSES
    }

def reload_config():
    """Reload configuration from config.ini file"""
    return load_config(is_reload=True)

# Initialize configuration on module import
load_config()