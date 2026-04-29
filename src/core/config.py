# -*- coding: utf-8 -*-
"""config.py
by fcascan 2025
"""
import os
import configparser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_ini = os.path.join(BASE_DIR, "config.ini")
parser = configparser.ConfigParser()

def _parse_color(value, fallback):
    """Parse a BGR color from either 'B,G,R' or '#RRGGBB' notation."""
    if value is None:
        return fallback

    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(int(max(0, min(255, component))) for component in value)

    text = str(value).strip()
    if not text:
        return fallback

    try:
        if text.startswith('#') and len(text) == 7:
            red = int(text[1:3], 16)
            green = int(text[3:5], 16)
            blue = int(text[5:7], 16)
            return (blue, green, red)

        parts = [int(part.strip()) for part in text.split(',')]
        if len(parts) == 3:
            return tuple(max(0, min(255, part)) for part in parts)
    except Exception:
        pass

    return fallback


def load_config(is_reload=False):
    """Load or reload configuration from config.ini file"""
    global parser, BENCHMARK_MODE, INFERENCE_DEVICE, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD, DEBUG_MODE
    global MODEL_PATH, ONNX_MODEL_PATH, VIDEO_FILE_PATH, IMG_SIZE, FPS_TEXT_SIZE, LABEL_TEXT_SIZE, OVERLAY_ENABLED, OVERLAY_TEXT_COLOR, MAX_CAMERAS_TO_SCAN, CLASSES, MODEL_LABELS_FILE_PATH
    global DETECTION_BOX_COLOR, DETECTION_LABEL_COLOR, DETECTION_LABEL_BACKGROUND_COLOR, DETECTION_BOX_THICKNESS, DETECTION_LABEL_TEXT_SIZE, DETECTION_LABEL_TEXT_THICKNESS
    
    # Clear and re-read the config file
    if is_reload:
        parser.clear()
    parser.read(config_ini)
    
    # Load all configuration values
    BENCHMARK_MODE = parser.getboolean("MODE", "benchmark_mode", fallback=False)
    INFERENCE_DEVICE = parser.get("INFERENCE", "device", fallback="NPU").strip().upper()
    ROCKCHIP_TARGET = parser.get("INFERENCE", "rockchip_target", fallback="rk3588").strip().lower()
    OBJ_THRESHOLD = parser.getfloat("INFERENCE", "obj_threshold", fallback=0.25)
    NMS_THRESHOLD = parser.getfloat("INFERENCE", "nms_threshold", fallback=0.45)
    DEBUG_MODE = parser.getboolean("INFERENCE", "debug_mode", fallback=False)
    
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
    OVERLAY_ENABLED = parser.getboolean("IMAGE", "show_overlay", fallback=True)
    OVERLAY_TEXT_COLOR = _parse_color(parser.get("IMAGE", "overlay_text_color", fallback="255,255,255"), (255, 255, 255))

    # Load detection styling settings
    DETECTION_BOX_COLOR = _parse_color(parser.get("DETECTION", "box_color", fallback="0,0,255"), (0, 0, 255))
    DETECTION_LABEL_COLOR = _parse_color(parser.get("DETECTION", "label_text_color", fallback="255,255,255"), (255, 255, 255))
    DETECTION_LABEL_BACKGROUND_COLOR = _parse_color(parser.get("DETECTION", "label_background_color", fallback="0,0,255"), (0, 0, 255))
    DETECTION_BOX_THICKNESS = parser.getint("DETECTION", "box_thickness", fallback=2)
    DETECTION_LABEL_TEXT_SIZE = parser.getfloat("DETECTION", "label_text_size", fallback=0.45)
    DETECTION_LABEL_TEXT_THICKNESS = parser.getint("DETECTION", "label_text_thickness", fallback=2)
    
    # Load camera settings
    MAX_CAMERAS_TO_SCAN = parser.getint("CAMERA", "max_cameras_to_scan", fallback=6)
    
    # Load classes with fallback order:
    # 1) PATHS.model_labels from config.ini
    # 2) project-root yolo11n.txt
    # 3) CLASSES.default_labels from config.ini
    model_labels_cfg = parser.get("PATHS", "model_labels", fallback="").strip()
    MODEL_LABELS_FILE_PATH = None
    labels = []

    if model_labels_cfg:
        configured_labels_path = model_labels_cfg
        if not os.path.isabs(configured_labels_path):
            configured_labels_path = os.path.join(BASE_DIR, configured_labels_path)
        if os.path.isfile(configured_labels_path):
            MODEL_LABELS_FILE_PATH = configured_labels_path

    if MODEL_LABELS_FILE_PATH is None:
        yolo11n_fallback = os.path.join(BASE_DIR, "yolo11n.txt")
        if os.path.isfile(yolo11n_fallback):
            MODEL_LABELS_FILE_PATH = yolo11n_fallback

    if MODEL_LABELS_FILE_PATH is not None:
        with open(MODEL_LABELS_FILE_PATH, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

    if not labels:
        default_labels_cfg = parser.get("CLASSES", "default_labels", fallback="person")
        labels = [name.strip() for name in default_labels_cfg.split(",") if name.strip()]

    CLASSES = tuple(labels)
    
    # Configure logging level based on debug mode
    import logging
    
    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if DEBUG_MODE:
        logging.basicConfig(
            level=logging.DEBUG, 
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        logging.debug("Debug mode enabled - detailed inference logging active")
        
        # Enable RKNN verbose logging if available
        os.environ['RKNN_LOG_LEVEL'] = '1'  # Enable verbose RKNN logging
        
    else:
        logging.basicConfig(
            level=logging.INFO, 
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
    
    action = "reloaded" if is_reload else "loaded"
    debug_status = "ON" if DEBUG_MODE else "OFF"
    logging.info(
        f"[CONFIG] Configuration {action}:\n"
        f"  benchmark_mode = {BENCHMARK_MODE}\n"
        f"  device = {INFERENCE_DEVICE}\n"
        f"  rockchip_target = {ROCKCHIP_TARGET}\n"
        f"  obj_threshold = {OBJ_THRESHOLD}\n"
        f"  nms_threshold = {NMS_THRESHOLD}\n"
        f"  overlay = {'ON' if OVERLAY_ENABLED else 'OFF'}\n"
        f"  detection_box_thickness = {DETECTION_BOX_THICKNESS}\n"
        f"  detection_label_text_size = {DETECTION_LABEL_TEXT_SIZE}\n"
        f"  detection_label_text_thickness = {DETECTION_LABEL_TEXT_THICKNESS}\n"
        f"  debug = {debug_status}\n"
        f"  max_cameras = {MAX_CAMERAS_TO_SCAN}"
    )
    
    # Return updated config for convenience
    return {
        'benchmark_mode': BENCHMARK_MODE,
        'inference_device': INFERENCE_DEVICE,
        'debug_mode': DEBUG_MODE,
        'overlay_enabled': OVERLAY_ENABLED,
        'max_cameras': MAX_CAMERAS_TO_SCAN,
        'img_size': IMG_SIZE,
        'classes': CLASSES,
        'detection_box_color': DETECTION_BOX_COLOR,
        'detection_label_color': DETECTION_LABEL_COLOR,
        'detection_label_background_color': DETECTION_LABEL_BACKGROUND_COLOR,
        'detection_box_thickness': DETECTION_BOX_THICKNESS,
        'detection_label_text_size': DETECTION_LABEL_TEXT_SIZE,
        'detection_label_text_thickness': DETECTION_LABEL_TEXT_THICKNESS
    }

def reload_config():
    """Reload configuration from config.ini file"""
    return load_config(is_reload=True)

# Initialize configuration on module import
load_config()
