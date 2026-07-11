# -*- coding: utf-8 -*-
"""config.py
Loads and reloads config.ini into module globals for the YOLO RKNN/NPU project.
by fcascan 2026
"""
import os
import configparser

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_ini = os.path.join(BASE_DIR, "config.ini")
# interpolation=None so values may contain a literal '%' (e.g. inference_device = CPU-50%).
parser = configparser.ConfigParser(interpolation=None)


def _parse_core_list(value):
    """Parse a CPU core-id list 'a,b,c' into [int]; '' / None -> None (no pinning)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        ids = [int(part.strip()) for part in text.split(',') if part.strip() != '']
        return ids or None
    except Exception:
        return None


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
    global parser, BENCHMARK_MODE, BENCHMARK_LOOP, INFERENCE_DEVICE, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD, DEBUG_MODE
    global MODEL_PATH, ONNX_MODEL_PATH, VIDEO_FILE_PATH, VIDEO_FILE_PATHS, IMG_SIZE, FPS_TEXT_SIZE, LABEL_TEXT_SIZE, OVERLAY_ENABLED, OVERLAY_TEXT_COLOR, SAVE_DEBUG_FRAMES, MAX_INFERENCE_INSTANCES, NPU_CORE_ASSIGNMENT, CLASSES, MODEL_LABELS_FILE_PATH
    global DETECTION_BOX_COLOR, DETECTION_LABEL_COLOR, DETECTION_LABEL_BACKGROUND_COLOR, DETECTION_BOX_THICKNESS, DETECTION_LABEL_TEXT_SIZE, DETECTION_LABEL_TEXT_THICKNESS
    global CPU50_THREADS, CPU50_AFFINITY, MAX_DETECTIONS_PER_FRAME
    global MNN_MODEL_PATH, MNN_PRECISION, MNN_BACKEND, HAILO8_MODEL_PATH

    # Clear and re-read the config file
    if is_reload:
        parser.clear()
    parser.read(config_ini)
    
    # Load all configuration values
    BENCHMARK_MODE = parser.getboolean("MODE", "benchmark_mode", fallback=False)
    # Benchmark Loop: same as benchmark mode but the selected video(s) replay on a loop until the
    # user stops processing (continuous inference). Implies benchmark_mode.
    BENCHMARK_LOOP = parser.getboolean("MODE", "benchmark_loop", fallback=False)
    if BENCHMARK_LOOP:
        BENCHMARK_MODE = True
    INFERENCE_DEVICE = parser.get("INFERENCE", "inference_device", fallback="RKNPU-AUTO").strip().upper()
    # Backward-compat: the old "GPU" device is now "GPU-OPENCV-OPENCL" (a future GPU-MNN mode
    # will be a separate device), so legacy configs keep working.
    if INFERENCE_DEVICE == "GPU":
        INFERENCE_DEVICE = "GPU-OPENCV-OPENCL"
    # Backward-compat: the old "NPU" mode (which used a separate npu_core_assignment setting) is now
    # two explicit device modes, RKNPU-AUTO / RKNPU-DISTRIBUTED. Legacy "NPU"/"RKNPU" -> RKNPU-AUTO.
    if INFERENCE_DEVICE in ("NPU", "RKNPU"):
        INFERENCE_DEVICE = "RKNPU-AUTO"
    ROCKCHIP_TARGET = parser.get("INFERENCE", "rockchip_target", fallback="rk3588").strip().lower()
    OBJ_THRESHOLD = parser.getfloat("INFERENCE", "obj_threshold", fallback=0.25)
    NMS_THRESHOLD = parser.getfloat("INFERENCE", "nms_threshold", fallback=0.45)
    DEBUG_MODE = parser.getboolean("INFERENCE", "debug_mode", fallback=False)
    # Robustness guard: a frame yielding more detections than this is treated as corrupted model
    # output (the OpenCV-OpenCL/Mali path intermittently mis-computes some models, flooding the
    # frame with saturated false boxes) and dropped. 0 disables the guard. Real frames for this
    # app have only a handful of detections, so 50 is comfortably above legitimate counts.
    MAX_DETECTIONS_PER_FRAME = parser.getint("INFERENCE", "max_detections_per_frame", fallback=50)
    
    # Load paths
    model_rknn_cfg = parser.get("PATHS", "model_rknn", fallback="assets/models/yolov11n.rknn")
    MODEL_PATH = os.path.join(BASE_DIR, model_rknn_cfg)
    
    model_onnx_cfg = parser.get("PATHS", "model_onnx", fallback="assets/models/yolov11n.onnx")
    ONNX_MODEL_PATH = os.path.join(BASE_DIR, model_onnx_cfg)

    model_mnn_cfg = parser.get("PATHS", "model_mnn", fallback="assets/models/detectS2.mnn")
    MNN_MODEL_PATH = os.path.join(BASE_DIR, model_mnn_cfg)

    model_hailo8_cfg = parser.get("PATHS", "model_hailo8", fallback="assets/models/detectS2.hef")
    HAILO8_MODEL_PATH = os.path.join(BASE_DIR, model_hailo8_cfg)

    # Load image settings
    img_width = parser.getint("IMAGE", "img_width", fallback=640)
    img_height = parser.getint("IMAGE", "img_height", fallback=640)
    IMG_SIZE = (img_width, img_height)
    
    FPS_TEXT_SIZE = parser.getfloat("IMAGE", "fps_text_size", fallback=0.5)
    LABEL_TEXT_SIZE = parser.getfloat("IMAGE", "label_text_size", fallback=0.4)
    OVERLAY_ENABLED = parser.getboolean("IMAGE", "show_overlay", fallback=True)
    OVERLAY_TEXT_COLOR = _parse_color(parser.get("IMAGE", "overlay_text_color", fallback="255,255,255"), (255, 255, 255))
    SAVE_DEBUG_FRAMES = parser.getboolean("IMAGE", "save_debug_frames", fallback=False)

    # Load detection styling settings
    DETECTION_BOX_COLOR = _parse_color(parser.get("DETECTION", "box_color", fallback="0,0,255"), (0, 0, 255))
    DETECTION_LABEL_COLOR = _parse_color(parser.get("DETECTION", "label_text_color", fallback="255,255,255"), (255, 255, 255))
    DETECTION_LABEL_BACKGROUND_COLOR = _parse_color(parser.get("DETECTION", "label_background_color", fallback="0,0,255"), (0, 0, 255))
    DETECTION_BOX_THICKNESS = parser.getint("DETECTION", "box_thickness", fallback=2)
    DETECTION_LABEL_TEXT_SIZE = parser.getfloat("DETECTION", "label_text_size", fallback=0.45)
    DETECTION_LABEL_TEXT_THICKNESS = parser.getint("DETECTION", "label_text_thickness", fallback=2)
    
    # Number of parallel inference instances (one per camera/stream)
    MAX_INFERENCE_INSTANCES = parser.getint("INFERENCE", "max_inference_instances", fallback=3)
    # Core assignment is now encoded in the device mode (RKNPU-AUTO = all instances on Core 0;
    # RKNPU-DISTRIBUTED = instance N -> RKNN core N), not a separate setting. Derive it here for the
    # multi-core distribution logic in the web video/camera workers.
    NPU_CORE_ASSIGNMENT = "distributed" if INFERENCE_DEVICE == "RKNPU-DISTRIBUTED" else "auto"

    # ---- CPU-50% mode (inference_device = CPU-50%) ----
    # Like CPU mode, but capped so it does NOT saturate all 8 cores — the device stays usable.
    # onnxruntime intra-op thread cap for the CPU engine (the "don't saturate" dial).
    CPU50_THREADS = parser.getint("INFERENCE", "cpu50_threads", fallback=4)
    # Core affinity for the CPU engine (RK3588: A76 big cluster = 4,5,6,7), so the A55 little
    # cores stay free for the OS/desktop. "" -> no pinning.
    CPU50_AFFINITY = _parse_core_list(parser.get("INFERENCE", "cpu50_affinity", fallback="4,5,6,7"))

    # ---- GPU-MNN mode (inference_device = GPU-MNN): .mnn model on the Mali-G610 via MNN + OpenCL ----
    # mnn_precision: low = fp16 (default, fastest) | high = fp32 (bit-exact to CPU) | normal.
    MNN_PRECISION = parser.get("INFERENCE", "mnn_precision", fallback="low").strip().lower()
    # mnn_backend: OPENCL (Mali GPU); CPU also accepted (MNN on CPU) for debugging.
    MNN_BACKEND = parser.get("INFERENCE", "mnn_backend", fallback="OPENCL").strip().upper()

    # Load benchmark video paths (one per inference instance, indexed benchmark_video_0..N-1)
    VIDEO_FILE_PATHS = []
    for _i in range(MAX_INFERENCE_INSTANCES):
        _cfg = parser.get("PATHS", f"benchmark_video_{_i}", fallback="").strip()
        if _cfg:
            VIDEO_FILE_PATHS.append(os.path.join(BASE_DIR, _cfg))
    if not VIDEO_FILE_PATHS:
        _fallback = parser.get("PATHS", "benchmark_video", fallback="assets/videos/benchmark.mp4").strip()
        VIDEO_FILE_PATHS.append(os.path.join(BASE_DIR, _fallback))
    VIDEO_FILE_PATH = VIDEO_FILE_PATHS[0]
    
    # Labels: each model folder (assets/models/<name>/) carries its own classes.txt, so labels
    # follow the selected model. If that file is absent, fall back to CLASSES.default_labels.
    MODEL_LABELS_FILE_PATH = None
    labels = []

    # Pick the model file for the ACTIVE inference device, then read its folder's classes.txt.
    if INFERENCE_DEVICE.startswith("RKNPU"):
        _active_model = MODEL_PATH
    elif INFERENCE_DEVICE == "NPU-HAILO8":
        _active_model = HAILO8_MODEL_PATH
    elif INFERENCE_DEVICE == "GPU-MNN":
        _active_model = MNN_MODEL_PATH
    else:
        _active_model = ONNX_MODEL_PATH
    if _active_model:
        _folder_classes = os.path.join(os.path.dirname(_active_model), "classes.txt")
        if os.path.isfile(_folder_classes):
            MODEL_LABELS_FILE_PATH = _folder_classes

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

    # Labels source (helps debug wrong class names, e.g. "class_2"): show whether a per-model
    # classes.txt was used or the CLASSES.default_labels fallback, plus the resolved class list.
    if MODEL_LABELS_FILE_PATH is not None:
        logging.info(f"[CONFIG] Labels: loaded {len(CLASSES)} class(es) from {MODEL_LABELS_FILE_PATH}: {list(CLASSES)}")
    else:
        _amf = os.path.dirname(_active_model) if _active_model else "?"
        logging.info(f"[CONFIG] Labels: no classes.txt in {_amf}; using default_labels: {list(CLASSES)}")

    action = "reloaded" if is_reload else "loaded"
    debug_status = "ON" if DEBUG_MODE else "OFF"
    _videos = ", ".join(os.path.basename(v) for v in VIDEO_FILE_PATHS)
    _labels_file = os.path.basename(MODEL_LABELS_FILE_PATH) if MODEL_LABELS_FILE_PATH else None
    # Log EVERY reloaded variable so a web "Save" shows the full effective config.
    logging.info(
        f"[CONFIG] Configuration {action}:\n"
        f"  benchmark_mode = {BENCHMARK_MODE}\n"
        f"  benchmark_loop = {BENCHMARK_LOOP}\n"
        f"  inference_device = {INFERENCE_DEVICE}\n"
        f"  rockchip_target = {ROCKCHIP_TARGET}\n"
        f"  obj_threshold = {OBJ_THRESHOLD}\n"
        f"  nms_threshold = {NMS_THRESHOLD}\n"
        f"  max_detections_per_frame = {MAX_DETECTIONS_PER_FRAME}\n"
        f"  debug = {debug_status}\n"
        f"  max_inference_instances = {MAX_INFERENCE_INSTANCES}\n"
        f"  npu_core_assignment = {NPU_CORE_ASSIGNMENT!r}\n"
        f"  cpu50_threads = {CPU50_THREADS}\n"
        f"  cpu50_affinity = {CPU50_AFFINITY}\n"
        f"  mnn_precision = {MNN_PRECISION}\n"
        f"  mnn_backend = {MNN_BACKEND}\n"
        f"  model_rknn = {os.path.basename(MODEL_PATH)}\n"
        f"  model_onnx = {os.path.basename(ONNX_MODEL_PATH)}\n"
        f"  model_mnn = {os.path.basename(MNN_MODEL_PATH)}\n"
        f"  model_hailo8 = {os.path.basename(HAILO8_MODEL_PATH)}\n"
        f"  model_labels = {_labels_file}\n"
        f"  classes ({len(CLASSES)}) = [{', '.join(CLASSES)}]\n"
        f"  benchmark_videos = [{_videos}]\n"
        f"  img_size = {IMG_SIZE}\n"
        f"  overlay = {'ON' if OVERLAY_ENABLED else 'OFF'}\n"
        f"  overlay_text_color = {OVERLAY_TEXT_COLOR}\n"
        f"  fps_text_size = {FPS_TEXT_SIZE}\n"
        f"  label_text_size = {LABEL_TEXT_SIZE}\n"
        f"  save_debug_frames = {SAVE_DEBUG_FRAMES}\n"
        f"  detection_box_color = {DETECTION_BOX_COLOR}\n"
        f"  detection_label_text_color = {DETECTION_LABEL_COLOR}\n"
        f"  detection_label_background_color = {DETECTION_LABEL_BACKGROUND_COLOR}\n"
        f"  detection_box_thickness = {DETECTION_BOX_THICKNESS}\n"
        f"  detection_label_text_size = {DETECTION_LABEL_TEXT_SIZE}\n"
        f"  detection_label_text_thickness = {DETECTION_LABEL_TEXT_THICKNESS}"
    )
    
    # Return updated config for convenience
    return {
        'benchmark_mode': BENCHMARK_MODE,
        'benchmark_loop': BENCHMARK_LOOP,
        'inference_device': INFERENCE_DEVICE,
        'debug_mode': DEBUG_MODE,
        'overlay_enabled': OVERLAY_ENABLED,
        'max_inference_instances': MAX_INFERENCE_INSTANCES,
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
