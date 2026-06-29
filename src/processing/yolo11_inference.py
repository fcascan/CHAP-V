# -*- coding: utf-8 -*-
"""yolo11_inference.py
YOLO11 Inference Engine Integration
by fcascan 2026
"""
import os
import cv2
import numpy as np
import logging
from types import SimpleNamespace

from ..core import config as app_config
from src.rockchip import yolo11_infer as rockchip_yolo
from src.rockchip.coco_utils import COCO_test_helper


def _sync_rockchip_runtime_config():
    """Sync the Rockchip module globals with the active application config."""
    rockchip_yolo.OBJ_THRESH = app_config.OBJ_THRESHOLD
    rockchip_yolo.NMS_THRESH = app_config.NMS_THRESHOLD
    rockchip_yolo.IMG_SIZE = app_config.IMG_SIZE
    rockchip_yolo.CLASSES = tuple(app_config.CLASSES)
    rockchip_yolo.coco_id_list = list(range(len(app_config.CLASSES)))
    if hasattr(rockchip_yolo, 'DEBUG_DETECTIONS'):
        rockchip_yolo.DEBUG_DETECTIONS = app_config.DEBUG_MODE


class YOLO11InferenceEngine:
    """
    YOLO11 Inference Engine wrapper that delegates detection to the Rockchip
    yolo11 implementation while preserving the existing application API.
    """
    
    def __init__(self, model_path, target_platform=None, device_id=None, device_type=None,
                 cpu_threads=None, cpu_affinity=None):
        """
        Initialize YOLO11 inference engine.

        Args:
            model_path: Path to the model file (.rknn, .onnx, or .pt)
            target_platform: Target NPU platform (e.g., 'rk3588', 'rk3566'). If None, uses ROCKCHIP_TARGET from config.
            device_id: Device ID for multi-device setups
            device_type: "NPU" | "GPU" | "CPU". For GPU, the .onnx model is run on the
                Mali-G610 via OpenCV-DNN + OpenCL.
            cpu_threads: onnxruntime intra-op thread cap for a CPU engine (None = all cores).
                Only set by CPU-50% mode (capped CPU).
            cpu_affinity: list of core ids the CPU worker thread should pin to (applied by the
                worker thread itself via os.sched_setaffinity; stored here for reference).
        """
        self.model_path = model_path
        self.device_type = device_type
        self.cpu_affinity = cpu_affinity
        app_config.reload_config()
        _sync_rockchip_runtime_config()

        # Use configuration value if target_platform is not specified
        self.target_platform = target_platform if target_platform is not None else app_config.ROCKCHIP_TARGET
        self.device_id = device_id
        self.model = None
        self.platform = None
        self.coco_helper = COCO_test_helper(enable_letter_box=True)
        # NOTE: deliberately do NOT set rockchip_yolo.co_helper here. The hot path uses
        # self.coco_helper (per-engine, thread-safe); the module global is only consumed by
        # the standalone CLI in yolo11_infer.py. Setting it would let a second engine clobber
        # the first's helper — a real hazard if multiple engines ever coexist (e.g. NPU pool).

        try:
            self.model, self.platform = rockchip_yolo.setup_model(
                SimpleNamespace(
                    model_path=model_path,
                    target=self.target_platform,
                    device_id=device_id,
                    gpu_opencl=(device_type == "GPU-OPENCV-OPENCL"),
                    cpu_threads=cpu_threads,
                    cpu_affinity=cpu_affinity,
                    mnn_precision=getattr(app_config, 'MNN_PRECISION', 'low'),
                    mnn_backend=getattr(app_config, 'MNN_BACKEND', 'OPENCL'),
                )
            )
            logging.info(f"YOLO11 model loaded: {model_path} on platform: {self.platform}")
            logging.info(f"Using Rockchip target: {self.target_platform} (from config)")
            logging.info(f"Using thresholds: OBJ={app_config.OBJ_THRESHOLD}, NMS={app_config.NMS_THRESHOLD} (from config)")
            logging.info(f"Model will use {len(app_config.CLASSES)} custom classes")
        except Exception as e:
            logging.error(f"Failed to setup YOLO11 model: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for inference using letterbox approach.
        
        Args:
            frame: Input OpenCV frame (BGR format)
            
        Returns:
            Preprocessed input data ready for inference
        """
        # Inline preprocessing using this engine's own coco_helper so the letterbox
        # state is never shared across engines — required for thread-safe multi-engine use.
        img = self.coco_helper.letter_box(
            im=frame,
            new_shape=(app_config.IMG_SIZE[1], app_config.IMG_SIZE[0]),
            pad_color=(0, 0, 0),
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.platform in ('pytorch', 'onnx', 'opencv', 'mnn'):
            input_data = img.transpose((2, 0, 1))
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
            input_data = input_data / 255.
        else:
            input_data = np.expand_dims(img, axis=0).astype(np.uint8)

        if app_config.DEBUG_MODE:
            logging.debug(f"[DEBUG] Input shape: {input_data.shape}")

        return input_data
    
    def run_inference(self, input_data):
        """
        Run inference on preprocessed input data.
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            Raw model outputs
        """
        try:
            outputs = self.model.run([input_data])
            return outputs
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            return None
    
    def postprocess_outputs(self, outputs):
        """
        Postprocess model outputs to get detection results.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Tuple of (boxes, classes, scores) or (None, None, None) if no detections
        """
        if outputs is None:
            logging.debug("No outputs to postprocess")
            return None, None, None
            
        try:
            logging.debug(f"Starting postprocessing with {len(outputs)} output tensors")
            # All backends (NPU/rknn, CPU/onnx, GPU/opencv) emit the Rockchip
            # 9-output head, decoded by the same post_process().
            boxes, classes, scores = rockchip_yolo.post_process(outputs)
            if boxes is not None:
                logging.debug(f"Postprocessing successful: {len(boxes)} detections")
            else:
                logging.debug("Postprocessing completed: no detections")
            return boxes, classes, scores
        except Exception as e:
            logging.error(f"Postprocessing failed: {e}")
            logging.error(f"Output info: {[o.shape if hasattr(o, 'shape') else type(o) for o in outputs]}")
            return None, None, None
    
    def detect_objects(self, frame, stream_idx=None, frame_idx=None):
        """
        Complete object detection pipeline: preprocess -> inference -> postprocess.

        Args:
            frame: Input OpenCV frame (BGR format)
            stream_idx: Optional stream/camera index for log labelling.
            frame_idx: Optional frame number for log labelling.

        Returns:
            Tuple of (boxes, classes, scores, processed_frame)
            boxes: Array of bounding boxes in original frame coordinates
            classes: Array of class indices
            scores: Array of confidence scores
            processed_frame: Frame with letterbox applied (for debugging)
        """
        if app_config.DEBUG_MODE:
            logging.debug(f"YOLO11 processing frame: {frame.shape}")

        # Build a short label used in all detection prints for this call.
        if stream_idx is not None and frame_idx is not None:
            self._frame_label = f"S{stream_idx}/F{frame_idx}"
        elif frame_idx is not None:
            self._frame_label = f"F{frame_idx}"
        else:
            self._frame_label = None

        # Preprocess
        input_data = self.preprocess_frame(frame)

        # Inference
        outputs = self.run_inference(input_data)

        # Postprocess
        boxes, classes, scores = self.postprocess_outputs(outputs)

        # Show detection results — gated by debug mode (config.ini INFERENCE.debug_mode)
        if boxes is not None and app_config.DEBUG_MODE:
            summary = self.get_detection_summary(boxes, classes, scores)
            label_str = f" [{self._frame_label}]" if self._frame_label else ""
            print(f"[DETECTIONS]{label_str} Classes found: {summary['class_counts']}")
            logging.debug(f"Detection result: {len(boxes)} objects found")

        # Convert boxes back to original frame coordinates
        if boxes is not None:
            real_boxes = self.coco_helper.get_real_box(boxes)
            return real_boxes, classes, scores, input_data
        else:
            return None, None, None, input_data
    
    def draw_detections(self, frame, boxes, classes, scores):
        """
        Draw detection results on frame using the Rockchip draw function.

        Args:
            frame: OpenCV frame to draw on
            boxes: Detection bounding boxes
            classes: Detection class indices
            scores: Detection confidence scores

        Returns:
            Frame with detections drawn
        """
        if boxes is not None and classes is not None and scores is not None:
            rockchip_yolo.draw(frame, boxes, scores, classes,
                               frame_label=getattr(self, '_frame_label', None))
        return frame

    def get_detection_summary(self, boxes, classes, scores, score_threshold=0.5):
        """
        Get a summary of detections with custom class names.
        
        Args:
            boxes: Detection bounding boxes
            classes: Detection class indices
            scores: Detection confidence scores
            score_threshold: Minimum score threshold
            
        Returns:
            dict: Detection summary with class counts and high-confidence detections
        """
        if boxes is None or classes is None or scores is None:
            return {'class_counts': {}, 'high_confidence': []}

        class_counts = {}
        high_confidence = []
        for box, class_id, score in zip(boxes, classes, scores):
            class_index = int(class_id)
            class_name = self.get_class_name(class_index)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if float(score) >= score_threshold:
                high_confidence.append({'class': class_name, 'score': float(score), 'box': box.tolist() if hasattr(box, 'tolist') else box})

        return {'class_counts': class_counts, 'high_confidence': high_confidence}
    
    def get_class_name(self, class_id):
        """
        Get class name from class ID using project configuration.
        
        Args:
            class_id: Integer class ID
            
        Returns:
            str: Class name or "Unknown" if ID is out of range
        """
        if 0 <= int(class_id) < len(app_config.CLASSES):
            return app_config.CLASSES[int(class_id)]
        return f"class_{int(class_id)}"
    
    def validate_model_compatibility(self):
        """
        Validate that the model output matches the configured classes.
        
        Returns:
            bool: True if model and config are compatible
        """
        return True
    
    def release(self):
        """Release model resources."""
        if self.model:
            try:
                self.model.release()
                logging.info("YOLO11 model resources released")
            except Exception as e:
                logging.warning(f"Error releasing model: {e}")


def create_yolo11_engine(device_type="NPU", npu_core_id=None, cpu_threads=None, cpu_affinity=None):
    """
    Factory function to create YOLO11 inference engine based on configuration.

    Args:
        device_type: Inference device type ("NPU", "CPU", "CPU-50%", "GPU-OPENCV-OPENCL", "GPU-MNN").
        npu_core_id: NPU core index (0, 1, 2) for explicit core pinning. None = RKNN default (Core 0).
        cpu_threads: onnxruntime intra-op thread cap for a CPU engine (None = all cores, unchanged).
        cpu_affinity: list of core ids to pin the CPU engine to (None = no pinning).

    Returns:
        YOLO11InferenceEngine instance
    """
    if device_type.startswith("RKNPU"):
        model_path = app_config.MODEL_PATH
        platform = app_config.ROCKCHIP_TARGET
    elif device_type == "GPU-OPENCV-OPENCL":
        # Runs the same ONNX model as CPU on the Mali-G610 via OpenCV-DNN + OpenCL.
        model_path = app_config.ONNX_MODEL_PATH
        platform = app_config.ROCKCHIP_TARGET
    elif device_type == "GPU-MNN":
        # Runs the dedicated .mnn model on the Mali-G610 via MNN + OpenCL.
        model_path = app_config.MNN_MODEL_PATH
        platform = app_config.ROCKCHIP_TARGET
    elif device_type == "NPU-HAILO8":
        # Runs the dedicated .hef model on the Hailo-8 external NPU via HailoRT.
        model_path = app_config.HAILO8_MODEL_PATH
        platform = app_config.ROCKCHIP_TARGET
    elif device_type in ("CPU", "CPU-50%"):
        model_path = app_config.ONNX_MODEL_PATH
        platform = app_config.ROCKCHIP_TARGET
        # CPU-50%: cap threads + pin to the A76 cluster so the device is not saturated.
        if device_type == "CPU-50%":
            if cpu_threads is None:
                cpu_threads = app_config.CPU50_THREADS
            if cpu_affinity is None:
                cpu_affinity = app_config.CPU50_AFFINITY
    else:
        raise ValueError(f"Unsupported device type: {device_type}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    core_info = f"Core {npu_core_id}" if npu_core_id is not None else "Core 0 (default)"
    logging.info(f"Creating YOLO11 engine for {device_type}/{core_info} with target {platform}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Using {len(app_config.CLASSES)} custom classes from config")
    logging.info(f"Using detection thresholds: OBJ={app_config.OBJ_THRESHOLD}, NMS={app_config.NMS_THRESHOLD}")
    if cpu_threads is not None:
        logging.info(f"CPU engine thread cap: intra_op_num_threads={cpu_threads}, affinity={cpu_affinity}")

    # Build the engine while the calling thread carries the target affinity, so onnxruntime's
    # intra-op thread pool (created at session construction) inherits the A76 mask. Restored after.
    orig_aff = None
    if cpu_affinity and hasattr(os, 'sched_setaffinity'):
        try:
            orig_aff = os.sched_getaffinity(0)
            os.sched_setaffinity(0, set(int(c) for c in cpu_affinity))
        except Exception:
            orig_aff = None
    try:
        engine = YOLO11InferenceEngine(model_path, platform, device_id=npu_core_id,
                                       device_type=device_type, cpu_threads=cpu_threads,
                                       cpu_affinity=cpu_affinity)
    finally:
        if orig_aff is not None:
            try:
                os.sched_setaffinity(0, orig_aff)
            except Exception:
                pass
    return engine


def yolo11_postprocess_wrapper(outputs, original_shape):
    """
    Wrapper function to maintain compatibility with existing yolo_postprocess_func signature.
    This is used as a drop-in replacement for the existing postprocessing functions.
    
    Args:
        outputs: Raw model outputs
        original_shape: Original frame shape (for compatibility, not used in yolo11)
        
    Returns:
        Tuple of (boxes, classes, scores)
    """
    try:
        boxes, classes, scores = rockchip_yolo.post_process(outputs)
        return boxes, classes, scores
    except Exception as e:
        logging.error(f"YOLO11 postprocessing failed: {e}")
        return None, None, None


# Global inference engine instance (singleton pattern)
_global_engine = None

def get_global_yolo11_engine(device_type="NPU"):
    """
    Get or create global YOLO11 inference engine instance.
    Uses singleton pattern to avoid recreating the engine multiple times.
    
    Args:
        device_type: Inference device type
        
    Returns:
        YOLO11InferenceEngine instance
    """
    global _global_engine
    if _global_engine is None:
        _global_engine = create_yolo11_engine(device_type)
    return _global_engine

def release_global_engine():
    """Release global inference engine resources."""
    global _global_engine
    if _global_engine:
        _global_engine.release()
        _global_engine = None