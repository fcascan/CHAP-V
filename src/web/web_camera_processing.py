# -*- coding: utf-8 -*-
"""web_camera_processing.py
Camera processing with web interface integration.
Each camera runs in its own thread so all NPU cores work in parallel.
by fcascan 2025
"""
import cv2
import time
import logging
import threading
import os
import pyudev
from ..core.config import *
from ..utils.frame_overlay import calculate_recent_average_ms, calculate_recent_fps, draw_processing_overlay
from ..utils.csv_analysis import save_instance_performance_data
from .video_integration import get_video_stream_manager
from .console_integration import get_web_logger
from ..processing.yolo11_inference import create_yolo11_engine

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    from ..utils.my_htop import get_npu_info, get_gpu_info
    _HTOP_AVAILABLE = True
except ImportError:
    _HTOP_AVAILABLE = False


def _read_system_stats():
    """Read current CPU/NPU/GPU usage. Returns (cpu_pct, npu_loads, gpu_pct)."""
    cpu_pct = psutil.cpu_percent() if _PSUTIL_AVAILABLE else 0.0
    if _HTOP_AVAILABLE:
        npu_loads, _ = get_npu_info()
        gpu_load, _ = get_gpu_info()
        gpu_pct = gpu_load if gpu_load is not None else 0
        if not npu_loads:
            npu_loads = [0, 0, 0]
    else:
        npu_loads, gpu_pct = [0, 0, 0], 0
    return cpu_pct, npu_loads, gpu_pct


def _camera_worker(idx, cap, engine, video_manager, processing_active_fn, output_dir, logger,
                   results_dir=None, run_timestamp=None, npu_core_id=None):
    """Process one camera stream until stopped or camera fails critically."""
    display_timestamps = []
    inftime_buf = []
    failure_counter = 0
    total_frames = 0
    csv_rows = []

    while processing_active_fn():
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            failure_counter += 1
            logger.error(f"Failed to read frame from camera {idx}.")
            if failure_counter >= 10:
                logger.error(f"Critical: Camera {idx} failed too many times. Stopping stream.")
                break
            continue

        failure_counter = 0

        if DEBUG_MODE:
            logging.debug(f"[DEBUG] Camera {idx} processing frame {total_frames + 1}")

        start_inf = time.time()
        boxes, classes, scores, _ = engine.detect_objects(frame)
        inf_time = time.time() - start_inf
        total_frame_ms = (time.time() - frame_start) * 1000

        inftime_buf.append(inf_time)
        if len(inftime_buf) > 30:
            inftime_buf.pop(0)
        avg_ms = calculate_recent_average_ms(inftime_buf[-30:])

        now = time.time()
        display_timestamps.append(now)
        if len(display_timestamps) > 30:
            display_timestamps.pop(0)
        fps = calculate_recent_fps(display_timestamps[-30:])

        try:
            cpu_pct, npu_loads, gpu_pct = _read_system_stats()
        except Exception:
            cpu_pct, npu_loads, gpu_pct = 0.0, [0, 0, 0], 0
        csv_rows.append({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((now % 1) * 1000):03d}",
            'frame_number': total_frames + 1,
            'inference_time_ms': round(inf_time * 1000, 2),
            'total_frame_time_ms': round(total_frame_ms, 2),
            'cpu_usage_percent': round(cpu_pct, 1),
            'npu_core0_percent': npu_loads[0] if len(npu_loads) > 0 else 0,
            'npu_core1_percent': npu_loads[1] if len(npu_loads) > 1 else 0,
            'npu_core2_percent': npu_loads[2] if len(npu_loads) > 2 else 0,
            'gpu_usage_percent': gpu_pct,
            'fps_actual': round(fps, 2),
            'detections_count': len(boxes) if boxes is not None else 0,
        })

        disp = frame.copy()
        draw_processing_overlay(
            disp, OVERLAY_ENABLED,
            f"Camera {idx} - Frame: {total_frames + 1}",
            inference_time_ms=avg_ms, fps_value=fps,
            text_size=FPS_TEXT_SIZE, text_color=OVERLAY_TEXT_COLOR,
        )

        if boxes is not None and classes is not None and scores is not None:
            engine.draw_detections(disp, boxes, classes, scores)
            if DEBUG_MODE:
                for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                    logging.debug(f"[DEBUG] Camera {idx} det {i+1}: {engine.get_class_name(cls)} ({score:.3f})")

        if SAVE_DEBUG_FRAMES:
            cv2.imwrite(os.path.join(output_dir, f"inference_output_cam{idx}.jpg"), disp)
        video_manager.update_frame(disp, camera_id=idx)
        total_frames += 1

    if csv_rows and results_dir and run_timestamp:
        model_name = os.path.basename(engine.model_path) if hasattr(engine, 'model_path') else None
        save_instance_performance_data(
            csv_rows, results_dir, INFERENCE_DEVICE, run_timestamp, f"cam{idx}", logger,
            npu_core_id=npu_core_id, model_name=model_name,
            camera_index=idx,
        )


def process_cameras_web(yolo_postprocess_func=None, web_server=None):
    """Process cameras — one thread per camera for parallel NPU execution."""
    from ..core import config as _cfg
    INFERENCE_DEVICE = _cfg.INFERENCE_DEVICE
    NPU_CORE_ASSIGNMENT = _cfg.NPU_CORE_ASSIGNMENT
    MAX_INFERENCE_INSTANCES = _cfg.MAX_INFERENCE_INSTANCES

    video_manager = get_video_stream_manager()
    logger = get_web_logger()

    context = pyudev.Context()
    video_devices = []
    for device in context.list_devices(subsystem='video4linux'):
        devnode = device.device_node
        if devnode and devnode.startswith('/dev/video'):
            try:
                video_devices.append(int(devnode.replace('/dev/video', '')))
            except ValueError:
                continue

    video_devices = sorted(set(video_devices))[:MAX_INFERENCE_INSTANCES]
    cameras = []
    for i in video_devices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(cap)
        else:
            cap.release()

    if not cameras:
        logger.error("No cameras detected, at least one camera is required.")
        return

    logger.info(f"Cameras detected: {len(cameras)}")
    video_manager.set_camera_count(len(cameras))

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "src", "processing", "results")
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    yolo_engines = []
    try:
        if INFERENCE_DEVICE == "NPU" and len(cameras) > 1:
            for idx in range(len(cameras)):
                core_id = idx if NPU_CORE_ASSIGNMENT == "distributed" else 0
                engine = create_yolo11_engine(INFERENCE_DEVICE, npu_core_id=core_id)
                yolo_engines.append(engine)
                logger.info(f"YOLO11 engine {idx} initialized for camera {idx} (NPU Core {core_id})")
                if DEBUG_MODE:
                    logging.debug(f"[DEBUG] Camera {idx} engine platform: {engine.platform}")
        else:
            engine = create_yolo11_engine(INFERENCE_DEVICE)
            yolo_engines = [engine]
            logger.info(f"YOLO11 engine initialized for {INFERENCE_DEVICE} inference")

        if web_server and yolo_engines:
            web_server.active_model_name = os.path.basename(yolo_engines[0].model_path)
            web_server.rknn_instance = yolo_engines[0].model if hasattr(yolo_engines[0].model, 'rknn') else None

    except Exception as e:
        logger.error(f"Failed to initialize YOLO11 engines: {e}")
        for cap in cameras:
            cap.release()
        return

    video_manager.start()

    logger.info("Live camera threat detection started")

    stop_event = threading.Event()
    processing_active = lambda: (not stop_event.is_set()) if web_server is None else web_server.processing_active

    threads = []
    for idx, cap in enumerate(cameras):
        engine = yolo_engines[idx if len(yolo_engines) > 1 else 0]
        core_id = idx if (INFERENCE_DEVICE == "NPU" and NPU_CORE_ASSIGNMENT == "distributed" and len(cameras) > 1) else (0 if INFERENCE_DEVICE == "NPU" else None)
        t = threading.Thread(
            target=_camera_worker,
            args=(idx, cap, engine, video_manager, processing_active, OUTPUT_DIR, logger),
            kwargs={'results_dir': RESULTS_DIR, 'run_timestamp': run_timestamp, 'npu_core_id': core_id},
            daemon=True,
            name=f"camera-{idx}",
        )
        threads.append(t)

    for t in threads:
        t.start()

    if web_server is None:
        try:
            while any(t.is_alive() for t in threads):
                for idx in range(len(cameras)):
                    frame = video_manager.get_latest_frame(camera_id=idx)
                    if frame is not None:
                        cv2.imshow(f"Camera {idx}", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    stop_event.set()
                    break
        except Exception as e:
            logger.warning(f"Display window error: {e}")
        finally:
            cv2.destroyAllWindows()
        stop_event.set()

    for t in threads:
        t.join()

    video_manager.stop()

    for engine in yolo_engines:
        try:
            engine.release()
            logger.info("YOLO11 engine resources released")
        except Exception as e:
            logger.warning(f"Error releasing YOLO11 engine: {e}")

    for cap in cameras:
        cap.release()

    logger.info("Camera Analysis Complete")
