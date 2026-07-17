# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""web_camera_processing.py
Camera processing with web interface integration.
Each camera runs in its own thread so all NPU cores work in parallel.
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
    from ..utils.my_htop import get_npu_info, get_gpu_info, get_soc_temp
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
    # CPU-50%: pin this worker thread to the engine's core set (A76 big cluster), matching the
    # benchmark-video worker. This thread participates in onnxruntime's intra-op pool, so pinning it
    # keeps CPU inference off the A55 little cores. No-op for other modes (affinity None).
    _aff = getattr(engine, 'cpu_affinity', None)
    if _aff and hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, set(int(c) for c in _aff))
            logger.info(f"Camera {idx}: CPU worker pinned to cores {sorted(_aff)}")
        except Exception:
            pass

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
        boxes, classes, scores, _ = engine.detect_objects(frame, stream_idx=idx, frame_idx=total_frames + 1)
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
        # Hailo metrics per frame (no device I/O): DEVICE occupancy over a trailing ~1 s window across
        # ALL streams (same value the live monitor shows -> report matches live) + this stream's last
        # device infer latency (ms). Matches the benchmark path.
        hailo_pct = 0.0
        hailo_infer_ms = 0.0
        hailo_temp_c = 0.0
        hailo_power_w = 0.0
        if getattr(engine, 'platform', None) == 'hailo':
            from ..processing.hailo_executor import hailo_device_occupancy, hailo_env
            hailo_infer_ms = float(getattr(getattr(engine, 'model', None), 'last_infer_s', 0.0) or 0.0) * 1000.0
            hailo_pct = hailo_device_occupancy()
            _t, _p = hailo_env()   # cached temp/power (monitor-sampled); no per-frame device I/O
            hailo_temp_c = round(_t, 1) if _t is not None else 0.0
            hailo_power_w = round(_p, 2) if _p is not None else 0.0
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
            'hailo_usage_percent': round(hailo_pct, 1),
            'hailo_infer_ms': round(hailo_infer_ms, 2),
            'hailo_temp_c': hailo_temp_c,
            'hailo_power_w': hailo_power_w,
            'rk3588_temp_c': get_soc_temp(),
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
    INFERENCE_TIMEOUT_MINUTES = _cfg.INFERENCE_TIMEOUT_MINUTES

    video_manager = get_video_stream_manager()
    logger = get_web_logger()

    context = pyudev.Context()
    discovered = []
    raw_nodes = []
    for device in context.list_devices(subsystem='video4linux'):
        devnode = device.device_node
        if not (devnode and devnode.startswith('/dev/video')):
            continue
        try:
            node = int(devnode.replace('/dev/video', ''))
        except ValueError:
            continue
        raw_nodes.append(node)
        # UVC webcams expose several /dev/video nodes: the true capture node
        # (ID_V4L_CAPABILITIES contains 'capture') plus metadata/non-capture nodes.
        # Keep only capture-capable nodes so a metadata node does not consume a
        # MAX_INFERENCE_INSTANCES slot and hide a real camera (the original 2-of-3 bug).
        capabilities = device.properties.get('ID_V4L_CAPABILITIES', '')
        if 'capture' not in capabilities:
            continue
        id_path = device.properties.get('ID_PATH', '')
        model = (device.properties.get('ID_MODEL', '')
                 or device.properties.get('ID_V4L_PRODUCT', '')
                 or 'Camera').replace('_', ' ')
        if id_path.startswith('platform-'):
            short_port = id_path[len('platform-'):].replace('.auto', '')
        else:
            short_port = id_path or 'unknown'
        discovered.append({'node': node, 'path': id_path, 'label': f"{model} ({short_port})"})

    # Fallback: if the ID_V4L_CAPABILITIES property is unavailable on this system and the
    # filter dropped everything, fall back to every /dev/video node so we never regress to
    # "no cameras" on hardware that does not populate the capability string.
    if not discovered and raw_nodes:
        logger.warning("No capture-capable nodes reported by udev; falling back to all /dev/video nodes.")
        discovered = [{'node': n, 'path': '', 'label': f"Camera ({n})"} for n in raw_nodes]

    # Order by physical USB port (ID_PATH) so each camera number is stable and reproducible
    # per port across reboots/replugs, instead of the non-deterministic udev enumeration order.
    discovered.sort(key=lambda d: (d['path'], d['node']))
    discovered = discovered[:MAX_INFERENCE_INSTANCES]

    cameras = []
    for info in discovered:
        cap = cv2.VideoCapture(info['node'])
        if cap.isOpened():
            cameras.append(cap)
            logger.info(f"Camera {len(cameras) - 1}: /dev/video{info['node']} -> {info['label']}")
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
    # RKNPU and GPU modes need one engine per camera: RKNPU to pin each stream to its
    # own NPU core; GPU because the worker threads otherwise share one engine's
    # letterbox helper/frame-label state, which corrupts boxes when the cameras have
    # different resolutions. For OpenCV/MNN the per-camera engines still reuse the one
    # process-cached Net/session, which the executor's _INFER_LOCK serializes.
    # CPU/CPU-50% (onnxruntime) are thread-safe, so they keep sharing one engine.
    per_stream_engines = (INFERENCE_DEVICE.startswith("RKNPU")
                          or INFERENCE_DEVICE.startswith("GPU")) and len(cameras) > 1
    try:
        if per_stream_engines:
            for idx in range(len(cameras)):
                core_id = (idx if NPU_CORE_ASSIGNMENT == "distributed" else 0) \
                    if INFERENCE_DEVICE.startswith("RKNPU") else None
                engine = create_yolo11_engine(INFERENCE_DEVICE, npu_core_id=core_id)
                yolo_engines.append(engine)
                if core_id is not None:
                    logger.info(f"YOLO11 engine {idx} initialized for camera {idx} (RKNPU Core {core_id})")
                else:
                    logger.info(f"YOLO11 engine {idx} initialized for camera {idx} ({INFERENCE_DEVICE})")
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
        core_id = idx if (INFERENCE_DEVICE.startswith("RKNPU") and NPU_CORE_ASSIGNMENT == "distributed" and len(cameras) > 1) else (0 if INFERENCE_DEVICE.startswith("RKNPU") else None)
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
    run_start = time.time()

    timeout_s = INFERENCE_TIMEOUT_MINUTES * 60 if INFERENCE_TIMEOUT_MINUTES > 0 else 0
    if timeout_s:
        logger.info(f"Inference timeout armed: {INFERENCE_TIMEOUT_MINUTES} min")

    if web_server is None:
        try:
            while any(t.is_alive() for t in threads):
                if timeout_s and (time.time() - run_start >= timeout_s):
                    logger.info(f"Inference timeout reached ({INFERENCE_TIMEOUT_MINUTES} min) - stopping processing")
                    stop_event.set()
                    break
                    
                for idx in range(len(cameras)):
                    frame = video_manager.get_latest_frame(camera_id=idx)
                    if frame is not None:
                        if not __import__('os').environ.get("CHAPV_HEADLESS"):
                            cv2.imshow(f"Camera {idx}", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    stop_event.set()
                    break
        except Exception as e:
            logger.warning(f"Display window error: {e}")
        finally:
            cv2.destroyAllWindows()
        stop_event.set()
    else:
        if timeout_s:
            while any(t.is_alive() for t in threads):
                if time.time() - run_start >= timeout_s:
                    logger.info(f"Inference timeout reached ({INFERENCE_TIMEOUT_MINUTES} min) - stopping processing")
                    web_server.processing_active = False
                    break
                for t in threads:
                    t.join(timeout=0.5)

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
