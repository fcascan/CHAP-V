# -*- coding: utf-8 -*-
"""web_video_processing.py
Benchmark video processing with web interface integration.
Each video stream runs in its own thread so all NPU cores work in parallel.
by fcascan 2025
"""
import cv2
import time
import logging
import threading
import os
from ..core.config import *
from ..utils.frame_overlay import calculate_recent_average_ms, calculate_recent_fps, draw_processing_overlay
from .video_integration import get_video_stream_manager
from .console_integration import get_web_logger
from ..processing.yolo11_inference import create_yolo11_engine


def _stream_worker(idx, cap, engine, video_manager, processing_active_fn, output_dir, logger):
    """Process one benchmark video stream until it ends or processing is stopped."""
    display_timestamps = []
    inftime_buf = []
    total_frames = 0

    while processing_active_fn():
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Stream {idx}: video completed ({total_frames} frames).")
            break

        start_inf = time.time()
        boxes, classes, scores, _ = engine.detect_objects(frame)
        inf_time = time.time() - start_inf

        inftime_buf.append(inf_time)
        if len(inftime_buf) > 30:
            inftime_buf.pop(0)
        avg_ms = calculate_recent_average_ms(inftime_buf[-30:])

        now = time.time()
        display_timestamps.append(now)
        if len(display_timestamps) > 30:
            display_timestamps.pop(0)
        fps = calculate_recent_fps(display_timestamps[-30:])

        disp = frame.copy()
        draw_processing_overlay(
            disp, OVERLAY_ENABLED,
            f"Stream {idx} - Frame: {total_frames + 1}",
            inference_time_ms=avg_ms, fps_value=fps,
            text_size=FPS_TEXT_SIZE, text_color=OVERLAY_TEXT_COLOR,
        )

        if boxes is not None and classes is not None and scores is not None:
            engine.draw_detections(disp, boxes, classes, scores)
            if DEBUG_MODE:
                for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                    logging.debug(f"[DEBUG] Stream {idx} det {i+1}: {engine.get_class_name(cls)} ({score:.3f})")

        if SAVE_DEBUG_FRAMES:
            cv2.imwrite(os.path.join(output_dir, f"inference_output_stream{idx}.jpg"), disp)
        video_manager.update_frame(disp, camera_id=idx)
        total_frames += 1


def process_video_web(yolo_postprocess_func=None, web_server=None):
    """Process benchmark video files — one thread per stream for parallel NPU execution."""
    video_manager = get_video_stream_manager()
    logger = get_web_logger()

    caps = []
    for path in VIDEO_FILE_PATHS:
        if not os.path.exists(path):
            logger.warning(f"Benchmark video not found: {path}")
            continue
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            caps.append(cap)
            logger.info(f"Opened benchmark video: {os.path.basename(path)}")
        else:
            logger.warning(f"Cannot open benchmark video: {path}")

    if not caps:
        logger.error("No benchmark video files could be opened. Check PATHS in config.ini")
        return

    logger.info(f"Benchmark streams loaded: {len(caps)}")
    video_manager.set_camera_count(len(caps))

    yolo_engines = []
    try:
        if INFERENCE_DEVICE == "NPU" and len(caps) > 1:
            for idx in range(len(caps)):
                core_id = idx if NPU_CORE_ASSIGNMENT == "distributed" else None
                engine = create_yolo11_engine(INFERENCE_DEVICE, npu_core_id=core_id)
                yolo_engines.append(engine)
                logger.info(f"YOLO11 engine {idx} initialized for stream {idx} (NPU Core {core_id if core_id is not None else 0})")
                if DEBUG_MODE:
                    logging.debug(f"[DEBUG] Stream {idx} engine platform: {engine.platform}")
        else:
            engine = create_yolo11_engine(INFERENCE_DEVICE)
            yolo_engines = [engine]
            logger.info(f"YOLO11 engine initialized for {INFERENCE_DEVICE} inference")

        if web_server and yolo_engines:
            web_server.active_model_name = os.path.basename(yolo_engines[0].model_path)
            web_server.rknn_instance = yolo_engines[0].model if hasattr(yolo_engines[0].model, 'rknn') else None

    except Exception as e:
        logger.error(f"Failed to initialize YOLO11 engines: {e}")
        for cap in caps:
            cap.release()
        return

    video_manager.start()

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger.info("Live benchmark threat detection started")

    stop_event = threading.Event()
    processing_active = lambda: (not stop_event.is_set()) if web_server is None else web_server.processing_active

    threads = []
    for idx, cap in enumerate(caps):
        engine = yolo_engines[idx if len(yolo_engines) > 1 else 0]
        t = threading.Thread(
            target=_stream_worker,
            args=(idx, cap, engine, video_manager, processing_active, OUTPUT_DIR, logger),
            daemon=True,
            name=f"benchmark-stream-{idx}",
        )
        threads.append(t)

    for t in threads:
        t.start()

    if web_server is None:
        try:
            while any(t.is_alive() for t in threads):
                for idx in range(len(caps)):
                    frame = video_manager.get_latest_frame(camera_id=idx)
                    if frame is not None:
                        cv2.imshow(f"Stream {idx}", frame)
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

    for cap in caps:
        cap.release()

    logger.info("Benchmark Analysis Complete")
