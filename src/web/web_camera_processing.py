# -*- coding: utf-8 -*-
"""web_camera_processing.py
Camera processing with web interface integration
by fcascan 2025
"""
import cv2
import time
import logging
import threading
import os
import sys
import numpy as np
import pyudev
from ..core.config import *
from .video_integration import get_video_stream_manager
from .console_integration import get_web_logger
from ..processing.yolo11_inference import create_yolo11_engine

def process_cameras_web(yolo_postprocess_func=None, web_server=None):
    """Process cameras with web interface integration."""
    video_manager = get_video_stream_manager()
    logger = get_web_logger()
    
    # Find cameras
    context = pyudev.Context()
    video_devices = []
    for device in context.list_devices(subsystem='video4linux'):
        devnode = device.device_node
        if devnode and devnode.startswith('/dev/video'):
            try:
                idx = int(devnode.replace('/dev/video', ''))
                video_devices.append(idx)
            except ValueError:
                continue
                
    video_devices = sorted(set(video_devices))[:MAX_CAMERAS_TO_SCAN]
    cameras = []
    
    for i in video_devices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(cap)
            # Camera initialized
        else:
            cap.release()
            
    if len(cameras) == 0:
        logger.error("No cameras detected, at least one camera is required.")
        return
        
    logger.info(f"Cameras detected: {len(cameras)}")
    
    # Configure video manager for multiple cameras
    video_manager.set_camera_count(len(cameras))
    
    # Create output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Initialize YOLO11 inference engines (one per camera for NPU, shared for GPU/CPU)
    yolo_engines = []
    try:
        if INFERENCE_DEVICE == "NPU" and len(cameras) > 1:
            # Create separate engines for each camera on NPU
            for idx in range(len(cameras)):
                engine = create_yolo11_engine(INFERENCE_DEVICE)
                yolo_engines.append(engine)
                logger.info(f"YOLO11 engine {idx} initialized for camera {idx}")
                if DEBUG_MODE:
                    logging.debug(f"[DEBUG] Camera {idx} engine platform: {engine.platform}")
        else:
            # Single shared engine for GPU/CPU or single camera
            engine = create_yolo11_engine(INFERENCE_DEVICE)
            yolo_engines = [engine]  # Use same engine for all cameras
            logger.info(f"YOLO11 engine initialized for {INFERENCE_DEVICE} inference")
            if DEBUG_MODE:
                logging.debug(f"[DEBUG] Shared engine platform: {engine.platform}")
            
        # Update web server with active model info (using first engine)
        if web_server and yolo_engines:
            web_server.active_model_name = os.path.basename(yolo_engines[0].model_path)
            web_server.rknn_instance = yolo_engines[0].model if hasattr(yolo_engines[0].model, 'rknn') else None
            
    except Exception as e:
        logger.error(f"Failed to initialize YOLO11 engines: {e}")
        return
    
    # Start video stream manager
    video_manager.start()
    
    display_timestamps = [[] for _ in range(len(cameras))]
    inftime_per_camera = [[] for _ in range(len(cameras))]
    failure_counters = [0] * len(cameras)
    imgs_to_draw = [None] * len(cameras)
    
    # Statistics
    camera_total_frames = [0] * len(cameras)
    camera_inference_times = [[] for _ in range(len(cameras))]
    camera_processing_times = [[] for _ in range(len(cameras))]
    start_global = time.time()
    
    logger.info("Live camera threat detection started")
    
    # Check if we have a web server for processing control
    processing_active = True
    if web_server:
        processing_active = lambda: web_server.processing_active
    else:
        processing_active = lambda: True
    
    while processing_active() if callable(processing_active) else processing_active:
        # Process cameras (focus on first camera for web display)
        for idx, cap in enumerate(cameras):
            ret, frame = cap.read()
            if not ret:
                failure_counters[idx] += 1
                logger.error(f"Failed to read the frame from camera {idx}.")
                if failure_counters[idx] >= 10:
                    logger.error(f"Critical: Camera {idx} failed too many times. Stopping the program.")
                    return
                continue
                
            failure_counters[idx] = 0
            start_time = time.time()
            
            start_inference = time.time()
            
            # YOLO11 inference with camera-specific debug logging
            if DEBUG_MODE:
                logging.debug(f"[DEBUG] Camera {idx} processing frame {camera_total_frames[idx] + 1}")
                logging.debug(f"[DEBUG] Camera {idx} frame shape: {frame.shape}")
            
            # Use appropriate engine (separate for NPU multi-camera, shared for others)
            engine_idx = idx if len(yolo_engines) > 1 else 0
            current_engine = yolo_engines[engine_idx]
            
            boxes, classes, scores, processed_frame = current_engine.detect_objects(frame)
            
            end_inference = time.time()
            
            # Debug logging for camera inference results
            if DEBUG_MODE:
                if boxes is not None:
                    logging.debug(f"[DEBUG] Camera {idx} inference: {len(boxes)} detections found")
                    detection_summary = current_engine.get_detection_summary(boxes, classes, scores)
                    logging.debug(f"[DEBUG] Camera {idx} detection classes: {detection_summary['class_counts']}")
                else:
                    logging.debug(f"[DEBUG] Camera {idx} inference: No detections found")
            inf_time = end_inference - start_inference
            camera_inference_times[idx].append(inf_time)
            
            end_time = time.time()
            total_frame_time = end_time - start_time
            camera_processing_times[idx].append(total_frame_time)
            
            inftime_per_camera[idx].append(inf_time)
            if len(inftime_per_camera[idx]) > 30:
                inftime_per_camera[idx].pop(0)
                
            avg_inf_time_ms = 1000 * sum(inftime_per_camera[idx]) / len(inftime_per_camera[idx])
            
            now = time.time()
            display_timestamps[idx].append(now)
            if len(display_timestamps[idx]) > 30:
                display_timestamps[idx].pop(0)
                
            display_fps = 0.0
            if len(display_timestamps[idx]) > 1:
                elapsed = display_timestamps[idx][-1] - display_timestamps[idx][0]
                if elapsed > 0:
                    display_fps = (len(display_timestamps[idx]) - 1) / elapsed
            
            # Create display frame
            imgs_to_draw[idx] = frame.copy()
            
            # Add overlay information
            cv2.putText(imgs_to_draw[idx], f"Camera {idx} - Frame: {camera_total_frames[idx]+1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 0), 2)
            cv2.putText(imgs_to_draw[idx], f"Inf time: {avg_inf_time_ms:.1f} ms", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 255), 2)
            cv2.putText(imgs_to_draw[idx], f"FPS: {display_fps:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (255, 255, 0), 2)
            
            # Draw detections using YOLO11
            if boxes is not None and classes is not None and scores is not None:
                # Use YOLO11's draw function for consistent rendering
                current_engine.draw_detections(imgs_to_draw[idx], boxes, classes, scores)
                
                # Debug logging for camera-specific detections
                if DEBUG_MODE:
                    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                        class_name = current_engine.get_class_name(cls)
                        logging.debug(f"[DEBUG] Camera {idx} detection {i+1}: {class_name} @ {box} (score: {score:.3f})")
            
            # Save image output
            if imgs_to_draw[idx] is not None:
                output_path = os.path.join(OUTPUT_DIR, f"inference_output_cam{idx}.jpg")
                cv2.imwrite(output_path, imgs_to_draw[idx])
                
                # Update web video stream for each camera
                video_manager.update_frame(imgs_to_draw[idx], camera_id=idx)
            
            camera_total_frames[idx] += 1
            
        # Small delay to prevent overwhelming the system
        time.sleep(0.01)
        
    end_global = time.time()
    
    # Stop video stream manager
    video_manager.stop()
    
    # Cleanup YOLO11 engines
    for engine in yolo_engines:
        try:
            engine.release()
            logger.info("YOLO11 engine resources released")
        except Exception as e:
            logger.warning(f"Error releasing YOLO11 engine: {e}")
    
    # Cleanup cameras
    for cap in cameras:
        cap.release()
    
    # Print concise camera statistics
    logger.info("Camera Analysis Complete")
    for idx in range(len(cameras)):
        if display_timestamps[idx] and len(display_timestamps[idx]) > 1:
            elapsed = display_timestamps[idx][-1] - display_timestamps[idx][0]
            avg_fps = (len(display_timestamps[idx]) - 1) / elapsed if elapsed > 0 else 0.0
            logger.info(f"Camera {idx}: {camera_total_frames[idx]} frames @ {avg_fps:.1f} FPS")

    # Processor usage statistics
    logger.info("PROCESSOR USAGE STATISTICS")
    logger.info("-" * 30)
    from ..utils.my_htop import get_processor_usage_stats
    proc_stats = get_processor_usage_stats(INFERENCE_DEVICE)
    if proc_stats['cpu']:
        logger.info(f"CPU Usage - Avg: {proc_stats['cpu']['avg']:.1f}%")
    else:
        logger.info("CPU Usage - N/A")
    if proc_stats['npu']:
        logger.info(f"NPU Usage - Avg: {proc_stats['npu']['avg']:.1f}% (per core: {proc_stats['npu']['per_core']})")
    else:
        logger.info("NPU Usage - N/A")
    if proc_stats['gpu']:
        samples_info = f" ({proc_stats['gpu']['samples']} samples)" if 'samples' in proc_stats['gpu'] else ""
        logger.info(f"GPU Usage - Avg: {proc_stats['gpu']['avg']:.1f}%{samples_info}")
    else:
        logger.info("GPU Usage - N/A")
    logger.info("="*50)