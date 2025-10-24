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

if INFERENCE_DEVICE == "NPU":
    from rknnlite.api import RKNNLite
    from ..utils.rknn_post_processing import post_process
    from ..utils.my_htop import log_npu_usage
elif INFERENCE_DEVICE == "GPU":
    from ..utils.my_htop import start_gpu_monitoring, stop_gpu_monitoring

def process_cameras_web(yolo_postprocess_func, web_server=None):
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
    
    # Create output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Initialize inference engine
    if INFERENCE_DEVICE == "NPU":
        npu_cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
        rknn_instances = []
        for idx, core in enumerate(npu_cores[:len(cameras)]):
            rknn = RKNNLite()
            rknn.load_rknn(MODEL_PATH)
            rknn.init_runtime(core_mask=core)
            rknn_instances.append(rknn)
            # Model loaded on NPU core
    else:
        net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
        gpu_backend_enabled = False
        
        # Configure GPU backend if available (OpenCL for Mali G610)
        if INFERENCE_DEVICE == "GPU":
            try:
                # Check if OpenCL is available
                if not cv2.ocl.haveOpenCL():
                    raise Exception("OpenCL not available")
                
                if not hasattr(cv2.dnn, 'DNN_TARGET_OPENCL'):
                    raise Exception("DNN_TARGET_OPENCL not available")
                
                # Enable OpenCL and test GPU backend
                cv2.ocl.setUseOpenCL(True)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                
                # Create a small test to verify OpenCL works
                test_blob = cv2.dnn.blobFromImage(np.zeros((640, 640, 3), dtype=np.uint8), 1/255.0, (640, 640), swapRB=True, crop=False)
                net.setInput(test_blob)
                net.forward()  # This will fail if OpenCL is not properly set up
                
                gpu_backend_enabled = True
                logger.info("Threat detection model loaded (GPU)")
            except Exception as e:
                logger.warning(f"GPU initialization failed, falling back to CPU: {e}")
                gpu_backend_enabled = False
        
        if not gpu_backend_enabled:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Threat detection model loaded (CPU)")
    
    # Start GPU monitoring if using GPU inference
    gpu_monitor_thread = None
    if INFERENCE_DEVICE == "GPU":
        gpu_monitor_thread = start_gpu_monitoring()
        # GPU monitoring started
    
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
            img = cv2.resize(frame, IMG_SIZE)
            
            start_inference = time.time()
            if INFERENCE_DEVICE == "NPU":
                img_input = np.expand_dims(img, 0)
                outputs = rknn_instances[idx].inference(inputs=[img_input])
                boxes, classes, scores = post_process(outputs)
            else:  # GPU or CPU
                blob = cv2.dnn.blobFromImage(img, 1/255.0, IMG_SIZE, swapRB=True, crop=False)
                net.setInput(blob)
                outputs = net.forward()
                boxes, classes, scores = yolo_postprocess_func(outputs, frame.shape)
                
            end_inference = time.time()
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
            
            # Draw detections
            if boxes is not None and classes is not None and scores is not None:
                for b, label, s in [(box, CLASSES[c], score) for box, c, score in zip(boxes, classes, scores) if c < len(CLASSES)]:
                    x1, y1, x2, y2 = map(int, b)
                    red = int(255 * s)
                    green = int(255 * (1 - s))
                    score_color = (0, green, red)
                    cv2.rectangle(imgs_to_draw[idx], (x1, y1), (x2, y2), score_color, 2)
                    cv2.putText(imgs_to_draw[idx], f"{label}: {s:.2f}", (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_TEXT_SIZE, score_color, 2)
            
            # Save image output
            if imgs_to_draw[idx] is not None:
                output_path = os.path.join(OUTPUT_DIR, f"inference_output_cam{idx}.jpg")
                cv2.imwrite(output_path, imgs_to_draw[idx])
                
                # Update web video stream (only first camera for now)
                if idx == 0:
                    video_manager.update_frame(imgs_to_draw[idx])
            
            camera_total_frames[idx] += 1
            
        # Small delay to prevent overwhelming the system
        time.sleep(0.01)
        
    end_global = time.time()
    
    # Stop GPU monitoring if it was started
    if INFERENCE_DEVICE == "GPU" and gpu_monitor_thread:
        stop_gpu_monitoring()
        # GPU monitoring stopped
    
    # Stop video stream manager
    video_manager.stop()
    
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