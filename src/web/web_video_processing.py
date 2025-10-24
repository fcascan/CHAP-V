# -*- coding: utf-8 -*-
"""web_video_processing.py
Video processing with web interface integration
by fcascan 2025
"""
import os
import sys
import cv2
import time
import numpy as np
import threading
import psutil
from ..core.config import *
from .video_integration import get_video_stream_manager
from .console_integration import get_web_logger

if INFERENCE_DEVICE == "NPU":
    from rknnlite.api import RKNNLite
    from ..utils.rknn_post_processing import post_process
    from ..utils.my_htop import log_npu_usage
elif INFERENCE_DEVICE == "GPU":
    from ..utils.my_htop import start_gpu_monitoring, stop_gpu_monitoring

def process_video_web(yolo_postprocess_func, web_server=None):
    """Process video file with web interface integration."""
    video_manager = get_video_stream_manager()
    logger = get_web_logger()
    
    if not os.path.exists(VIDEO_FILE_PATH):
        logger.error(f"Video file not found: {VIDEO_FILE_PATH}")
        logger.info("Please update the 'benchmark_video' path in config.ini")
        return
        
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {VIDEO_FILE_PATH}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video analysis started: {os.path.basename(VIDEO_FILE_PATH)} ({total_frames} frames)")
    
    # Initialize inference engine based on device
    if INFERENCE_DEVICE == "NPU":
        rknn = RKNNLite()
        rknn.load_rknn(MODEL_PATH)
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        logger.info("Threat detection model loaded (NPU)")
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
    
    inference_times = []
    processing_times = []
    processed_frames = 0
    cpu_usage_samples = []
    npu_usage_samples = []
    monitoring_active = True
    
    def monitor_usage():
        while monitoring_active:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_usage_samples.append(cpu_percent)
            if INFERENCE_DEVICE == "NPU":
                try:
                    npu_usage = 0
                    try:
                        with open('/sys/kernel/debug/rknpu/load', 'r') as f:
                            content = f.read().strip()
                            for line in content.split('\n'):
                                if 'NPU load:' in line:
                                    npu_usage = float(line.split(':')[1].strip().rstrip('%'))
                                    break
                    except (FileNotFoundError, PermissionError, ValueError):
                        npu_usage = min(100.0, len(inference_times) * 10 if inference_times else 0)
                    npu_usage_samples.append(npu_usage)
                except Exception:
                    npu_usage_samples.append(0)
            time.sleep(0.1)
            
    monitor_thread = threading.Thread(target=monitor_usage, daemon=True)
    monitor_thread.start()
    
    start_total = time.time()
    logger.info("Video threat analysis in progress...")
    
    frame_times = []
    
    # Check if we have a web server for processing control
    processing_active = True
    if web_server:
        processing_active = lambda: web_server.processing_active
    else:
        processing_active = lambda: True
    
    while processing_active() if callable(processing_active) else processing_active:
        ret, frame = cap.read()
        if not ret:
            logger.info(f"Video analysis completed: {processed_frames} frames processed")
            break
            
        start_frame = time.time()
        img = cv2.resize(frame, IMG_SIZE)
        
        start_inference = time.time()
        if INFERENCE_DEVICE == "NPU":
            img_input = np.expand_dims(img, 0)
            outputs = rknn.inference(inputs=[img_input])
            boxes, classes, scores = post_process(outputs)
        else:  # GPU or CPU
            blob = cv2.dnn.blobFromImage(img, 1/255.0, IMG_SIZE, swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()
            boxes, classes, scores = yolo_postprocess_func(outputs, frame.shape)
            
        end_inference = time.time()
        inf_time = end_inference - start_inference
        inference_times.append(inf_time)
        
        # Create display frame
        frame_display = frame.copy()
        if boxes is not None and classes is not None and scores is not None:
            for b, label, s in [(box, CLASSES[c], score) for box, c, score in zip(boxes, classes, scores) if c < len(CLASSES)]:
                x1, y1, x2, y2 = map(int, b)
                red = int(255 * s)
                green = int(255 * (1 - s))
                score_color = (0, green, red)
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), score_color, 2)
                cv2.putText(frame_display, f"{label}: {s:.2f}", (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_TEXT_SIZE, score_color, 2)
        
        end_frame = time.time()
        total_frame_time = end_frame - start_frame
        processing_times.append(total_frame_time)
        frame_times.append(end_frame)
        
        if len(frame_times) > 30:
            frame_times.pop(0)
            
        if len(frame_times) > 1:
            fps_actual = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps_actual = 0.0
            
        # Add overlay information
        cv2.putText(frame_display, f"Frame: {processed_frames + 1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Inf time: {inf_time*1000:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 255), 2)
        cv2.putText(frame_display, f"FPS: {fps_actual:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (255, 255, 0), 2)
        
        # Update web video stream
        video_manager.update_frame(frame_display)
        
        processed_frames += 1
        
        if processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100
            # Progress update (silent for cleaner logs)
            
    end_total = time.time()
    total_time = end_total - start_total
    monitoring_active = False
    
    # Stop GPU monitoring if it was started
    if INFERENCE_DEVICE == "GPU" and gpu_monitor_thread:
        stop_gpu_monitoring()
        # GPU monitoring stopped
    
    # Stop video stream manager
    video_manager.stop()
    
    cap.release()
    
    # Print statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times) * 1000
        avg_processing_time = np.mean(processing_times) * 1000
        processing_fps = processed_frames / total_time
        inference_fps = 1.0 / np.mean(inference_times)
        
        logger.info("Analysis Complete")
        logger.info(f"Analysis stats: {processed_frames} frames in {total_time:.1f}s using {INFERENCE_DEVICE}")
        logger.info(f"Average inference time: {avg_inference_time:.2f} ms")
        logger.info(f"Average total frame processing time: {avg_processing_time:.2f} ms")
        logger.info(f"Processing FPS: {processing_fps:.2f}")
        logger.info(f"Inference FPS: {inference_fps:.2f}")
        logger.info(f"Min inference time: {min(inference_times)*1000:.2f} ms")
        logger.info(f"Max inference time: {max(inference_times)*1000:.2f} ms")
        
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
    else:
        logger.error("No frames were processed successfully.")