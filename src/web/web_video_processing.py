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
import csv
from datetime import datetime
from ..core import config
from .video_integration import get_video_stream_manager
from .console_integration import get_web_logger

def process_video_web(yolo_postprocess_func, web_server=None):
    """Process video file with web interface integration."""
    video_manager = get_video_stream_manager()
    logger = get_web_logger()
    
    def reload_display_config():
        """Reload display-related configuration that can change during processing."""
        return {
            'label_text_size': config.LABEL_TEXT_SIZE,
            'fps_text_size': config.FPS_TEXT_SIZE,
            'classes': config.CLASSES
        }
    
    # Get current configuration dynamically
    current_device = config.INFERENCE_DEVICE
    current_model_path = config.MODEL_PATH
    current_onnx_path = config.ONNX_MODEL_PATH
    current_video_path = config.VIDEO_FILE_PATH
    current_img_size = config.IMG_SIZE
    current_classes = config.CLASSES
    current_label_text_size = config.LABEL_TEXT_SIZE
    current_fps_text_size = config.FPS_TEXT_SIZE
    
    logger.info(f"Starting video processing with device: {current_device}")
    
    if not os.path.exists(current_video_path):
        logger.error(f"Video file not found: {current_video_path}")
        logger.info("Please update the 'benchmark_video' path in config.ini")
        return
        
    cap = cv2.VideoCapture(current_video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {current_video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video analysis started: {os.path.basename(current_video_path)} ({total_frames} frames)")
    
    # Initialize inference engine based on current device configuration
    rknn = None
    net = None
    
    if current_device == "NPU":
        try:
            from rknnlite.api import RKNNLite
            from ..utils.rknn_post_processing import post_process
            from ..utils.my_htop import log_npu_usage
        except ImportError as e:
            logger.error(f"Failed to import NPU modules: {e}")
            return
        rknn = RKNNLite()
        rknn.load_rknn(current_model_path)
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        logger.info("Threat detection model loaded (NPU)")
    else:
        net = cv2.dnn.readNetFromONNX(current_onnx_path)
        gpu_backend_enabled = False
        
        # Configure GPU backend if available (OpenCL for Mali G610)
        if current_device == "GPU":
            try:
                from ..utils.my_htop import start_gpu_monitoring, stop_gpu_monitoring
            except ImportError as e:
                logger.warning(f"Failed to import GPU monitoring modules: {e}")
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
    if current_device == "GPU":
        try:
            gpu_monitor_thread = start_gpu_monitoring()
            # GPU monitoring started
        except NameError:
            logger.warning("GPU monitoring not available")
    
    # Start video stream manager
    video_manager.start()
    
    # Create CSV file for performance metrics in src/processing/results directory
    results_dir = os.path.join(os.getcwd(), "src", "processing", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"performance_metrics_{current_device}_{timestamp_str}.csv"
    csv_filepath = os.path.join(results_dir, csv_filename)
    
    # CSV headers
    csv_headers = [
        'timestamp', 'frame_number', 'inference_time_ms', 'total_frame_time_ms',
        'cpu_usage_percent', 'npu_core0_percent', 'npu_core1_percent', 'npu_core2_percent',
        'gpu_usage_percent', 'fps_actual', 'detections_count'
    ]
    
    csv_file = open(csv_filepath, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_headers)
    
    # CSV line limitation
    max_csv_lines = 5000
    csv_lines_written = 0
    csv_data_buffer = []  # Buffer to store recent CSV rows
    
    logger.info(f"Performance metrics will be saved to: src/processing/results/{csv_filename}")
    
    inference_times = []
    processing_times = []
    processed_frames = 0
    cpu_usage_samples = []
    npu_usage_samples = []  # Will store per-core usage samples as tuples (core0, core1, core2)
    monitoring_active = True
    
    # Variables for CSV logging
    current_cpu_usage = 0
    current_npu_usage = [0, 0, 0]
    current_gpu_usage = 0
    
    def monitor_usage():
        nonlocal current_cpu_usage, current_npu_usage, current_gpu_usage
        while monitoring_active:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_usage_samples.append(cpu_percent)
            current_cpu_usage = cpu_percent
            
            if current_device == "NPU":
                try:
                    # Import the function from my_htop to get consistent NPU readings
                    from ..utils.my_htop import get_npu_info
                    npu_load, _ = get_npu_info()
                    if npu_load and len(npu_load) >= 3:
                        # Store per-core usage [core0, core1, core2]
                        npu_usage_samples.append(tuple(npu_load))
                        current_npu_usage = npu_load[:3]
                    else:
                        npu_usage_samples.append((0, 0, 0))
                        current_npu_usage = [0, 0, 0]
                except Exception:
                    npu_usage_samples.append((0, 0, 0))
                    current_npu_usage = [0, 0, 0]
            elif current_device == "GPU":
                try:
                    from ..utils.my_htop import get_gpu_info
                    gpu_load, _ = get_gpu_info()
                    if gpu_load is not None:
                        current_gpu_usage = gpu_load
                    else:
                        current_gpu_usage = 0
                except Exception:
                    current_gpu_usage = 0
            
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
        img = cv2.resize(frame, current_img_size)
        
        start_inference = time.time()
        if current_device == "NPU" and rknn is not None:
            img_input = np.expand_dims(img, 0)
            outputs = rknn.inference(inputs=[img_input])
            boxes, classes, scores = post_process(outputs)
        else:  # GPU or CPU
            blob = cv2.dnn.blobFromImage(img, 1/255.0, current_img_size, swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()
            boxes, classes, scores = yolo_postprocess_func(outputs, frame.shape)
            
        end_inference = time.time()
        inf_time = end_inference - start_inference
        inference_times.append(inf_time)
        
        # Reload display configuration every 50 frames to catch web updates
        if processed_frames % 50 == 0:
            display_config = reload_display_config()
            current_classes = display_config['classes']
            current_label_text_size = display_config['label_text_size']
            current_fps_text_size = display_config['fps_text_size']
        
        # Create display frame and count detections
        detections_count = 0
        frame_display = frame.copy()
        if boxes is not None and classes is not None and scores is not None:
            detections_count = len(boxes)
            for b, label, s in [(box, current_classes[c], score) for box, c, score in zip(boxes, classes, scores) if c < len(current_classes)]:
                x1, y1, x2, y2 = map(int, b)
                red = int(255 * s)
                green = int(255 * (1 - s))
                score_color = (0, green, red)
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), score_color, 2)
                cv2.putText(frame_display, f"{label}: {s:.2f}", (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, current_label_text_size, score_color, 2)
        
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
        
        # Write metrics to CSV
        try:
            csv_row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],  # timestamp with milliseconds
                processed_frames + 1,  # frame_number
                round(inf_time * 1000, 2),  # inference_time_ms
                round(total_frame_time * 1000, 2),  # total_frame_time_ms
                round(current_cpu_usage, 1),  # cpu_usage_percent
                current_npu_usage[0] if len(current_npu_usage) > 0 else 0,  # npu_core0_percent
                current_npu_usage[1] if len(current_npu_usage) > 1 else 0,  # npu_core1_percent
                current_npu_usage[2] if len(current_npu_usage) > 2 else 0,  # npu_core2_percent
                current_gpu_usage,  # gpu_usage_percent
                round(fps_actual, 2),  # fps_actual
                detections_count  # detections_count
            ]
            
            # Add to buffer
            csv_data_buffer.append(csv_row)
            
            # Keep only last max_csv_lines
            if len(csv_data_buffer) > max_csv_lines:
                csv_data_buffer.pop(0)
            
            # Write to CSV (rewrite file if we hit the limit to keep only recent data)
            csv_lines_written += 1
            if csv_lines_written > max_csv_lines:
                # Rewrite the entire file with only recent data
                csv_file.close()
                csv_file = open(csv_filepath, 'w', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(csv_headers)
                
                # Write all buffered data
                for buffered_row in csv_data_buffer:
                    csv_writer.writerow(buffered_row)
                
                csv_lines_written = len(csv_data_buffer)
            else:
                # Normal write
                csv_writer.writerow(csv_row)
            
            # Flush CSV file every 50 frames to ensure data is saved
            if processed_frames % 50 == 0:
                csv_file.flush()
        except Exception as e:
            logger.warning(f"Failed to write CSV row: {e}")
            
        # Add overlay information
        cv2.putText(frame_display, f"Frame: {processed_frames + 1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, current_fps_text_size, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Inf time: {inf_time*1000:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, current_fps_text_size, (0, 255, 255), 2)
        cv2.putText(frame_display, f"FPS: {fps_actual:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, current_fps_text_size, (255, 255, 0), 2)
        
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
    if current_device == "GPU" and gpu_monitor_thread:
        try:
            stop_gpu_monitoring()
            # GPU monitoring stopped
        except NameError:
            logger.warning("GPU monitoring stop function not available")
    
    # Stop video stream manager
    video_manager.stop()
    
    # Close CSV file
    try:
        csv_file.close()
        logger.info(f"Performance metrics saved to: src/processing/results/{csv_filename}")
        logger.info(f"CSV file contains {processed_frames} rows of performance data")
    except Exception as e:
        logger.error(f"Error closing CSV file: {e}")
    
    cap.release()
    
    # Print statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times) * 1000
        avg_processing_time = np.mean(processing_times) * 1000
        processing_fps = processed_frames / total_time
        inference_fps = 1.0 / np.mean(inference_times)
        
        logger.info("Analysis Complete")
        logger.info(f"Basic stats: {processed_frames} frames in {total_time:.1f}s using {current_device}")
        logger.info(f"Processing FPS: {processing_fps:.2f}")
        logger.info(f"Average inference time: {avg_inference_time:.2f} ms")
        
        # Use automatic CSV analysis instead of manual statistics
        logger.info("="*50)
        from ..utils.my_htop import auto_analyze_latest_csv
        auto_analyze_latest_csv(current_device, logger, csv_filepath)
    else:
        logger.error("No frames were processed successfully.")