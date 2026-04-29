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
import logging
from datetime import datetime
from ..core import config
from ..utils.frame_overlay import calculate_recent_average_ms, calculate_recent_fps, draw_processing_overlay
from .video_integration import get_video_stream_manager
from .console_integration import get_web_logger
from ..processing.yolo11_inference import get_global_yolo11_engine, release_global_engine

def process_video_web(yolo_postprocess_func=None, web_server=None):
    """Process video file with web interface integration."""
    video_manager = get_video_stream_manager()
    logger = get_web_logger()
    
    def reload_display_config():
        """Reload display-related configuration that can change during processing."""
        return {
            'fps_text_size': config.FPS_TEXT_SIZE,
            'overlay_enabled': config.OVERLAY_ENABLED
        }
    
    # Get current configuration dynamically
    current_device = config.INFERENCE_DEVICE
    current_model_path = config.MODEL_PATH
    current_onnx_path = config.ONNX_MODEL_PATH
    current_video_path = config.VIDEO_FILE_PATH
    current_img_size = config.IMG_SIZE
    current_fps_text_size = config.FPS_TEXT_SIZE
    current_overlay_enabled = config.OVERLAY_ENABLED
    
    logger.info(f"Starting YOLO11 video processing with device: {current_device}")
    
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
    
    # Initialize YOLO11 inference engine
    try:
        yolo_engine = get_global_yolo11_engine(current_device)
        logger.info(f"YOLO11 engine initialized for {current_device} inference")
        if config.DEBUG_MODE:
            logging.debug(f"[DEBUG] YOLO11 engine platform: {yolo_engine.platform}")
            logging.debug(f"[DEBUG] Model path: {yolo_engine.model_path}")
        
        # Update web server with active model info
        if web_server:
            web_server.active_model_name = os.path.basename(yolo_engine.model_path)
            web_server.rknn_instance = yolo_engine.model if hasattr(yolo_engine.model, 'rknn') else None
            
    except Exception as e:
        logger.error(f"Failed to initialize YOLO11 engine: {e}")
        return
    
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
    # Write metadata as comments
    csv_writer.writerow([f"#model_path={yolo_engine.model_path}"])
    csv_writer.writerow([f"#benchmark_video={current_video_path}"])
    csv_writer.writerow([f"#inference_device={current_device}"])
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
        
        start_inference = time.time()
        
        # YOLO11 inference with debug logging
        if config.DEBUG_MODE:
            logging.debug(f"[DEBUG] Processing frame {processed_frames + 1}/{total_frames}")
            logging.debug(f"[DEBUG] Frame shape: {frame.shape}")
        
        boxes, classes, scores, processed_frame = yolo_engine.detect_objects(frame)
        
        end_inference = time.time()
        
        # Debug logging for inference results
        if config.DEBUG_MODE:
            if boxes is not None:
                logging.debug(f"[DEBUG] Inference result: {len(boxes)} detections found")
                detection_summary = yolo_engine.get_detection_summary(boxes, classes, scores)
                logging.debug(f"[DEBUG] Detection classes: {detection_summary['class_counts']}")
            else:
                logging.debug(f"[DEBUG] Inference result: No detections found")
        inf_time = end_inference - start_inference
        inference_times.append(inf_time)
        
        # Reload display configuration every 50 frames to catch web updates
        if processed_frames % 50 == 0:
            display_config = reload_display_config()
            current_fps_text_size = display_config['fps_text_size']
            current_overlay_enabled = display_config['overlay_enabled']
        
        # Create display frame and count detections using YOLO11
        detections_count = 0
        frame_display = frame.copy()
        if boxes is not None and classes is not None and scores is not None:
            detections_count = len(boxes)
            # Use YOLO11's draw function for consistent rendering
            yolo_engine.draw_detections(frame_display, boxes, classes, scores)
            
            # Debug logging for specific detections
            if config.DEBUG_MODE:
                for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                    class_name = yolo_engine.get_class_name(cls)
                    logging.debug(f"[DEBUG] Detection {i+1}: {class_name} @ {box} (score: {score:.3f})")
        
        end_frame = time.time()
        total_frame_time = end_frame - start_frame
        processing_times.append(total_frame_time)
        frame_times.append(end_frame)
        
        if len(frame_times) > 30:
            frame_times.pop(0)
            
        fps_actual = calculate_recent_fps(frame_times[-30:])
        avg_inf_time_ms = calculate_recent_average_ms(inference_times[-30:])
        
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
            
        draw_processing_overlay(
            frame_display,
            current_overlay_enabled,
            f"Frame: {processed_frames + 1}/{total_frames}",
            inference_time_ms=avg_inf_time_ms,
            fps_value=fps_actual,
            text_size=current_fps_text_size,
        )
        
        # Update web video stream
        video_manager.update_frame(frame_display)
        
        processed_frames += 1
        
        if processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100
            # Progress update (silent for cleaner logs)
            
    end_total = time.time()
    total_time = end_total - start_total
    monitoring_active = False
    
    # Stop video stream manager
    video_manager.stop()
    
    # Release YOLO11 engine resources
    try:
        release_global_engine()
        logger.info("YOLO11 engine resources released")
    except Exception as e:
        logger.warning(f"Error releasing YOLO11 engine: {e}")
    
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
        from ..utils.csv_analysis import auto_analyze_latest_csv
        auto_analyze_latest_csv(current_device, logger, csv_filepath)
    else:
        logger.error("No frames were processed successfully.")