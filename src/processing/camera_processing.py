# -*- coding: utf-8 -*-
"""camera_processing.py
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
from ..core import config

def process_cameras(yolo_postprocess_func):
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
    current_img_size = config.IMG_SIZE
    current_classes = config.CLASSES
    current_max_cameras = config.MAX_CAMERAS_TO_SCAN
    current_label_text_size = config.LABEL_TEXT_SIZE
    current_fps_text_size = config.FPS_TEXT_SIZE
    
    print(f"Starting camera processing with device: {current_device}")
    
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
    video_devices = sorted(set(video_devices))[:current_max_cameras]
    cameras = []
    for i in video_devices:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(cap)
            print(f"Camera {i} initialized.")
        else:
            cap.release()
    if len(cameras) == 0:
        print(f"[ERROR] No cameras detected, at least one camera is required.")
        exit()
    print(f"Cameras detected = {len(cameras)}.")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "images")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Initialize inference engine based on current device configuration
    rknn_instances = []
    net = None
    gpu_backend_enabled = False
    
    if current_device == "NPU":
        try:
            from rknnlite.api import RKNNLite
            from ..utils.my_htop import log_npu_usage
            
            npu_cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
            for idx, core in enumerate(npu_cores[:len(cameras)]):
                rknn = RKNNLite()
                rknn.load_rknn(current_model_path)
                rknn.init_runtime(core_mask=core)
                rknn_instances.append(rknn)
                print(f"[Camera {idx}] Model loaded on NPU Core {core}.")
        except ImportError as e:
            print(f"[ERROR] Failed to import NPU modules: {e}")
            return
    else:
        net = cv2.dnn.readNetFromONNX(current_onnx_path)
        
        # Configure GPU backend if available (OpenCL for Mali G610)
        if current_device == "GPU":
            try:
                from ..utils.my_htop import start_gpu_monitoring, stop_gpu_monitoring
            except ImportError as e:
                print(f"[WARNING] Failed to import GPU monitoring modules: {e}")
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
                test_blob = cv2.dnn.blobFromImage(np.zeros((*current_img_size, 3), dtype=np.uint8), 1/255.0, current_img_size, swapRB=True, crop=False)
                net.setInput(test_blob)
                net.forward()  # This will fail if OpenCL is not properly set up
                
                gpu_backend_enabled = True
                print(f"ONNX model loaded for GPU inference (OpenCL): {ONNX_MODEL_PATH}")
            except Exception as e:
                print(f"[WARNING] GPU initialization failed, falling back to CPU: {e}")
                gpu_backend_enabled = False
        
        if not gpu_backend_enabled:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"ONNX model loaded for CPU inference: {current_onnx_path}")
    
    # Start GPU monitoring if using GPU inference
    gpu_monitor_thread = None
    if current_device == "GPU":
        try:
            gpu_monitor_thread = start_gpu_monitoring()
        except NameError:
            gpu_monitor_thread = None
            print("[WARNING] GPU monitoring not available")
        print("[INFO] GPU monitoring started")
    
    display_timestamps = [[] for _ in range(len(cameras))]
    inftime_per_camera = [[] for _ in range(len(cameras))]
    failure_counters = [0] * len(cameras)
    imgs_to_draw = [None] * len(cameras)
    # if INFERENCE_DEVICE == "NPU":
        # npu_thread = threading.Thread(target=log_npu_usage, daemon=True)
        # npu_thread.start()
    print("Starting camera processing...")
    # Estadísticas globales
    camera_total_frames = [0] * len(cameras)
    camera_inference_times = [[] for _ in range(len(cameras))]
    camera_processing_times = [[] for _ in range(len(cameras))]
    start_global = time.time()
    frame_counter = 0
    while True:
        for idx, cap in enumerate(cameras):
            ret, frame = cap.read()
            if not ret:
                failure_counters[idx] += 1
                print(f"[ERROR] Failed to read the frame from camera {idx}.")
                if failure_counters[idx] >= 10:
                    print(f"[ERROR] Critical: Camera {idx} failed too many times. Stopping the program.")
                    sys.exit(1)
                continue
            failure_counters[idx] = 0
            
            # Reload display configuration every 50 frames to catch web updates
            if frame_counter % 50 == 0:
                display_config = reload_display_config()
                current_classes = display_config['classes']
                current_label_text_size = display_config['label_text_size']
                current_fps_text_size = display_config['fps_text_size']
            
            start_time = time.time()
            img = cv2.resize(frame, current_img_size)
            start_inference = time.time()
            if current_device == "NPU" and idx < len(rknn_instances):
                img_input = np.expand_dims(img, 0)
                outputs = rknn_instances[idx].inference(inputs=[img_input])
                boxes, classes, scores = yolo_postprocess_func(outputs, frame.shape)
            else:  # GPU or CPU
                blob = cv2.dnn.blobFromImage(img, 1/255.0, current_img_size, swapRB=True, crop=False)
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
            imgs_to_draw[idx] = frame.copy()
            # Add overlay information
            cv2.putText(imgs_to_draw[idx], f"Camera {idx} - Frame: {camera_total_frames[idx]+1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, current_fps_text_size, (0, 255, 0), 2)
            cv2.putText(imgs_to_draw[idx], f"Inf time: {avg_inf_time_ms:.1f} ms", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, current_fps_text_size, (0, 255, 255), 2)
            cv2.putText(imgs_to_draw[idx], f"FPS: {display_fps:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, current_fps_text_size, (255, 255, 0), 2)
            if boxes is not None and classes is not None and scores is not None:
                for box, cls, score in zip(boxes, classes, scores):
                    if cls < len(current_classes):
                        x1, y1, x2, y2 = map(int, box)
                        label = current_classes[cls]
                        red = int(255 * score)
                        green = int(255 * (1 - score))
                        score_color = (0, green, red)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), score_color, 2)
                        cv2.putText(frame, f"{label}: {score:.2f}", (x1 + 5, y1 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, current_label_text_size, score_color, 2)
            if imgs_to_draw[idx] is not None:
                output_path = os.path.join(OUTPUT_DIR, f"inference_output_cam{idx}.jpg")
                cv2.imwrite(output_path, imgs_to_draw[idx])
                cv2.imshow(f"Detections Camera {idx}", imgs_to_draw[idx])
            camera_total_frames[idx] += 1
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break
    end_global = time.time()
    
    # Stop GPU monitoring if it was started
    if current_device == "GPU" and gpu_monitor_thread:
        try:
            stop_gpu_monitoring()
        except NameError:
            print("[WARNING] GPU monitoring stop function not available")
        print("[INFO] GPU monitoring stopped")
    
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    # Per-camera statistics
    print("\n" + "="*50)
    print("CAMERA PROCESSING STATISTICS")
    print("="*50)
    for idx in range(len(cameras)):
        print(f"Camera {idx}:")
        print(f"  Total frames processed: {camera_total_frames[idx]}")
        print(f"  Total processing time: {end_global - start_global:.2f} seconds")
        if camera_inference_times[idx]:
            avg_inf = np.mean(camera_inference_times[idx]) * 1000
            min_inf = np.min(camera_inference_times[idx]) * 1000
            max_inf = np.max(camera_inference_times[idx]) * 1000
            print(f"  Average inference time: {avg_inf:.2f} ms")
            print(f"  Min inference time: {min_inf:.2f} ms")
            print(f"  Max inference time: {max_inf:.2f} ms")
        if camera_processing_times[idx]:
            avg_proc = np.mean(camera_processing_times[idx]) * 1000
            print(f"  Average total frame processing time: {avg_proc:.2f} ms")
        if display_timestamps[idx] and len(display_timestamps[idx]) > 1:
            elapsed = display_timestamps[idx][-1] - display_timestamps[idx][0]
            avg_fps = (len(display_timestamps[idx]) - 1) / elapsed if elapsed > 0 else 0.0
            print(f"  Display FPS: {avg_fps:.2f}")
        else:
            print(f"  [ERROR] No frames were processed for camera {idx}.")
    print("="*50)

    # Processor usage statistics (CPU/NPU/GPU)
    print("\nPROCESSOR USAGE STATISTICS")
    print("-" * 30)
    from ..utils.my_htop import get_processor_usage_stats
    proc_stats = get_processor_usage_stats(current_device)
    if proc_stats['cpu']:
        print(f"CPU Usage - Avg: {proc_stats['cpu']['avg']:.1f}%")
    else:
        print("CPU Usage - N/A")
    if proc_stats['npu']:
        print(f"NPU Usage - Avg: {proc_stats['npu']['avg']:.1f}% (per core: {proc_stats['npu']['per_core']})")
    else:
        print("NPU Usage - N/A")
    if proc_stats['gpu']:
        samples_info = f" ({proc_stats['gpu']['samples']} samples)" if 'samples' in proc_stats['gpu'] else ""
        print(f"GPU Usage - Avg: {proc_stats['gpu']['avg']:.1f}%{samples_info}")
    else:
        print("GPU Usage - N/A")
    print("="*50)
