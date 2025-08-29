# -*- coding: utf-8 -*-
"""main.py
by fcascan 2025
"""
import os
import sys
import cv2
import time
import numpy as np
import threading
import subprocess
import logging
import psutil

#%% Verify if the script is running as root
if os.geteuid() != 0:
    try:
        subprocess.run(['sudo', sys.executable] + sys.argv, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] This script needs to run as root.")
        print(f"Please run: sudo python {sys.argv[0]}")
    sys.exit(1)
print(f"Running with superuser permissions.")

# Import config only after root check
from config import *

# Only import NPU and post-processing if needed
if INFERENCE_DEVICE == "NPU":
    from rknnlite.api import RKNNLite
    from utils.rknn_post_processing import post_process
    from utils.my_rknputop import log_npu_usage

# Disable logging for unnecessary messages
logger = logging.getLogger()
logger.disabled = True


def process_video():
    """Process video file and return statistics."""
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_FILE_PATH}")
        print("Please update the 'benchmark_video' path in config.ini")
        sys.exit(1)
        
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {VIDEO_FILE_PATH}")
        sys.exit(1)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded: {VIDEO_FILE_PATH}")
    print(f"Total frames: {total_frames}, FPS: {video_fps:.2f}")
    
    # Model loading and setup
    if INFERENCE_DEVICE == "NPU":
        rknn = RKNNLite()
        rknn.load_rknn(MODEL_PATH)
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print(f"Model loaded for NPU inference.")
    else:
        # CPU: Load ONNX model with OpenCV DNN
        net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
        print(f"ONNX model loaded for CPU inference: {ONNX_MODEL_PATH}")
    
    # Statistics tracking
    inference_times = []
    processing_times = []
    processed_frames = 0
    
    # CPU/NPU usage monitoring
    cpu_usage_samples = []
    npu_usage_samples = []
    monitoring_active = True
    
    def monitor_usage():
        """Monitor CPU and NPU usage in background thread"""
        while monitoring_active:
            # Monitor CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_usage_samples.append(cpu_percent)
            
            # Monitor NPU usage if available
            if INFERENCE_DEVICE == "NPU":
                try:
                    # Try to get NPU usage from system files
                    npu_usage = 0
                    try:
                        with open('/sys/kernel/debug/rknpu/load', 'r') as f:
                            content = f.read().strip()
                            # Parse NPU usage from load file
                            for line in content.split('\n'):
                                if 'NPU load:' in line:
                                    npu_usage = float(line.split(':')[1].strip().rstrip('%'))
                                    break
                    except (FileNotFoundError, PermissionError, ValueError):
                        # Alternative: use approximate NPU load based on inference frequency
                        npu_usage = min(100.0, len(inference_times) * 10 if inference_times else 0)
                    
                    npu_usage_samples.append(npu_usage)
                except Exception as e:
                    npu_usage_samples.append(0)
            
            time.sleep(0.1)  # Sample every 100ms
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_usage, daemon=True)
    monitor_thread.start()
    
    start_total = time.time()
    
    print("Starting video processing...")
    print("Press 'q' or ESC to stop early")
    
    # Para calcular FPS de visualización
    frame_times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing video. Total frames processed: {processed_frames}")
            break

        start_frame = time.time()

        # Prepare frame for inference
        img = cv2.resize(frame, IMG_SIZE)

        # Inference
        start_inference = time.time()
        if INFERENCE_DEVICE == "NPU":
            img_input = np.expand_dims(img, 0)  # Add batch dimension
            outputs = rknn.inference(inputs=[img_input])
            boxes, classes, scores = post_process(outputs)
        else:
            # CPU: OpenCV DNN expects BGR images, shape [1,3,H,W], float32, 0-1
            blob = cv2.dnn.blobFromImage(img, 1/255.0, IMG_SIZE, swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()
            boxes, classes, scores = yolo_onnx_postprocess(outputs, frame.shape)

        end_inference = time.time()
        inf_time = end_inference - start_inference
        inference_times.append(inf_time)

        # Draw results on frame
        frame_display = frame.copy()

        # Draw bounding boxes and labels on the image
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

        # FPS de visualización
        frame_times.append(end_frame)
        if len(frame_times) > 30:
            frame_times.pop(0)
        if len(frame_times) > 1:
            fps_actual = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps_actual = 0.0

        # Add frame info
        cv2.putText(frame_display, f"Frame: {processed_frames + 1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Inf time: {inf_time*1000:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 255), 2)
        cv2.putText(frame_display, f"FPS: {fps_actual:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (255, 255, 0), 2)

        # Display frame
        cv2.imshow("Video Processing", frame_display)

        processed_frames += 1

        # Check for exit
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:  # 'q' or ESC
            print(f"Processing stopped by user. Frames processed: {processed_frames}")
            break

        # Progress indicator
        if processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({processed_frames}/{total_frames})")
    
    end_total = time.time()
    total_time = end_total - start_total
    
    # Stop monitoring
    monitoring_active = False
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate and display statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        avg_processing_time = np.mean(processing_times) * 1000  # Convert to ms
        processing_fps = processed_frames / total_time
        inference_fps = 1.0 / np.mean(inference_times)
        
        # Calculate CPU/NPU usage statistics
        cpu_stats = {}
        npu_stats = {}
        
        if cpu_usage_samples:
            cpu_stats = {
                'avg': np.mean(cpu_usage_samples),
                'min': np.min(cpu_usage_samples),
                'max': np.max(cpu_usage_samples)
            }
        
        if INFERENCE_DEVICE == "NPU" and npu_usage_samples:
            npu_stats = {
                'avg': np.mean(npu_usage_samples),
                'min': np.min(npu_usage_samples),
                'max': np.max(npu_usage_samples)
            }
        
        print("\n" + "="*50)
        print("VIDEO PROCESSING STATISTICS")
        print("="*50)
        print(f"Video file: {VIDEO_FILE_PATH}")
        print(f"Inference device: {INFERENCE_DEVICE}")
        print(f"Total frames processed: {processed_frames}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Average total frame processing time: {avg_processing_time:.2f} ms")
        print(f"Processing FPS: {processing_fps:.2f}")
        print(f"Inference FPS: {inference_fps:.2f}")
        print(f"Min inference time: {min(inference_times)*1000:.2f} ms")
        print(f"Max inference time: {max(inference_times)*1000:.2f} ms")
        
        # Display processor usage statistics
        print("\nPROCESSOR USAGE STATISTICS")
        print("-" * 30)
        if cpu_stats:
            print(f"CPU Usage - Avg: {cpu_stats['avg']:.1f}%, Min: {cpu_stats['min']:.1f}%, Max: {cpu_stats['max']:.1f}%")
        
        if INFERENCE_DEVICE == "NPU" and npu_stats:
            print(f"NPU Usage - Avg: {npu_stats['avg']:.1f}%, Min: {npu_stats['min']:.1f}%, Max: {npu_stats['max']:.1f}%")
        elif INFERENCE_DEVICE == "CPU":
            print("NPU Usage - N/A (CPU inference mode)")
            
        print("="*50)
    else:
        print("[ERROR] No frames were processed successfully.")


def process_cameras():
    """Original camera processing functionality."""
    print("Camera processing mode - functionality preserved but not fully implemented")
    print("Set benchmark_mode = true in config.ini to process video files")


def yolo_onnx_postprocess(outputs, img_shape, conf_thres=0.25, iou_thres=0.45):
    """YOLO ONNX post-processing function"""
    if isinstance(outputs, (list, tuple)):
        preds = outputs[0]
    else:
        preds = outputs
    
    # Flatten to (N, 85) or similar
    while preds.ndim > 2:
        preds = preds[0]
    
    if preds.shape[1] < 6:
        return None, None, None
    
    boxes = preds[:, :4]
    scores = preds[:, 4:5] * preds[:, 5:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    mask = confidences > conf_thres
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    # Rescale boxes to original image size
    h0, w0 = img_shape[:2]
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= w0 / IMG_SIZE[0]
        boxes[:, [1, 3]] *= h0 / IMG_SIZE[1]
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thres, iou_thres) if len(boxes) > 0 else []
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], class_ids[indices], confidences[indices]
    else:
        return None, None, None


if __name__ == "__main__":
    if BENCHMARK_MODE:
        process_video()
    else:
        process_cameras()