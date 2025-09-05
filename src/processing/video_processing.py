# -*- coding: utf-8 -*-
"""video_processing.py
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

if INFERENCE_DEVICE == "NPU":
    from rknnlite.api import RKNNLite
    from ..utils.rknn_post_processing import post_process

def process_video(yolo_postprocess_func):
    """Process video file and return statistics."""
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_FILE_PATH}")
        print("Please update the 'benchmark_video' path in config.ini")
        sys.exit(1)
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {VIDEO_FILE_PATH}")
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded: {VIDEO_FILE_PATH}")
    print(f"Total frames: {total_frames}, FPS: {video_fps:.2f}")
    if INFERENCE_DEVICE == "NPU":
        rknn = RKNNLite()
        rknn.load_rknn(MODEL_PATH)
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print(f"Model loaded for NPU inference.")
    else:
        net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
        print(f"ONNX model loaded for CPU inference: {ONNX_MODEL_PATH}")
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
    print("Starting video processing...")
    print("Press 'q' or ESC to stop early")
    frame_times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing video. Total frames processed: {processed_frames}")
            break
        start_frame = time.time()
        img = cv2.resize(frame, IMG_SIZE)
        start_inference = time.time()
        if INFERENCE_DEVICE == "NPU":
            img_input = np.expand_dims(img, 0)
            outputs = rknn.inference(inputs=[img_input])
            boxes, classes, scores = post_process(outputs)
        else:
            blob = cv2.dnn.blobFromImage(img, 1/255.0, IMG_SIZE, swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()
            boxes, classes, scores = yolo_postprocess_func(outputs, frame.shape)
        end_inference = time.time()
        inf_time = end_inference - start_inference
        inference_times.append(inf_time)
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
        cv2.putText(frame_display, f"Frame: {processed_frames + 1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Inf time: {inf_time*1000:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 255), 2)
        cv2.putText(frame_display, f"FPS: {fps_actual:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (255, 255, 0), 2)
        cv2.imshow("Video Processing", frame_display)
        processed_frames += 1
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            print(f"Processing stopped by user. Frames processed: {processed_frames}")
            break
        if processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({processed_frames}/{total_frames})")
    end_total = time.time()
    total_time = end_total - start_total
    monitoring_active = False
    cap.release()
    cv2.destroyAllWindows()
    if inference_times:
        avg_inference_time = np.mean(inference_times) * 1000
        avg_processing_time = np.mean(processing_times) * 1000
        processing_fps = processed_frames / total_time
        inference_fps = 1.0 / np.mean(inference_times)
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
        print("\nPROCESSOR USAGE STATISTICS")
        print("-" * 30)
        from ..utils.my_htop import get_processor_usage_stats
        proc_stats = get_processor_usage_stats(INFERENCE_DEVICE)
        if proc_stats['cpu']:
            print(f"CPU Usage - Avg: {proc_stats['cpu']['avg']:.1f}%")
        else:
            print("CPU Usage - N/A")
        if proc_stats['npu']:
            print(f"NPU Usage - Avg: {proc_stats['npu']['avg']:.1f}% (per core: {proc_stats['npu']['per_core']})")
        else:
            print("NPU Usage - N/A")
        if proc_stats['gpu']:
            print(f"GPU Usage - Last sample: {proc_stats['gpu']['avg']:.1f}%")
        else:
            print("GPU Usage - N/A")
        print("="*50)
    else:
        print("[ERROR] No frames were processed successfully.")
