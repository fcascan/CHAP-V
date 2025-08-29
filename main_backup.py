# -*- coding: utf-8 -*-
"""main.py
by fcascan 2025
"""

import cv2 
import time
import logging
import threading
import os
import sys
import subprocess
import numpy as np

#%% Verify if the script is running as root
if os.geteuid() != 0:
    try:
        subprocess.check_call(["sudo", "python3"] + sys.argv)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to obtain superuser permissions.")
        print(f"The program needs superuser permissions.")
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


#%%

def process_video():
    """Process video file and return statistics."""
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"[ERROR] Video file not found: {VIDEO_FILE_PATH}")
        print("Please update the 'video_file' path in config.ini")
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
    
    start_total = time.time()
    
    print("Starting video processing...")
    print("Press 'q' or ESC to stop early")
    
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
        
        # Add frame info
        cv2.putText(frame_display, f"Frame: {processed_frames + 1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Inf time: {inf_time*1000:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 255), 2)
        
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
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate and display statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        avg_processing_time = np.mean(processing_times) * 1000  # Convert to ms
        processing_fps = processed_frames / total_time
        inference_fps = 1.0 / np.mean(inference_times)
        
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
        print("="*50)
    else:
        print("[ERROR] No frames were processed successfully.")

def process_cameras():
    """Original camera processing functionality."""
    import pyudev
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
            print(f"Camera {i} initialized.")
        else:
            cap.release()

    # Verify if at least one camera is connected
    if len(cameras) == 0:
        print(f"[ERROR] No cameras detected, at least one camera is required.")
        exit()
    print(f"Cameras detected = {len(cameras)}.")

    # Create output directory
    OUTPUT_DIR = os.path.join(BASE_DIR, "images")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Model loading and setup for cameras
    if INFERENCE_DEVICE == "NPU":
        npu_cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
        rknn_instances = []
        for idx, core in enumerate(npu_cores[:len(cameras)]): 
            rknn = RKNNLite()
            rknn.load_rknn(MODEL_PATH)
            rknn.init_runtime(core_mask=core)
            rknn_instances.append(rknn)
            print(f"[Camera {idx}] Model loaded on NPU Core {core}.")
    else:
        net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
        print(f"ONNX model loaded for CPU inference: {ONNX_MODEL_PATH}")

    # Set up camera processing
    display_timestamps = [[] for _ in range(len(cameras))]
    inftime_per_camera = [[] for _ in range(len(cameras))]
    failure_counters = [0] * len(cameras)
    start_global = time.time()
    imgs_to_draw = [None] * len(cameras)

    # Log thread for NPU usage
    if INFERENCE_DEVICE == "NPU":
        npu_thread = threading.Thread(target=log_npu_usage, daemon=True)
        npu_thread.start()

    # Main camera loop would go here...
    print("Camera processing not fully implemented in this version")
    

if __name__ == "__main__":
    if BENCHMARK_MODE:
        process_video()
    else:
        process_cameras()
        rknn.init_runtime(core_mask=core)
        rknn_instances.append(rknn)
        print(f"Model loaded for camera {idx} on NPU_CORE_{idx}.")
else:
    # CPU: Load ONNX model with OpenCV DNN
    net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
    print(f"ONNX model loaded for CPU inference: {ONNX_MODEL_PATH}")


#%% 
# Set up
display_timestamps = [[] for _ in range(len(cameras))]  # Store timestamps of last 30 displayed frames
inftime_per_camera = [[] for _ in range(len(cameras))]  # Store last 30 inference times (seconds)
failure_counters = [0] * len(cameras)
start_global = time.time()
imgs_to_draw = [None] * len(cameras)


# Log thread for NPU usage
if INFERENCE_DEVICE == "NPU":
    log_thread = threading.Thread(target=log_npu_usage, daemon=True)
    log_thread.start()


# Main loop
def yolo_onnx_postprocess(outputs, img_shape, conf_thres=0.25, iou_thres=0.45):
    # Robust output shape handling for ONNX output
    # Print shapes for debugging
    if hasattr(outputs, 'shape'):
        print("outputs shape:", outputs.shape)
    else:
        print("outputs type:", type(outputs))
    if isinstance(outputs, (list, tuple)):
        print("outputs[0] shape:", outputs[0].shape if hasattr(outputs[0], 'shape') else type(outputs[0]))
        preds = outputs[0]
    else:
        preds = outputs
    # Flatten to (N, 85) or (N, 4) as needed
    while preds.ndim > 2:
        preds = preds[0]
    print("preds shape after squeeze:", preds.shape)
    if preds.shape[1] < 6:
        print("[ERROR] Unexpected output shape for YOLO ONNX model.")
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

        # Reset the failure counter if the camera is working correctly
        failure_counters[idx] = 0

        start_time = time.time()
        img = cv2.resize(frame, IMG_SIZE)

        if INFERENCE_DEVICE == "NPU":
            img_input = np.expand_dims(img, 0)  # Add batch dimension
            outputs = rknn_instances[idx].inference(inputs=[img_input])
            boxes, classes, scores = post_process(outputs)
        else:
            # CPU: OpenCV DNN expects BGR images, shape [1,3,H,W], float32, 0-1
            blob = cv2.dnn.blobFromImage(img, 1/255.0, IMG_SIZE, swapRB=True, crop=False)
            net.setInput(blob)

            outputs = net.forward()
            print("outputs shape:", outputs.shape if hasattr(outputs, 'shape') else type(outputs))
            print("outputs[0] shape:", outputs[0].shape if hasattr(outputs[0], 'shape') else type(outputs[0]))
            boxes, classes, scores = yolo_onnx_postprocess(outputs, frame.shape)

        end_time = time.time()

        # Calculate and draw display FPS and rolling average inference time for this camera
        inf_time = end_time - start_time
        inftime_per_camera[idx].append(inf_time)
        if len(inftime_per_camera[idx]) > 30:
            inftime_per_camera[idx].pop(0)
        avg_inf_time_ms = 1000 * sum(inftime_per_camera[idx]) / len(inftime_per_camera[idx])

        # Display FPS calculation (based on visualization rate)
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
        cv2.putText(imgs_to_draw[idx], f"FPS: {display_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 0), 2)
        cv2.putText(imgs_to_draw[idx], f"Avg inf: {avg_inf_time_ms:.1f} ms (30f)", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 255), 2)

        # Draw bounding boxes and labels on the image
        if boxes is not None and classes is not None and scores is not None:
            for b, label, s in [(box, CLASSES[c], score) for box, c, score in zip(boxes, classes, scores) if c < len(CLASSES)]:
                x1, y1, x2, y2 = map(int, b)
                red = int(255 * s)
                green = int(255 * (1 - s))
                score_color = (0, green, red)
                cv2.rectangle(imgs_to_draw[idx], (x1, y1), (x2, y2), score_color, 2)
                cv2.putText(imgs_to_draw[idx], f"{label}: {s:.2f}", (x1 + 5, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_TEXT_SIZE, score_color, 2)

        # Save and display the processed image in a window
        if imgs_to_draw[idx] is not None:
            output_path = os.path.join(OUTPUT_DIR, f"inference_output_cam{idx}.jpg")
            cv2.imwrite(output_path, imgs_to_draw[idx])
            cv2.imshow(f"Detections Camera {idx}", imgs_to_draw[idx])

    # Exit if 'q' or 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break


#%% 
# Release resources
for cap in cameras:
    cap.release()
cv2.destroyAllWindows()

# Display the Average FPS per camera
for idx, ts_list in enumerate(display_timestamps):
    if len(ts_list) > 1:
        elapsed = ts_list[-1] - ts_list[0]
        avg_fps = (len(ts_list) - 1) / elapsed if elapsed > 0 else 0.0
        print(f"Average display FPS for camera {idx}: {avg_fps:.2f}")
    else:
        print(f"[ERROR] No frames were processed for camera {idx}.")
