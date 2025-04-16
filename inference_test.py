# -*- coding: utf-8 -*-
"""inference_test.py
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
from rknnlite.api import RKNNLite
from utils.rknn_post_processing import post_process
from utils.my_rknputop import log_npu_usage
from config import *


#%% Verify if the script is running as root
if os.geteuid() != 0:
    print(f"The program needs superuser permissions.")
    try:
        subprocess.check_call(["sudo", "python3"] + sys.argv)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to obtain superuser permissions.")
    sys.exit(1)
print(f"Running with superuser permissions.")

# Disable logging for unnecessary messages
logger = logging.getLogger()
logger.disabled = True


#%%
# Initialize cameras
cameras = []
for i in range(MAX_CAMERAS_TO_SCAN):
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


#%% 
# Load the model and initialize RKNN instances for each camera
npu_cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
rknn_instances = []
for idx, core in enumerate(npu_cores[:len(cameras)]): 
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=core)
    rknn_instances.append(rknn)
    print(f"Model loaded for camera {idx} on NPU_CORE_{idx}.")


#%% 
# Set up
fps_per_camera = [[] for _ in range(len(cameras))]
failure_counters = [0] * len(cameras)
start_global = time.time()
imgs_to_draw = [None] * len(cameras)

# Log thread for NPU usage
log_thread = threading.Thread(target=log_npu_usage, daemon=True)
log_thread.start()

# Main loop
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

        # Resize the frame to the expected input size of the model
        start_time = time.time()
        img = cv2.resize(frame, IMG_SIZE)
        img = np.expand_dims(img, 0)  # Add batch dimension
        outputs = rknn_instances[idx].inference(inputs=[img])
        boxes, classes, scores = post_process(outputs)
        end_time = time.time()

        # Calculate and draw FPS for this camera
        fps = 1 / (end_time - start_time)
        fps_per_camera[idx].append(fps)
        imgs_to_draw[idx] = frame.copy()
        cv2.putText(imgs_to_draw[idx], f"FPS: {fps:.2f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, FPS_TEXT_SIZE, (0, 255, 0), 2)

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
for idx, fps_list in enumerate(fps_per_camera):
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"Average FPS for camera {idx}: {avg_fps:.2f}")
    else:
        print(f"[ERROR] No frames were processed for camera {idx}.")
