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
    
if __name__ == "__main__":
    """main.py - Entry point"""
    import os
    import sys
    import subprocess

    # Ensure running as root
    if os.geteuid() != 0:
        try:
            subprocess.run(['sudo', sys.executable] + sys.argv, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] This script needs to run as root.")
            print(f"Please run: sudo python {sys.argv[0]}")
        sys.exit(1)
    print(f"Running with superuser permissions.")

    from config import BENCHMARK_MODE
    from video_processing import process_video
    from camera_processing import process_cameras
    from yolo_post import yolo_onnx_postprocess

    if __name__ == "__main__":
        if BENCHMARK_MODE:
            process_video(yolo_onnx_postprocess)
        else:
            process_cameras(yolo_onnx_postprocess)