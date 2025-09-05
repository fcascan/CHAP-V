# -*- coding: utf-8 -*-
"""main.py
Main Entry Point
by fcascan 2025
"""
import sys
import os

# Add src to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.system_setup import setup_system, setup_inference_device, disable_unnecessary_logging
from src.core.config import *


def main():
    """Main application entry point."""
    # Setup system dependencies and permissions
    setup_system()
    
    # Import required modules after dependencies are verified
    import cv2
    import time
    import numpy as np
    import threading
    import psutil
    
    # Setup and configure inference device
    actual_device, rknn_available, rknn_modules = setup_inference_device(INFERENCE_DEVICE)
    
    # Update the global INFERENCE_DEVICE if it changed
    if actual_device != INFERENCE_DEVICE:
        globals()['INFERENCE_DEVICE'] = actual_device
    
    # Disable unnecessary logging
    disable_unnecessary_logging()
    
    # Import processing modules
    from src.processing.video_processing import process_video
    from src.processing.camera_processing import process_cameras
    from src.processing.yolo_post import yolo_onnx_postprocess
    
    # Run the appropriate processing mode
    print(f"[INFO] Starting in {'BENCHMARK' if BENCHMARK_MODE else 'CAMERA'} mode using {actual_device}")
    
    if BENCHMARK_MODE:
        process_video(yolo_onnx_postprocess)
    else:
        process_cameras(yolo_onnx_postprocess)


if __name__ == "__main__":
    main()