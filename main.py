# -*- coding: utf-8 -*-
"""main.py
Main Entry Point
by fcascan 2025
"""
import sys
import os

# Add src to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.system_setup import setup_system, setup_web_system, setup_inference_device, disable_unnecessary_logging
from src.core.config import *


def main():
    """Main application entry point."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO RKNN Object Detection')
    parser.add_argument('--web', action='store_true', help='Start web interface server')
    parser.add_argument('--web-port', type=int, default=8080, help='Web server port (default: 8080)')
    parser.add_argument('--web-host', type=str, default='0.0.0.0', help='Web server host (default: 0.0.0.0)')
    parser.add_argument('--http-logging', action='store_true', help='Enable HTTP request logging')
    args = parser.parse_args()
    
    if args.web:
        # Start web interface with system setup (includes SUDO check)
        if not setup_web_system():
            sys.exit(1)  # Exit gracefully if setup fails
        from src.web.web_server import create_web_server
        web_server = create_web_server(host=args.web_host, port=args.web_port, http_logging=args.http_logging)
        web_server.run(debug=False)
    else:
        # Run traditional console mode
        run_console_mode()

def run_console_mode():
    """Run in traditional console mode."""
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
    print(f"🛡️  Threat Detection System - {'Video Analysis' if BENCHMARK_MODE else 'Live Camera'} Mode ({actual_device})")
    
    if BENCHMARK_MODE:
        process_video(yolo_onnx_postprocess)
    else:
        process_cameras(yolo_onnx_postprocess)


if __name__ == "__main__":
    main()