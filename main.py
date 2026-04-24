# -*- coding: utf-8 -*-
"""main.py
Main Entry Point
by fcascan 2025
"""
import sys
import os
import argparse
import runpy

# Add src to the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
ROCKCHIP_DIR = os.path.join(SRC_DIR, 'rockchip')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROCKCHIP_DIR)

from src.core.config import (
    BENCHMARK_MODE,
    MODEL_PATH,
    MODEL_LABELS_FILE_PATH,
    VIDEO_FILE_PATH,
    ROCKCHIP_TARGET,
    OBJ_THRESHOLD,
    NMS_THRESHOLD,
    IMG_SIZE,
)
from src.core.system_setup import setup_system, setup_web_system


def resolve_default_classes_file():
    """Return resolved classes file path or None to use config default_labels fallback."""
    if MODEL_LABELS_FILE_PATH and os.path.isfile(MODEL_LABELS_FILE_PATH):
        return MODEL_LABELS_FILE_PATH

    yolo11n_fallback = os.path.join(PROJECT_ROOT, 'yolo11n.txt')
    if os.path.isfile(yolo11n_fallback):
        return yolo11n_fallback

    return None


def resolve_default_video_source():
    """Return the first available camera index, or the benchmark video as fallback."""
    try:
        import cv2
        import pyudev
    except ImportError:
        return VIDEO_FILE_PATH

    context = pyudev.Context()
    camera_indices = []
    for device in context.list_devices(subsystem='video4linux'):
        devnode = device.device_node
        if devnode and devnode.startswith('/dev/video'):
            try:
                camera_indices.append(int(devnode.replace('/dev/video', '')))
            except ValueError:
                continue

    for camera_index in sorted(set(camera_indices)):
        capture = cv2.VideoCapture(camera_index)
        try:
            if capture.isOpened():
                capture.release()
                return str(camera_index)
        finally:
            if capture is not None:
                capture.release()

    return VIDEO_FILE_PATH


def build_yolo11_argv(args):
    """Build the argv list used to execute the Rockchip YOLO11 script."""
    infer_script = os.path.join(PROJECT_ROOT, 'src', 'rockchip', 'yolo11_infer.py')

    argv = [
        infer_script,
        '--model_path', args.model_path,
        '--target', args.target,
        '--video_source', args.video_source,
        '--obj_thresh', str(args.obj_thresh),
        '--nms_thresh', str(args.nms_thresh),
        '--img_size', str(args.img_size),
    ]

    if args.classes_file:
        argv.extend(['--classes_file', args.classes_file])

    if args.img_show:
        argv.append('--img_show')
    if args.img_save:
        argv.append('--img_save')
    if args.debug_detections:
        argv.append('--debug_detections')

    return infer_script, argv


def run_rockchip_yolo11_launcher(args):
    """Run the Rockchip YOLO11 inference flow from the main entry point."""
    if not setup_system():
        raise SystemExit(1)

    infer_script, infer_argv = build_yolo11_argv(args)

    original_argv = sys.argv[:]
    try:
        sys.argv = infer_argv
        runpy.run_path(infer_script, run_name='__main__')
    finally:
        sys.argv = original_argv


def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO RKNN Object Detection')
    parser.add_argument('--web', action='store_true', help='Start web interface server')
    parser.add_argument('--web-port', type=int, default=8080, help='Web server port (default: 8080)')
    parser.add_argument('--web-host', type=str, default='0.0.0.0', help='Web server host (default: 0.0.0.0)')
    parser.add_argument('--http-logging', action='store_true', help='Enable HTTP request logging')

    # Prefer a detected camera when running live, otherwise fall back to the benchmark video.
    default_video_source = VIDEO_FILE_PATH if BENCHMARK_MODE else resolve_default_video_source()
    parser.add_argument('--model_path', default=MODEL_PATH, help='Path to the RKNN model file')
    parser.add_argument('--video_source', default=default_video_source, help='Video file path or camera index')
    parser.add_argument('--classes_file', default=resolve_default_classes_file(), help='Path to classes file; if missing uses config model_labels/yolo11n/default_labels fallback')
    parser.add_argument('--target', default=ROCKCHIP_TARGET, help='Rockchip target, for example rk3588')
    parser.add_argument('--obj_thresh', type=float, default=OBJ_THRESHOLD, help='Object confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=NMS_THRESHOLD, help='NMS IoU threshold')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE[0], help='Square inference size')
    parser.add_argument('--img_show', action='store_true', help='Show the result window')
    parser.add_argument('--img_save', action='store_true', help='Save the output video or images')
    parser.add_argument('--debug_detections', action='store_true', help='Print raw class ids and score summaries')
    args = parser.parse_args()
    
    if args.web:
        # Start web interface with system setup (includes SUDO check)
        if not setup_web_system():
            sys.exit(1)  # Exit gracefully if setup fails
        from src.web.web_server import create_web_server
        web_server = create_web_server(host=args.web_host, port=args.web_port, http_logging=args.http_logging)
        web_server.run(debug=False)
    else:
        # Run the Rockchip YOLO11 launcher directly from the main entry point.
        print('[MAIN] Launching Rockchip YOLO11 inference flow')
        print(f'[MAIN] Model: {args.model_path}')
        print(f'[MAIN] Video source: {args.video_source}')
        if args.classes_file:
            print(f'[MAIN] Classes file: {args.classes_file}')
        else:
            print('[MAIN] Classes source: config.ini default_labels fallback')
        run_rockchip_yolo11_launcher(args)

if __name__ == "__main__":
    main()