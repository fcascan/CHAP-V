# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""Application launcher helpers for console and web entry points.

This module keeps the execution paths explicit while leaving CLI parsing to
the top-level entry point.
"""

import os
import runpy
import sys

from .config import (
    BENCHMARK_MODE,
    MAX_INFERENCE_INSTANCES,
    MODEL_LABELS_FILE_PATH,
    VIDEO_FILE_PATH,
)
from .system_setup import setup_system, setup_web_system


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
ROCKCHIP_DIR = os.path.join(SRC_DIR, "rockchip")

for path in (PROJECT_ROOT, SRC_DIR, ROCKCHIP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def resolve_default_classes_file():
    """Return the preferred classes file or None to use config default labels."""
    if MODEL_LABELS_FILE_PATH and os.path.isfile(MODEL_LABELS_FILE_PATH):
        return MODEL_LABELS_FILE_PATH

    yolo11n_fallback = os.path.join(PROJECT_ROOT, "yolo11n.txt")
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
    for device in context.list_devices(subsystem="video4linux"):
        devnode = device.device_node
        if devnode and devnode.startswith("/dev/video"):
            try:
                camera_indices.append(int(devnode.replace("/dev/video", "")))
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
    infer_script = os.path.join(PROJECT_ROOT, "src", "rockchip", "yolo11_infer.py")

    argv = [
        infer_script,
        "--model_path", args.model_path,
        "--target", args.target,
        "--video_source", args.video_source,
        "--obj_thresh", str(args.obj_thresh),
        "--nms_thresh", str(args.nms_thresh),
        "--img_size", str(args.img_size),
    ]

    if args.classes_file:
        argv.extend(["--classes_file", args.classes_file])
    if args.img_show:
        argv.append("--img_show")
    if args.img_save:
        argv.append("--img_save")
    if args.debug_detections:
        argv.append("--debug_detections")

    return infer_script, argv


def run_console_mode(args):
    """Run console inference.

    Interactive single instance (MAX_INFERENCE_INSTANCES == 1 AND one of
    --img_show / --img_save / --debug_detections given): delegates to the
    Rockchip yolo11_infer.py script, a manual visual test harness.

    Everything else — every benchmark/camera run without interactive flags, and
    all multi-instance runs — goes through the threaded web processing functions
    (without a web server) so that per-frame CSV metrics, the inference-timeout
    watchdog, benchmark_loop and NPU_CORE_ASSIGNMENT are all honoured, and so it
    is headless-safe. This is what run_all_benchmarks.py drives, so a
    single-stream (--instances 1) sweep records results exactly like a 3-stream
    one instead of falling into the metrics-less runpy path.
    """
    if not setup_system():
        raise SystemExit(1)

    interactive = args.img_show or args.img_save or args.debug_detections
    if MAX_INFERENCE_INSTANCES > 1 or not interactive:
        _run_console_threaded()
        return

    infer_script, infer_argv = build_yolo11_argv(args)

    original_argv = sys.argv[:]
    try:
        sys.argv = infer_argv
        runpy.run_path(infer_script, run_name="__main__")
    finally:
        sys.argv = original_argv


def _run_console_threaded():
    """Threaded console flow (1..N streams) — reuses web processing without a web server."""
    if BENCHMARK_MODE:
        from ..web.web_video_processing import process_video_web
        process_video_web(web_server=None)
    else:
        from ..web.web_camera_processing import process_cameras_web
        process_cameras_web(web_server=None)


def run_web_mode(args):
    """Run the web server flow with its dedicated system setup."""
    if not setup_web_system():
        raise SystemExit(1)

    from src.web.web_server import create_web_server

    web_server = create_web_server(host=args.web_host, port=args.web_port, http_logging=args.http_logging)
    web_server.run(debug=False)