# -*- coding: utf-8 -*-
"""main_yolo11_test.py
Root launcher for the Rockchip YOLO11 test script.

This wrapper reuses the system setup from the main application and then
executes src/rockchip/yolo11_infer.py with project defaults.
"""

import argparse
import os
import runpy
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
ROCKCHIP_DIR = os.path.join(SRC_DIR, "rockchip")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROCKCHIP_DIR)

from src.core.config import MODEL_PATH, VIDEO_FILE_PATH, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD, IMG_SIZE
from src.core.system_setup import setup_system


def resolve_default_classes_file():
    """Return the preferred classes file for the test launcher."""
    root_classes = os.path.join(PROJECT_ROOT, "yolo11n.txt")
    if os.path.isfile(root_classes):
        return root_classes

    config_classes = os.path.join(PROJECT_ROOT, "assets", "models", "crime2.txt")
    if os.path.isfile(config_classes):
        return config_classes

    return root_classes


def build_infer_argv(args):
    """Build the argv list used to execute yolo11_infer.py."""
    infer_script = os.path.join(PROJECT_ROOT, "src", "rockchip", "yolo11_infer.py")

    argv = [
        infer_script,
        "--model_path", args.model_path,
        "--target", args.target,
        "--video_source", args.video_source,
        "--classes_file", args.classes_file,
        "--obj_thresh", str(args.obj_thresh),
        "--nms_thresh", str(args.nms_thresh),
        "--img_size", str(args.img_size),
    ]

    if args.img_show:
        argv.append("--img_show")
    if args.img_save:
        argv.append("--img_save")
    if args.debug_detections:
        argv.append("--debug_detections")

    return infer_script, argv


def main():
    """Launch the Rockchip YOLO11 inference script with project defaults."""
    parser = argparse.ArgumentParser(description="Root launcher for Rockchip YOLO11 test inference")
    parser.add_argument("--model_path", default=MODEL_PATH, help="Path to the RKNN model file")
    parser.add_argument("--video_source", default=VIDEO_FILE_PATH, help="Video file path or camera index")
    parser.add_argument("--classes_file", default=resolve_default_classes_file(), help="Path to the classes text file")
    parser.add_argument("--target", default=ROCKCHIP_TARGET, help="Rockchip target, for example rk3588")
    parser.add_argument("--obj_thresh", type=float, default=OBJ_THRESHOLD, help="Object confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=NMS_THRESHOLD, help="NMS IoU threshold")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE[0], help="Square inference size")
    parser.add_argument("--img_show", action="store_true", help="Show the result window")
    parser.add_argument("--img_save", action="store_true", help="Save the output video or images")
    parser.add_argument("--debug_detections", action="store_true", help="Print raw class ids and score summaries")

    args = parser.parse_args()

    print("[TEST] Preparing Rockchip YOLO11 test launch")
    print(f"[TEST] Model: {args.model_path}")
    print(f"[TEST] Video source: {args.video_source}")
    print(f"[TEST] Classes file: {args.classes_file}")
    print(f"[TEST] Target: {args.target}")

    if not setup_system():
        raise SystemExit(1)

    infer_script, infer_argv = build_infer_argv(args)

    original_argv = sys.argv[:]
    try:
        sys.argv = infer_argv
        runpy.run_path(infer_script, run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()