# -*- coding: utf-8 -*-
"""Thin application entry point with CLI parsing."""

import argparse

from src.core.config import (
    BENCHMARK_MODE,
    IMG_SIZE,
    INFERENCE_DEVICE,
    MODEL_PATH,
    ONNX_MODEL_PATH,
    NMS_THRESHOLD,
    OBJ_THRESHOLD,
    ROCKCHIP_TARGET,
    VIDEO_FILE_PATH,
)
from src.core.app_launcher import (
    resolve_default_classes_file,
    resolve_default_video_source,
    run_console_mode,
    run_web_mode,
)


def build_parser():
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(description="YOLO RKNN Object Detection")
    parser.add_argument("--web", action="store_true", help="Start web interface server")
    parser.add_argument("--web-port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--http-logging", action="store_true", help="Enable HTTP request logging")

    default_video_source = VIDEO_FILE_PATH if BENCHMARK_MODE else resolve_default_video_source()
    default_model_path = ONNX_MODEL_PATH if INFERENCE_DEVICE in {"CPU", "GPU"} else MODEL_PATH
    if not default_model_path or not default_model_path.endswith((".onnx", ".rknn", ".pt", ".torchscript")):
        default_model_path = MODEL_PATH
    parser.add_argument("--model_path", default=default_model_path, help="Path to the model file")
    parser.add_argument("--video_source", default=default_video_source, help="Video file path or camera index")
    parser.add_argument(
        "--classes_file",
        default=resolve_default_classes_file(),
        help="Path to classes file; if missing uses config model_labels/yolo11n/default_labels fallback",
    )
    parser.add_argument("--target", default=ROCKCHIP_TARGET, help="Rockchip target, for example rk3588")
    parser.add_argument("--obj_thresh", type=float, default=OBJ_THRESHOLD, help="Object confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=NMS_THRESHOLD, help="NMS IoU threshold")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE[0], help="Square inference size")
    parser.add_argument("--img_show", action="store_true", help="Show the result window")
    parser.add_argument("--img_save", action="store_true", help="Save the output video or images")
    parser.add_argument("--debug_detections", action="store_true", help="Print raw class ids and score summaries")
    return parser


def main():
    """Parse CLI arguments and dispatch to web or console mode."""
    parser = build_parser()
    args = parser.parse_args()

    if args.web:
        print("[MAIN] Launching web mode")
        run_web_mode(args)
        return

    print("[MAIN] Launching console mode")
    print(f"[MAIN] Inference device: {INFERENCE_DEVICE}")
    print(f"[MAIN] Model: {args.model_path}")
    print(f"[MAIN] Video source: {args.video_source}")
    if args.classes_file:
        print(f"[MAIN] Classes file: {args.classes_file}")
    else:
        print("[MAIN] Classes source: config.ini default_labels fallback")
    run_console_mode(args)


if __name__ == "__main__":
    main()