# -*- coding: utf-8 -*-
"""CLI argument parsing and console startup banner."""

import argparse
import textwrap

from .config import (
    BENCHMARK_MODE,
    IMG_SIZE,
    INFERENCE_DEVICE,
    MAX_INFERENCE_INSTANCES,
    MODEL_PATH,
    NPU_CORE_ASSIGNMENT,
    NMS_THRESHOLD,
    OBJ_THRESHOLD,
    ONNX_MODEL_PATH,
    ROCKCHIP_TARGET,
    VIDEO_FILE_PATH,
)
from .app_launcher import (
    resolve_default_classes_file,
    resolve_default_video_source,
)


def build_parser():
    """Build the top-level CLI parser."""

    description = textwrap.dedent("""\
        YOLO RKNN Object Detection — Rockchip NPU/GPU/CPU inference engine.

        Primary configuration is read from config.ini (project root).
        The settings below are controlled ONLY via config.ini and are not
        configurable as CLI arguments:

          [MODE]
            benchmark_mode         true = video file source | false = live cameras

          [INFERENCE]
            inference_device       NPU | GPU | CPU
            rockchip_target        rk3588 | rk3566 | rk3562 | rk3576
            max_inference_instances  number of parallel streams (1..3)
            npu_core_assignment    auto (all on Core 0) | distributed (stream N -> Core N)
            obj_threshold          detection confidence threshold
            nms_threshold          NMS IoU threshold

          [PATHS]
            model_rknn / model_onnx      model files used by device
            benchmark_video_0..N         video sources for benchmark mode

          [CLASSES]
            default_labels         fallback class list when no labels file is found

        ─────────────────────────────────────────────────────────────────────
        SINGLE-INSTANCE mode  (max_inference_instances = 1)
          Runs via the Rockchip yolo11_infer script.
          CLI flags --model_path, --video_source, --target, --obj_thresh,
          --nms_thresh, --img_size, --classes_file, --img_show, --img_save,
          and --debug_detections are all active.

        MULTI-INSTANCE mode  (max_inference_instances > 1)
          Runs the threaded engine; all sources and parameters come from
          config.ini.  --img_show / --img_save / --debug_detections are
          ignored in this mode.
        ─────────────────────────────────────────────────────────────────────
    """)

    epilog = textwrap.dedent("""\
        Examples:
          python main.py                         Console, reads config.ini
          python main.py --img_show              Single-instance with live window
          python main.py --img_save              Single-instance, saves output frames
          python main.py --debug_detections      Print raw detection ids and scores
          python main.py --web                   Start web interface server
          python main.py --web --web-port 9090   Web server on custom port
          python main.py --web --http-logging    Web server with HTTP request logging
    """)

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Web mode ──────────────────────────────────────────────────────────
    web_group = parser.add_argument_group("web mode")
    web_group.add_argument("--web", action="store_true",
                           help="Start the web interface server instead of console mode")
    web_group.add_argument("--web-port", type=int, default=8080,
                           help="Listening port for the web server (default: 8080)")
    web_group.add_argument("--web-host", type=str, default="0.0.0.0",
                           help="Listening host for the web server (default: 0.0.0.0)")
    web_group.add_argument("--http-logging", action="store_true",
                           help="Enable per-request HTTP logging in the web server")

    # ── Single-instance console overrides ─────────────────────────────────
    console_group = parser.add_argument_group(
        "single-instance console overrides (max_inference_instances = 1 only)"
    )
    default_video_source = VIDEO_FILE_PATH if BENCHMARK_MODE else resolve_default_video_source()
    default_model_path = ONNX_MODEL_PATH if INFERENCE_DEVICE in {"CPU", "GPU"} else MODEL_PATH
    if not default_model_path or not default_model_path.endswith((".onnx", ".rknn", ".pt", ".torchscript")):
        default_model_path = MODEL_PATH

    console_group.add_argument("--model_path", default=default_model_path,
                               help="Path to the model file (.rknn / .onnx / .pt)")
    console_group.add_argument("--video_source", default=default_video_source,
                               help="Video file path or camera index (e.g. 0)")
    console_group.add_argument("--classes_file", default=resolve_default_classes_file(),
                               help="Path to a plain-text class labels file "
                                    "(one label per line); falls back to config.ini default_labels")
    console_group.add_argument("--target", default=ROCKCHIP_TARGET,
                               help="Rockchip SoC target — rk3588 | rk3566 | rk3562 | rk3576 "
                                    f"(config.ini: {ROCKCHIP_TARGET})")
    console_group.add_argument("--obj_thresh", type=float, default=OBJ_THRESHOLD,
                               help=f"Detection confidence threshold (config.ini: {OBJ_THRESHOLD})")
    console_group.add_argument("--nms_thresh", type=float, default=NMS_THRESHOLD,
                               help=f"NMS IoU threshold (config.ini: {NMS_THRESHOLD})")
    console_group.add_argument("--img_size", type=int, default=IMG_SIZE[0],
                               help=f"Square inference resolution (config.ini: {IMG_SIZE[0]})")
    console_group.add_argument("--img_show", action="store_true",
                               help="Display inference output in a live window")
    console_group.add_argument("--img_save", action="store_true",
                               help="Save annotated output frames to disk")
    console_group.add_argument("--debug_detections", action="store_true",
                               help="Print raw class IDs and score summaries per frame")

    return parser


def print_console_banner(args):
    """Print a concise startup summary for console mode."""
    mode_label = "BENCHMARK (video)" if BENCHMARK_MODE else "CAMERA (live)"
    print("=" * 56)
    print("  YOLO RKNN — Console Mode")
    print("=" * 56)
    print(f"  Mode              : {mode_label}")
    print(f"  Device            : {INFERENCE_DEVICE}")
    print(f"  Rockchip target   : {ROCKCHIP_TARGET}")
    print(f"  Instances         : {MAX_INFERENCE_INSTANCES}")
    print(f"  NPU core assign   : {NPU_CORE_ASSIGNMENT}")
    if MAX_INFERENCE_INSTANCES == 1:
        print(f"  Model             : {args.model_path}")
        print(f"  Video source      : {args.video_source}")
        if args.classes_file:
            print(f"  Classes file      : {args.classes_file}")
        else:
            print("  Classes source    : config.ini default_labels")
    else:
        print("  (sources and parameters read from config.ini)")
    print("=" * 56)
