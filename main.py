# -*- coding: utf-8 -*-
"""Application entry point."""

from src.core.cli import build_parser, print_console_banner
from src.core.app_launcher import run_console_mode, run_web_mode


def main():
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("[WARNING] No X11/Wayland display detected. GUI windows (cv2.imshow) will be disabled.")
        os.environ["CHAPV_HEADLESS"] = "1"
    args = build_parser().parse_args()

    if args.web:
        print("[MAIN] Launching web mode")
        run_web_mode(args)
        return

    print_console_banner(args)
    run_console_mode(args)


if __name__ == "__main__":
    main()
