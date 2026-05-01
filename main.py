# -*- coding: utf-8 -*-
"""Application entry point."""

from src.core.cli import build_parser, print_console_banner
from src.core.app_launcher import run_console_mode, run_web_mode


def main():
    args = build_parser().parse_args()

    if args.web:
        print("[MAIN] Launching web mode")
        run_web_mode(args)
        return

    print_console_banner(args)
    run_console_mode(args)


if __name__ == "__main__":
    main()
