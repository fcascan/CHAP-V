# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""install_dependencies.py
Dependency installer for CHAP-V. Installs the Python requirements and the local
RKNN Toolkit Lite2 wheel. Invoked by src/core/dependency_manager.py (which shells
out to this file, and imports install_rknn_wheel from it) and usable directly:

    python src/install_dependencies.py
"""

import glob
import os
import subprocess
import sys

# This file lives in src/, so the project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def install_requirements():
    """pip-install requirements.txt into the current interpreter's environment."""
    req = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.isfile(req):
        print(f"[WARNING] requirements.txt not found at {req}")
        return False
    print(f"[INFO] Installing Python requirements from {req} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req], check=True)
    return True


def install_rknn_wheel():
    """Install the local RKNN Toolkit Lite2 wheel from installation/ (aarch64 / cp312).

    The wheel is git-ignored (redistribution unclear); download it from
    https://github.com/airockchip/rknn-toolkit2 and drop it in installation/.
    Mirrors setup.sh: --force-reinstall --no-deps so it does not disturb the
    numpy<2 / onnxruntime pins.
    """
    inst_dir = os.path.join(PROJECT_ROOT, "installation")
    wheels = sorted(glob.glob(os.path.join(inst_dir, "rknn_toolkit_lite2-*.whl")))
    if not wheels:
        print("[WARNING] No rknn_toolkit_lite2-*.whl found in installation/. "
              "Download it from github.com/airockchip/rknn-toolkit2 to enable RKNPU modes.")
        return False
    print(f"[INFO] Installing RKNN Toolkit Lite2 wheel: {os.path.basename(wheels[-1])}")
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--force-reinstall", "--no-deps", wheels[-1]], check=True)
    return True


def main():
    ok = install_requirements()
    install_rknn_wheel()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
