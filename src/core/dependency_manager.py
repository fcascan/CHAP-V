# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""dependency_manager.py
Automatic dependency management for YOLO RKNN/NPU project
by fcascan 2026
"""
import os
import sys
import subprocess
import importlib.util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check if GPU acceleration is available via OpenCL (Mali G610)."""
    try:
        import cv2
        
        # Check if OpenCL is available in OpenCV
        if not cv2.ocl.haveOpenCL():
            return False, "OpenCL not available in OpenCV"
        
        # Check if DNN module supports OpenCL target
        if not hasattr(cv2.dnn, 'DNN_TARGET_OPENCL'):
            return False, "DNN_TARGET_OPENCL not available"
        
        # Try to enable OpenCL
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            return False, "Failed to enable OpenCL"
        
        return True, "OpenCL GPU acceleration available (Mali G610)"
        
    except ImportError:
        return False, "OpenCV not available"
    except Exception as e:
        return False, f"GPU check failed: {e}"


def check_and_install_dependencies():
    """Check if dependencies are installed, if not, run the installer."""
    try:
        import cv2
        import numpy as np
        import psutil
        import pyudev
        print("[INFO] All core dependencies are available.")
        return True
    except ImportError as e:
        missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
        print(f"[WARNING] Missing dependency: {missing_module}")
        print("[INFO] Attempting to install dependencies automatically...")
        
        try:
            # Run the dependency installer
            installer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "install_dependencies.py")
            result = subprocess.run([sys.executable, installer_path], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # Try importing again
            import cv2
            import numpy as np
            import psutil
            import pyudev
            print("[INFO] Dependencies installed successfully!")
            return True
            
        except (subprocess.CalledProcessError, ImportError) as e:
            print(f"[ERROR] Failed to install dependencies: {e}")
            print("[ERROR] Please install dependencies manually:")
            print("  Option 1: Run 'python install_dependencies.py'")
            print("  Option 2: Run 'pip install -r requirements.txt'")
            print("  Option 3: Install manually: pip install opencv-python numpy psutil pyudev")
            return False


def check_rknn_availability():
    """Check if RKNN is available and try to install if not."""
    try:
        from rknnlite.api import RKNNLite  # noqa: F401
        print("[INFO] RKNN toolkit is available.")
        return True
    except Exception:
        print("[WARNING] RKNN toolkit not available, attempting installation...")
        try:
            # Run only the RKNN installation part
            result = subprocess.run([sys.executable, "-c", 
                                   "import sys; sys.path.append('.'); from install_dependencies import install_rknn_wheel; install_rknn_wheel()"], 
                                  capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
            
            # Try importing again
            from rknnlite.api import RKNNLite  # noqa: F401
            print("[INFO] RKNN toolkit installed successfully!")
            return True
        except (subprocess.CalledProcessError, Exception):
            print("[WARNING] RKNN toolkit installation failed. Continuing in CPU mode.")
            return False


def ensure_root_permissions():
    """Ensure the script is running with root permissions."""
    if os.geteuid() != 0:
        try:
            subprocess.run(['sudo', sys.executable] + sys.argv, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] This script needs to run as root.")
            print(f"Please run: sudo python {sys.argv[0]}")
        sys.exit(1)
    print(f"Running with superuser permissions.")


def check_root_permissions():
    """Check if the script is running with root permissions without exiting."""
    return os.geteuid() == 0


def require_root_permissions():
    """Require root permissions and provide clear error message if not available."""
    if not check_root_permissions():
        print("=" * 50)
        print("[ERROR] Root permissions required")
        print("=" * 50)
        print("This application requires root access to:")
        print("• Access NPU hardware acceleration")
        print("• Configure system resources")
        print("• Manage camera devices")
        print("• Monitor system performance")
        print()
        print("[INFO] To start the web server, run:")
        print(f"   sudo python3 {' '.join(sys.argv)}")
        print()
        return False
    print("[INFO] Root permissions successfully verified")
    return True