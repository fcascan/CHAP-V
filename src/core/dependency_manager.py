# -*- coding: utf-8 -*-
"""dependency_manager.py
Automatic dependency management for YOLO RKNN/NPU project
by fcascan 2025
"""
import os
import sys
import subprocess
import importlib.util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


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
        import rknnlite
        print("[INFO] RKNN toolkit is available.")
        return True
    except ImportError:
        print("[WARNING] RKNN toolkit not available, attempting installation...")
        try:
            installer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "install_dependencies.py")
            # Run only the RKNN installation part
            result = subprocess.run([sys.executable, "-c", 
                                   "import sys; sys.path.append('.'); from install_dependencies import install_rknn_wheel; install_rknn_wheel()"], 
                                  capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
            
            # Try importing again
            import rknnlite
            print("[INFO] RKNN toolkit installed successfully!")
            return True
        except (subprocess.CalledProcessError, ImportError):
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