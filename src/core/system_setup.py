# -*- coding: utf-8 -*-
"""system_setup.py
System setup and configuration for YOLO RKNN/NPU project
by fcascan 2026
"""
import sys
import logging
from .dependency_manager import check_and_install_dependencies, check_rknn_availability, check_gpu_availability, ensure_root_permissions, require_root_permissions

def setup_system():
    """Complete system setup including dependencies and permissions."""
    # Check dependencies before proceeding
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Verify root permissions
    ensure_root_permissions()
    
    return True


def setup_web_system():
    """Setup system for web interface with graceful permission handling."""
    # Check root permissions first (non-exiting)
    if not require_root_permissions():
        return False
    
    # Check dependencies after permission verification
    if not check_and_install_dependencies():
        print("[ERROR] Failed to install required dependencies")
        return False
    
    print("[INFO] System setup complete")
    return True


def setup_inference_device(inference_device):
    """Setup and configure the inference device (NPU, GPU, or CPU)."""
    if inference_device == "NPU":
        # NPU setup
        if check_rknn_availability():
            try:
                from importlib import import_module
                RKNNLite = import_module("rknnlite.api.rknn_lite").RKNNLite
                from src.utils.rknn_post_processing import post_process
                from src.utils.my_htop import log_npu_usage
                print("[INFO] RKNN NPU libraries loaded successfully.")
                return "NPU", True, {"RKNNLite": RKNNLite, "post_process": post_process, "log_npu_usage": log_npu_usage}
            except ImportError as e:
                print(f"[WARNING] RKNN NPU libraries could not be imported: {e}")
        
        print("[INFO] NPU not available, switching to CPU inference mode...")
        return "CPU", False, {}
    
    elif inference_device == "GPU-OPENCV-OPENCL":
        # Runs the ONNX model on the Mali-G610 via OpenCV-DNN + OpenCL.
        # check_gpu_availability() returns (bool, str) — unpack explicitly
        gpu_ok, gpu_msg = check_gpu_availability()
        if gpu_ok:
            try:
                import cv2
                if cv2.ocl.haveOpenCL():
                    cv2.ocl.setUseOpenCL(True)
                    try:
                        dev = cv2.ocl.Device_getDefault()
                        print(f"[INFO] GPU-OpenCV-OpenCL available: OpenCL on "
                              f"{dev.name()} ({dev.vendorName()}).")
                    except Exception:
                        print("[INFO] GPU-OpenCV-OpenCL available: OpenCL.")
                    return "GPU-OPENCV-OPENCL", True, {"gpu_available": True}
                print("[WARNING] OpenCV built without OpenCL support — GPU-OpenCV-OpenCL mode needs an "
                      "OpenCL-enabled opencv-python plus the Mali OpenCL ICD "
                      "(/etc/OpenCL/vendors/mali.icd).")
            except ImportError:
                print("[WARNING] opencv-python not installed — GPU-OpenCV-OpenCL requires OpenCV with OpenCL.")
            except Exception as e:
                print(f"[WARNING] GPU/OpenCL setup failed: {e}")
        else:
            print(f"[WARNING] GPU check failed: {gpu_msg}")

        print("[INFO] GPU-OpenCV-OpenCL not available, switching to CPU inference mode...")
        return "CPU", False, {}

    elif inference_device == "CPU-50%":
        # CPU-50% is plain CPU inference, just capped to fewer cores (and pinned to the A76 big
        # cluster) so it does not saturate all 8 cores — the device stays usable. Always available.
        print("[INFO] CPU-50% mode: capped CPU inference (leaves the little cores free).")
        return "CPU-50%", False, {}

    # For CPU mode or fallback
    print("[INFO] Using CPU inference mode.")
    return "CPU", False, {}


def disable_unnecessary_logging():
    """Disable logging for unnecessary messages."""
    logger = logging.getLogger()
    logger.disabled = True
