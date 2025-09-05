# -*- coding: utf-8 -*-
"""system_setup.py
System setup and configuration for YOLO RKNN/NPU project
by fcascan 2025
"""
import sys
import logging
from .dependency_manager import check_and_install_dependencies, check_rknn_availability, check_gpu_availability, ensure_root_permissions


def setup_system():
    """Complete system setup including dependencies and permissions."""
    # Check dependencies before proceeding
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Verify root permissions
    ensure_root_permissions()
    
    return True


def setup_inference_device(inference_device):
    """Setup and configure the inference device (NPU, GPU, or CPU)."""
    if inference_device == "NPU":
        # NPU setup
        if check_rknn_availability():
            try:
                from rknnlite.api import RKNNLite
                from src.utils.rknn_post_processing import post_process
                from src.utils.my_htop import log_npu_usage
                print("[INFO] RKNN NPU libraries loaded successfully.")
                return "NPU", True, {"RKNNLite": RKNNLite, "post_process": post_process, "log_npu_usage": log_npu_usage}
            except ImportError as e:
                print(f"[WARNING] RKNN NPU libraries could not be imported: {e}")
        
        print("[INFO] NPU not available, switching to CPU inference mode...")
        return "CPU", False, {}
    
    elif inference_device == "GPU":
        # GPU setup
        if check_gpu_availability():
            try:
                import cv2
                # Test GPU functionality
                test_mat = cv2.cuda_GpuMat()
                print("[INFO] GPU acceleration available and functional.")
                return "GPU", True, {"gpu_available": True}
            except Exception as e:
                print(f"[WARNING] GPU setup failed: {e}")
        
        print("[INFO] GPU not available, switching to CPU inference mode...")
        return "CPU", False, {}
    
    # For CPU mode or fallback
    print("[INFO] Using CPU inference mode.")
    return "CPU", False, {}


def disable_unnecessary_logging():
    """Disable logging for unnecessary messages."""
    logger = logging.getLogger()
    logger.disabled = True