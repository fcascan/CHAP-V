# -*- coding: utf-8 -*-
"""system_setup.py
System setup and configuration for YOLO RKNN/NPU project
by fcascan 2025
"""
import sys
import logging
from .dependency_manager import check_and_install_dependencies, check_rknn_availability, ensure_root_permissions


def setup_system():
    """Complete system setup including dependencies and permissions."""
    # Check dependencies before proceeding
    if not check_and_install_dependencies():
        sys.exit(1)
    
    # Verify root permissions
    ensure_root_permissions()
    
    return True


def setup_inference_device(inference_device):
    """Setup and configure the inference device (NPU or CPU)."""
    rknn_available = False
    
    if inference_device == "NPU":
        # First check if RKNN is available, try to install if not
        if check_rknn_availability():
            try:
                from rknnlite.api import RKNNLite
                from src.utils.rknn_post_processing import post_process
                from src.utils.my_htop import log_npu_usage
                rknn_available = True
                print("[INFO] RKNN NPU libraries loaded successfully.")
                return "NPU", rknn_available, {"RKNNLite": RKNNLite, "post_process": post_process, "log_npu_usage": log_npu_usage}
            except ImportError as e:
                print(f"[WARNING] RKNN NPU libraries could not be imported: {e}")
                rknn_available = False
        
        if not rknn_available:
            print("[INFO] Switching to CPU inference mode...")
            return "CPU", False, {}
    
    # For CPU mode or fallback
    return "CPU", False, {}


def disable_unnecessary_logging():
    """Disable logging for unnecessary messages."""
    logger = logging.getLogger()
    logger.disabled = True