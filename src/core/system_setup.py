# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""system_setup.py
System setup and configuration for CHAP-V project
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
    if inference_device.startswith("RKNPU"):
        # RKNPU setup (RK3588 on-chip NPU via RKNN; modes RKNPU-AUTO / RKNPU-DISTRIBUTED)
        if check_rknn_availability():
            try:
                from importlib import import_module
                RKNNLite = import_module("rknnlite.api.rknn_lite").RKNNLite
                from src.utils.my_htop import log_npu_usage
                print("[INFO] RKNN (RKNPU) libraries loaded successfully.")
                # NOTE: inference decoding uses src/rockchip/yolo11_infer.py::post_process (the live,
                # Apache-2.0-attributed decoder). The old src/utils/rknn_post_processing.py duplicate was
                # removed as dead code.
                return inference_device, True, {"RKNNLite": RKNNLite, "log_npu_usage": log_npu_usage}
            except ImportError as e:
                print(f"[WARNING] RKNN (RKNPU) libraries could not be imported: {e}")

        print("[INFO] RKNPU not available, switching to CPU inference mode...")
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

    elif inference_device == "GPU-MNN":
        # Runs the .mnn model on the Mali-G610 via MNN (Alibaba) + OpenCL.
        try:
            from importlib import import_module
            import_module("MNN")
            _provision_opencl_so()  # MNN dlopens a bare "libOpenCL.so"; this device ships only .so.1
            print("[INFO] GPU-MNN available: MNN + OpenCL on the Mali-G610.")
            return "GPU-MNN", True, {"gpu_available": True}
        except ImportError:
            print("[WARNING] pymnn (MNN) not installed — GPU-MNN requires the MNN OpenCL build "
                  "(see CLAUDE.md / README). ")
        except Exception as e:
            print(f"[WARNING] GPU-MNN setup failed: {e}")
        print("[INFO] GPU-MNN not available, switching to CPU inference mode...")
        return "CPU", False, {}

    elif inference_device == "NPU-HAILO8":
        # Runs a .hef on the Hailo-8 external NPU (M.2/PCIe) via HailoRT + pyhailort.
        import os as _os
        try:
            from importlib import import_module
            import_module("hailo_platform")
            if _os.path.exists("/dev/hailo0"):
                print("[INFO] NPU-Hailo8 available: HailoRT + Hailo-8 on /dev/hailo0.")
                return "NPU-HAILO8", True, {"hailo_available": True}
            print("[WARNING] pyhailort present but /dev/hailo0 missing — is the HailoRT PCIe "
                  "driver loaded? (sudo modprobe hailo_pci)")
        except ImportError:
            print("[WARNING] pyhailort (hailo_platform) not installed — NPU-Hailo8 requires the "
                  "HailoRT runtime + wheel (see CLAUDE.md / README).")
        except Exception as e:
            print(f"[WARNING] NPU-Hailo8 setup failed: {e}")
        print("[INFO] NPU-Hailo8 not available, switching to CPU inference mode...")
        return "CPU", False, {}

    # For CPU mode or fallback
    print("[INFO] Using CPU inference mode.")
    return "CPU", False, {}


def _provision_opencl_so():
    """Ensure a bare 'libOpenCL.so' exists for MNN's OpenCL loader.

    MNN's OpenCLWrapper dlopens "libOpenCL.so" and "/usr/lib/libOpenCL.so", but this
    device ships only "libOpenCL.so.1". Create the symlink(s) so MNN finds the (working,
    OpenCV-shared) ICD loader. Best-effort; needs root, which the app runs as. This only
    adds a generic loader symlink — it does NOT touch the Mali driver or its ICD vendor file.
    """
    import os
    real = "/usr/lib/aarch64-linux-gnu/libOpenCL.so.1"
    if not os.path.exists(real):
        return
    for link in ("/usr/lib/libOpenCL.so", "/usr/lib/aarch64-linux-gnu/libOpenCL.so"):
        if os.path.exists(link):
            continue
        try:
            os.symlink(real, link)
            print(f"[INFO] GPU-MNN: created OpenCL loader symlink {link} -> {real}")
        except PermissionError:
            print(f"[WARNING] GPU-MNN: need root to create {link}; run once: sudo ln -sf {real} {link}")
        except Exception:
            pass


def disable_unnecessary_logging():
    """Disable logging for unnecessary messages."""
    logger = logging.getLogger()
    logger.disabled = True
