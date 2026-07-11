# =============================================================================
# opencv_executor.py
# YOLO11 GPU inference on the Mali-G610 via OpenCV-DNN with the OpenCL target.
# by fcascan 2026
#
# Why this exists (and not ncnn for GPU):
#   ncnn + Vulkan numerically mis-computes YOLO11 on this Mali-G610: the
#   stride-8/16 class heads collapse to 0.0 on most frames and saturate to 1.0
#   (garbage) on others — confirmed identical on ncnn's CPU and Vulkan backends,
#   across multiple ncnn versions, while the SAME model runs correctly on the NPU
#   (rknn) and CPU (onnxruntime).  OpenCV-DNN's OpenCL backend, by contrast,
#   computes YOLO11 correctly on the Mali-G610 (verified: detections match the
#   CPU/NPU boxes to within a couple of pixels) — just slower.  So GPU mode runs
#   the ONNX model through OpenCV-DNN/OpenCL and decodes with the same
#   post_process() used by the CPU and NPU paths.
#
# Interface mirrors ONNX_model_container so it is a drop-in for setup_model():
#   run(input_datas) -> list of output ndarrays.  For the Rockchip 9-output head
#   the outputs are re-ordered to the [box, cls, sum] x 3-scales layout that
#   post_process() expects (OpenCV returns them grouped differently than ONNX).
#
# -----------------------------------------------------------------------------
# OpenCL teardown crash guard (see release() and _install_opencl_exit_guard):
#   Destroying a cv2.dnn.Net built on OpenCV's OpenCL (T-API) backend, or letting
#   OpenCV tear its OpenCL context down during interpreter / static-destructor
#   exit, intermittently aborts the whole process on this Mali stack with:
#       OpenCV(4.11.0) .../ocl.cpp: (-215:Assertion failed) u->mapcount == 0
#       in function 'deallocate'
#   It only fires AFTER all inference and result files are done (harmless to the
#   data) but it kills the app with a C++ std::terminate that no Python try/except
#   can catch.  Its likelihood scales with how many UMats the run left in OpenCV's
#   OpenCL buffer pool and with worker-thread teardown ordering, so it is
#   intermittent.  We cannot patch OpenCV's C++ destructors from Python, so we
#   avoid the buggy path entirely:
#     1. The loaded Net is cached at module scope and is never destroyed while the
#        process runs (release() only drops the caller's reference).  This removes
#        the mid-run destructor the crash rides on, and lets repeated GPU runs
#        reuse the already-compiled OpenCL kernels (faster engine startup).
#     2. A single atexit hook flushes stdio and calls os._exit(0), terminating the
#        process before OpenCV's static destructors can run the OpenCL deallocate
#        path.
#   Both engage ONLY when GPU/OpenCL inference is actually used: NPU and CPU modes
#   never construct this container, so their shutdown is completely untouched.
# =============================================================================

import os
import sys
import atexit
import threading

import numpy as np
import cv2


# Persistent OpenCL kernel-tuning cache.  Without it, OpenCV's ocl4dnn re-runs the
# (multi-second) convolution auto-tuning on every process start — the slow first
# frame and the "consider to specify kernel configuration cache directory" warning.
# Pointing OPENCV_OCL4DNN_CONFIG_PATH at a writable dir makes the tuned configs
# persist across runs, so only the very first run pays the tuning cost.
_OCL4DNN_CACHE = os.path.expanduser('~/.cache/PythonYoloRKNPU/ocl4dnn')
try:
    os.makedirs(_OCL4DNN_CACHE, exist_ok=True)
    os.environ.setdefault('OPENCV_OCL4DNN_CONFIG_PATH', _OCL4DNN_CACHE)
except Exception:
    pass


# model_path -> cv2.dnn.Net.  Intentionally keeps the Net alive for the whole
# process lifetime so it is never destroyed (see module header / release()).
_NET_CACHE = {}
_EXIT_GUARD_INSTALLED = False

# Serializes inference on the single cached Net.  The web layer shares one cached
# cv2.dnn.Net (per model_path) across all stream/camera worker threads, and
# cv2.dnn.Net.setInput()/forward() are stateful and NOT thread-safe: concurrent
# calls on one OpenCL Net segfault on this Mali stack (surfaces at >1 instance).
# The Mali is a single GPU with one shared OpenCL context, so time-sharing it
# under a lock is correct (multi-instance is serialized, not parallel — same as
# the single Hailo accelerator's _VDEVICE_LOCK in hailo_executor.py).
_INFER_LOCK = threading.Lock()


def _install_opencl_exit_guard():
    """Register a one-shot process-exit hook that bypasses OpenCV's buggy OpenCL
    static teardown.  Installed the first time an OpenCL net is built."""
    global _EXIT_GUARD_INSTALLED
    if _EXIT_GUARD_INSTALLED:
        return
    _EXIT_GUARD_INSTALLED = True

    def _hard_exit():
        # Flush our own output, then terminate before the C++ static destructors
        # (which abort with "u->mapcount == 0 in deallocate" on this Mali OpenCL
        # stack) get a chance to run.  os._exit skips them cleanly.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

    atexit.register(_hard_exit)


class OpenCV_OpenCL_model_container:
    def __init__(self, model_path, use_opencl=True):
        if not str(model_path).endswith('.onnx'):
            raise ValueError(
                f"OpenCV GPU executor requires an .onnx model, got: {model_path}")
        self.model_path = str(model_path)

        net = _NET_CACHE.get(self.model_path)
        if net is None:
            net = cv2.dnn.readNetFromONNX(self.model_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

            target = cv2.dnn.DNN_TARGET_CPU
            if use_opencl and cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                # fp32 only: OpenCV-DNN restricts DNN_TARGET_OPENCL_FP16 to Intel
                # GPUs and silently falls back to fp32 on Mali, so there is no
                # fp16 speed-up to be had here.
                target = cv2.dnn.DNN_TARGET_OPENCL
                try:
                    d = cv2.ocl.Device_getDefault()
                    print(f"[INFO] OpenCV-DNN GPU target: OpenCL on {d.name()} "
                          f"({d.vendorName()}, OpenCL {d.OpenCLVersion()})")
                except Exception:
                    print("[INFO] OpenCV-DNN GPU target: OpenCL")
                print("[INFO] Note: a '-cl-no-subgroup-ifp' / CL_INVALID_BUILD_OPTIONS line "
                      "may appear below — it is a benign Intel-only kernel probe by OpenCV and "
                      "does not affect results (the Mali path just uses the generic kernel).")
            else:
                print("[WARNING] OpenCL not available — OpenCV-DNN falling back to CPU target")
            net.setPreferableTarget(target)

            _NET_CACHE[self.model_path] = net
            _install_opencl_exit_guard()
        else:
            print(f"[INFO] OpenCV-DNN GPU: reusing cached OpenCL net "
                  f"for {os.path.basename(self.model_path)}")

        self.net = net
        self._out_names = self.net.getUnconnectedOutLayersNames()

    def run(self, input_datas):
        inp = np.ascontiguousarray(np.asarray(input_datas[0], dtype=np.float32))
        # setInput + forward mutate the shared cached Net, so they must run as one
        # critical section: the Net is shared across all worker threads and is not
        # thread-safe (see _INFER_LOCK). The output regroup below is on local
        # arrays and stays outside the lock.
        with _INFER_LOCK:
            self.net.setInput(inp)
            outs = [np.asarray(o, dtype=np.float32) for o in self.net.forward(self._out_names)]

        # Rockchip 9-output head: OpenCV returns the 9 blobs grouped as
        # [box*3, sum*3, cls*3]; post_process() wants per FPN scale [box, cls, sum].
        # Re-group by spatial size; within a scale pick by channel count
        # (box = 64ch reg, cls = nc ch, sum = 1ch).
        if len(outs) == 9 and all(o.ndim == 4 for o in outs):
            by_hw = {}
            for o in outs:
                by_hw.setdefault(o.shape[2], {})[o.shape[1]] = o
            if all(len(g) == 3 for g in by_hw.values()):
                ordered = []
                for hw in sorted(by_hw.keys(), reverse=True):  # 80, 40, 20
                    g = by_hw[hw]
                    chs = sorted(g.keys())          # e.g. [1, nc, 64]
                    ordered += [g[chs[-1]], g[chs[1]], g[chs[0]]]  # box, cls, sum
                return ordered
        return outs

    def release(self):
        # Do NOT destroy the OpenCL net here (see module header): tearing a
        # cv2.dnn.Net built on the OpenCL backend down on this Mali stack aborts
        # the process.  The net stays cached in _NET_CACHE for reuse; final
        # teardown is handled by the os._exit(0) exit guard.  We only drop this
        # container's reference.
        self.net = None
