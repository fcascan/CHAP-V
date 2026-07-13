# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
# =============================================================================
# mnn_executor.py
# YOLO11 GPU inference on the Mali-G610 via MNN (Alibaba) with the OpenCL backend.
# by fcascan 2026
#
# Why this exists (and how it differs from the OpenCV-OpenCL GPU mode):
#   GPU-OpenCV-OpenCL runs the ONNX model through OpenCV-DNN's OpenCL backend.
#   It is numerically correct on this Mali-G610 but very slow (~2.4 s/frame),
#   because OpenCV-DNN does not fuse SiLU, has no Winograd, and forces fp32 on
#   non-Intel GPUs.  MNN's OpenCL backend is purpose-built for mobile GPUs
#   (operator fusion, fp16 compute, kernel auto-tuning), and on this device it
#   runs the SAME YOLO11 graph correctly and ~18x faster (~135 ms/frame at fp16).
#   Verified on detectS2: every output head (incl. the stride-8/16 class heads
#   that ncnn+Vulkan corrupted) matches the CPU result with cosine >= 0.9993 at
#   fp16 and >= 0.9999 at fp32.
#
# Interface mirrors ONNX_model_container / OpenCV_OpenCL_model_container so it is
# a drop-in for setup_model():
#   run(input_datas) -> list of output ndarrays.  For the Rockchip 9-output head
#   the outputs are re-ordered to the per-FPN-scale [box, cls, sum] layout that
#   post_process() expects (MNN returns them keyed by tensor name, grouped
#   differently than the decoder wants).
#
# -----------------------------------------------------------------------------
# Model: this mode uses a dedicated .mnn model (PATHS.model_mnn), converted from
#   the same ONNX in Google Colab with `mnnconvert -f ONNX`.  The .mnn is fp32 on
#   disk; the COMPUTE precision is chosen at runtime via the session config
#   ('low' = fp16, 'high' = fp32), so one model file serves both precisions.
#
# OpenCL loader note: MNN's OpenCLWrapper dlopens "libOpenCL.so" (bare name) and
#   a few absolute paths, but NOT "libOpenCL.so.1" nor the aarch64-linux-gnu
#   multiarch path.  This device ships only libOpenCL.so.1, so a bare
#   "libOpenCL.so" symlink must be reachable by the dynamic loader, otherwise MNN
#   prints "Can't create Runtime: OPENCL" and falls back to CPU.  Provisioning of
#   that symlink is handled at startup by src/core/system_setup.py (the app runs
#   as root); this module only verifies/loads it best-effort and logs clearly.
#
# CRITICAL — OpenCV-OpenCL conflict: OpenCV's OpenCL (T-API) and MNN's OpenCL
#   contend for the single Mali OpenCL context in the same process.  When OpenCV's
#   OpenCL is active (it auto-enables when the Mali ICD is present), MNN's OpenCL
#   computes garbage — the FIRST inference may be correct, then every output is
#   NaN.  GPU-MNN only uses OpenCV for CPU-side preprocessing (letterbox/cvtColor),
#   so we disable OpenCV's OpenCL (cv2.ocl.setUseOpenCL(False)) before creating the
#   MNN session.  This is local to the GPU-MNN process; it does not affect the
#   separate GPU-OpenCV-OpenCL mode, which runs on its own.
#
# CRITICAL — output reading: outputs are read via copyToHostTensor() into NCHW
#   host tensors, NOT getNumpyData() directly.  The OpenCL output tensors are in
#   MNN's packed GPU layout; reading them raw across a sequence of distinct frames
#   corrupts the session (first couple frames correct, then all-NaN).  Both this
#   and the OpenCV-OpenCL conflict were found during standalone validation — they
#   pass a single-frame check but only surface on a real multi-frame video stream.
#
# OpenCL kernel-tuning: the first OpenCL run compiles and auto-tunes the conv
#   kernels (~tens of seconds, first frame only).  MNN's on-disk persistence of
#   that tuning (setCacheFile + updateCacheFile) is deliberately NOT used: on this
#   Mali stack updateCacheFile() corrupts the live session (first frame correct,
#   all later frames NaN).  So tuning is re-done once per process start.
# =============================================================================

import os
import ctypes
import logging
import threading

import numpy as np

try:
    import MNN
except Exception as _e:  # pragma: no cover - import guard
    MNN = None
    _MNN_IMPORT_ERROR = _e


# (model_path, backend, precision) -> (Interpreter, Session).  Kept alive for the
# whole process so repeated GPU runs reuse the compiled/auto-tuned OpenCL kernels
# (the expensive first-run tuning is paid only once).
_SESSION_CACHE = {}
_OPENCL_LOADER_CHECKED = False

# Serializes inference on the single cached MNN session.  The web layer shares one
# cached (Interpreter, Session) — keyed by (model, backend, precision) — across all
# stream/camera worker threads, and the session plus the shared input tensor are
# stateful: concurrent copyFrom()/runSession()/copyToHostTensor() on one OpenCL
# session are not thread-safe on this Mali stack. One GPU + one shared OpenCL
# context means time-sharing under a lock is correct (mirrors hailo_executor's
# _VDEVICE_LOCK; and the OpenCV GPU path's _INFER_LOCK).
_INFER_LOCK = threading.Lock()


def _ensure_opencl_loadable():
    """Best-effort: make sure MNN can dlopen the OpenCL ICD loader.

    The dynamic loader fixes its search path at process start, so we cannot
    repair a missing 'libOpenCL.so' here by editing LD_LIBRARY_PATH; the real
    provisioning (a bare-name symlink to libOpenCL.so.1) is done by
    system_setup.py at startup.  Here we only ctypes-preload the loader so any
    later failure surfaces with a clear message instead of a silent CPU fallback.
    """
    global _OPENCL_LOADER_CHECKED
    if _OPENCL_LOADER_CHECKED:
        return
    _OPENCL_LOADER_CHECKED = True
    for name in ('libOpenCL.so', 'libOpenCL.so.1'):
        try:
            ctypes.CDLL(name, mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
            return
        except OSError:
            continue
    logging.warning(
        "[MNN] Could not preload an OpenCL loader (libOpenCL.so[.1]); if GPU-MNN "
        "falls back to CPU, ensure a bare 'libOpenCL.so' symlink exists "
        "(system_setup provisions it, or: sudo ln -sf "
        "/usr/lib/aarch64-linux-gnu/libOpenCL.so.1 /usr/lib/aarch64-linux-gnu/libOpenCL.so).")


def _disable_opencv_opencl():
    """Turn OpenCV's OpenCL (T-API) off so it does not contend with MNN for the
    Mali OpenCL context (their coexistence makes MNN return NaN — see header)."""
    try:
        import cv2
        if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
            cv2.ocl.setUseOpenCL(False)
            logging.info("[MNN] disabled OpenCV-OpenCL (T-API) to free the Mali "
                         "OpenCL context for MNN")
    except Exception:
        pass


def _regroup_rockchip_outputs(outs):
    """Re-order MNN's 9 head outputs into the per-FPN-scale [box, cls, sum]
    layout post_process() expects.

    Same heuristic as the OpenCV executor: group by spatial size; within each
    scale pick by channel count (box = 64-ch DFL reg, cls = nc-ch, sum = 1-ch).
    """
    if len(outs) == 9 and all(o.ndim == 4 for o in outs):
        by_hw = {}
        for o in outs:
            by_hw.setdefault(o.shape[2], {})[o.shape[1]] = o
        if all(len(g) == 3 for g in by_hw.values()):
            ordered = []
            for hw in sorted(by_hw.keys(), reverse=True):  # 80, 40, 20
                g = by_hw[hw]
                chs = sorted(g.keys())                      # e.g. [1, nc, 64]
                ordered += [g[chs[-1]], g[chs[1]], g[chs[0]]]  # box, cls, sum
            return ordered
    return outs


class MNN_model_container:
    def __init__(self, model_path, backend="OPENCL", precision="low", num_thread=4):
        if MNN is None:
            raise ImportError(f"pymnn (MNN) is not available: {_MNN_IMPORT_ERROR}")
        if not str(model_path).endswith('.mnn'):
            raise ValueError(f"MNN executor requires a .mnn model, got: {model_path}")

        self.model_path = str(model_path)
        self.backend = str(backend).upper()
        self.precision = str(precision).lower()

        if self.backend == "OPENCL":
            _ensure_opencl_loadable()
            _disable_opencv_opencl()

        key = (self.model_path, self.backend, self.precision)
        cached = _SESSION_CACHE.get(key)
        if cached is None:
            interp = MNN.Interpreter(self.model_path)
            # NOTE: MNN's OpenCL tuning-cache persistence (setCacheFile +
            # updateCacheFile) is intentionally NOT used here. On this Mali stack,
            # calling updateCacheFile() after the first inference corrupts the live
            # OpenCL session: the first frame is correct, every later frame returns
            # NaN. Disabling it keeps results correct at the cost of re-running the
            # ~tens-of-seconds kernel auto-tuning on each process start (first frame
            # only). A safe persistence path can be revisited later.
            config = {"backend": self.backend, "precision": self.precision}
            # IMPORTANT: do NOT set "numThread" for the OpenCL backend. MNN
            # repurposes that field as a GPU tuning/mode bitmask (not a CPU thread
            # count) on non-CPU backends; a nonzero value here selects a GPU mode
            # that returns NaN for every inference on this Mali stack. It is only a
            # real thread count for the CPU backend.
            if self.backend != "OPENCL":
                config["numThread"] = int(num_thread)
            session = interp.createSession(config)

            # Report the backend that was actually created (3 = OpenCL, 0 = CPU);
            # MNN silently falls back to CPU if OpenCL can't initialize.
            # pymnn exposes getSessionInfo with integer codes (2 = BACKENDS); the
            # C++ enum MNN.Interpreter.BACKENDS is NOT a python attribute.
            try:
                bt = interp.getSessionInfo(session, 2)  # 2 = BACKENDS
                bt = bt[0] if isinstance(bt, (list, tuple)) else bt
                if self.backend == "OPENCL" and bt != 3:
                    logging.warning(
                        "[MNN] requested OPENCL but session backendType=%s "
                        "(0=CPU) — running on CPU. Check the libOpenCL.so symlink.", bt)
                else:
                    logging.info("[MNN] session backendType=%s, precision=%s",
                                 bt, self.precision)
            except Exception:
                pass

            _SESSION_CACHE[key] = (interp, session)
        else:
            interp, session = cached
            logging.info("[MNN] reusing cached session for %s (%s/%s)",
                         os.path.basename(self.model_path), self.backend, self.precision)

        self.interp = interp
        self.session = session
        self._input = self.interp.getSessionInput(self.session)

    def run(self, input_datas):
        data = np.ascontiguousarray(np.asarray(input_datas[0], dtype=np.float32))
        shape = self._input.getShape()
        tmp = MNN.Tensor(shape, MNN.Halide_Type_Float, data,
                         MNN.Tensor_DimensionType_Caffe)   # Caffe = NCHW
        # copyFrom -> runSession -> output read all mutate/read the shared session,
        # shared input tensor and output tensors, so they run as one critical
        # section: the session is shared across all worker threads and is not
        # thread-safe (see _INFER_LOCK). The regroup below is on local arrays and
        # stays outside the lock.
        with _INFER_LOCK:
            self._input.copyFrom(tmp)
            self.interp.runSession(self.session)

            # Read each output via copyToHostTensor into an NCHW (Caffe) host tensor.
            # Do NOT use getNumpyData() directly on the OpenCL output tensors: those
            # live in MNN's packed GPU layout, and reading them raw across a sequence
            # of distinct inputs corrupts the session (first ~2 frames are correct,
            # then every output goes NaN). copyToHostTensor does the proper GPU->host
            # NCHW sync and is stable frame after frame.
            outputs = self.interp.getSessionOutputAll(self.session)
            outs = []
            for t in outputs.values():
                shape = t.getShape()
                host = MNN.Tensor(shape, t.getDataType(),
                                  np.zeros(shape, dtype=np.float32),
                                  MNN.Tensor_DimensionType_Caffe)
                t.copyToHostTensor(host)
                outs.append(np.array(host.getNumpyData(), dtype=np.float32, copy=True))

        return _regroup_rockchip_outputs(outs)

    def release(self):
        # Keep the interpreter/session cached for process lifetime (so repeated runs
        # reuse the already-compiled OpenCL kernels); only drop this container's refs.
        self.interp = None
        self.session = None
        self._input = None
