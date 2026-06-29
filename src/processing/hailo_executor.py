# =============================================================================
# hailo_executor.py
# YOLO11 inference on the Hailo-8 (26 TOPS) external NPU over M.2/PCIe, via HailoRT.
# by fcascan 2026
#
# Why this exists:
#   The Hailo-8 is a dedicated edge AI accelerator on the OrangePi's M.2 slot. This
#   runs the YOLO11 model compiled to a .hef (Hailo Executable Format) on it, as a
#   5th inference backend ("NPU-Hailo8") alongside RKNPU / CPU / CPU-50% /
#   GPU-OpenCV-OpenCL / GPU-MNN.
#
# Interface mirrors the other executors (ONNX_/MNN_/OpenCV_OpenCL_model_container)
# so it is a drop-in for setup_model():
#   run(input_datas) -> list of output ndarrays.  The .hef is compiled WITHOUT
#   on-chip NMS (raw FPN, 9 heads), so the SAME post_process() used by every other
#   backend decodes the result.  Hailo emits NHWC tensors -> run() transposes each
#   to NCHW and re-groups them to the per-FPN-scale [box, cls, sum] layout that
#   post_process() expects (same shape+channel heuristic as mnn_executor).
#
# Model: PATHS.model_hailo8 (a .hef converted from the SAME Rockchip 9-head ONNX
#   in the Hailo Dataflow Compiler — see the conversion notebook / README).
#
# Multi-stream: the Hailo-8 is ONE accelerator (not 3 cores like the RKNPU). A
#   single module-level VDevice is created with the ROUND_ROBIN scheduler, so up to
#   3 camera streams (separate Hailo_model_container instances sharing the VDevice)
#   time-share the device fairly. No per-core pinning.
#
# Input: NHWC uint8 (1,H,W,3) — the same letterboxed-uint8 tensor the RKNN path
#   uses (preprocess_frame leaves 'hailo' on the uint8 branch). Normalization is
#   baked into the HEF at compile time.
#
# >>> VALIDATION STATUS: the inference path (vstream creation + infer call + output
#     dtype/layout) is written from the pyhailort 4.24 API surface + docs but has
#     NOT been runtime-tested (requires a real .hef). Expect to fine-tune run() /
#     the configure path during end-to-end validation, the way the MNN executor
#     needed adjustments once it ran on real frames. <<<
#
# HailoRT runtime install (one-time, already done on this device): HailoRT 4.24.0
#   PCIe driver (DKMS) + runtime .deb + pyhailort cp312 aarch64 wheel.
# =============================================================================

import os
import time
import logging
import threading

import numpy as np

try:
    import hailo_platform as hp
    from hailo_platform import (
        VDevice, HEF, ConfigureParams, HailoStreamInterface,
        InferVStreams, InputVStreamParams, OutputVStreamParams,
        FormatType, HailoSchedulingAlgorithm,
    )
except Exception as _e:  # pragma: no cover - import guard (pyhailort may be absent)
    hp = None
    _HAILO_IMPORT_ERROR = _e


# One shared VDevice for the whole process, with the round-robin scheduler enabled
# so multiple containers (camera streams) time-share the single Hailo-8 fairly.
_VDEVICE = None
_VDEVICE_LOCK = threading.Lock()

# ---- Live utilization stats for the web System Monitor ----------------------
# pyhailort 4.24 has no direct utilization API, so we report a busy-fraction:
# inference wall-time accumulated across all streams divided by elapsed time.
# Temperature is read from the shared VDevice's physical device.
_STATS_LOCK = threading.Lock()
_busy_seconds = 0.0
_infer_count = 0
_last_sample_t = None


def _add_busy(seconds):
    global _busy_seconds, _infer_count
    with _STATS_LOCK:
        _busy_seconds += float(seconds)
        _infer_count += 1


def get_hailo_stats():
    """Stats for the System Monitor: {'utilization': %|None, 'temperature': degC|None, 'power': W|None, 'fps': float}.
    utilization = Hailo inference busy-time / wall-time since the previous call (capped at 100; an
    approximation since concurrent streams overlap). None until first sample / when idle. Safe to
    call when the Hailo is idle or absent."""
    global _busy_seconds, _infer_count, _last_sample_t
    now = time.time()
    with _STATS_LOCK:
        busy, cnt, last = _busy_seconds, _infer_count, _last_sample_t
        _busy_seconds = 0.0
        _infer_count = 0
        _last_sample_t = now
    util, fps = None, 0.0
    if last is not None:
        elapsed = max(1e-6, now - last)
        util = min(100.0, (busy / elapsed) * 100.0)
        fps = cnt / elapsed
    temp = None
    power = None
    try:
        if _VDEVICE is not None:
            phys = _VDEVICE.get_physical_devices()
            if phys:
                _ctrl = phys[0].control
                temp = round(float(_ctrl.get_chip_temperature().ts0_temperature), 1)
                # Real on-board power sensor (W). Use SINGLE-shot reads: they auto-restore the
                # over-current protection, unlike continuous measurement. ~0.9 W idle on the M.2.
                try:
                    power = round(float(_ctrl.power_measurement()), 2)
                except Exception:
                    power = None
    except Exception:
        temp = None
    # NOTE: this is sampled by the web System Monitor at low cadence (~500 ms). Never call it
    # per-frame: temp/power are PCIe control round-trips that contend with inference (FPS drop).
    return {'utilization': util, 'temperature': temp, 'power': power, 'fps': round(fps, 1)}


def _get_vdevice():
    global _VDEVICE
    if _VDEVICE is None:
        with _VDEVICE_LOCK:
            if _VDEVICE is None:
                params = VDevice.create_params()
                # ROUND_ROBIN: fair time-sharing of the one accelerator across the
                # (up to 3) configured network groups / streams.
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                _VDEVICE = VDevice(params)
                logging.info("[Hailo] VDevice created (scheduler=ROUND_ROBIN)")
    return _VDEVICE


def _regroup_rockchip_outputs(outs):
    """Re-order the 9 head outputs into the per-FPN-scale [box, cls, sum] layout
    post_process() expects (same heuristic as the MNN/OpenCV executors): group by
    spatial size; within each scale pick by channel count (box = 64-ch, cls = nc,
    sum = 1). Inputs here are already NCHW."""
    if len(outs) == 9 and all(o.ndim == 4 for o in outs):
        by_hw = {}
        for o in outs:
            by_hw.setdefault(o.shape[2], {})[o.shape[1]] = o
        if all(len(g) == 3 for g in by_hw.values()):
            ordered = []
            for hw in sorted(by_hw.keys(), reverse=True):   # 80, 40, 20
                g = by_hw[hw]
                chs = sorted(g.keys())                        # e.g. [1, nc, 64]
                ordered += [g[chs[-1]], g[chs[1]], g[chs[0]]]  # box, cls, sum
            return ordered
    return outs


class Hailo_model_container:
    def __init__(self, model_path, target=None, device_id=None):
        if hp is None:
            raise ImportError(f"pyhailort (hailo_platform) is not available: {_HAILO_IMPORT_ERROR}")
        if not str(model_path).endswith('.hef'):
            raise ValueError(f"Hailo executor requires a .hef model, got: {model_path}")

        self.model_path = str(model_path)
        self.hef = HEF(self.model_path)
        vdev = _get_vdevice()

        # Configure the network group on the shared VDevice (PCIe interface).
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = vdev.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        # Stream info / names.
        self.input_infos = self.hef.get_input_vstream_infos()
        self.output_infos = self.hef.get_output_vstream_infos()
        self.input_name = self.input_infos[0].name
        self.output_names = [o.name for o in self.output_infos]

        # Input as uint8 (normalization is baked into the HEF); outputs dequantized
        # to float32 for post_process.
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8)
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32)

        # Persistent infer pipeline (entered once, reused per frame).
        self._pipe_cm = InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params)
        self._pipe = self._pipe_cm.__enter__()

        logging.info("[Hailo] configured %s | inputs=%s outputs=%d",
                     os.path.basename(self.model_path), self.input_name, len(self.output_names))

    def run(self, input_datas):
        # input_datas[0]: NHWC uint8 (1, H, W, 3) from preprocess_frame ('hailo' path).
        data = np.ascontiguousarray(input_datas[0])
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)

        t0 = time.time()
        results = self._pipe.infer({self.input_name: data})
        _add_busy(time.time() - t0)

        # Hailo returns a dict {name: ndarray} (NHWC). Transpose each to NCHW for
        # post_process, then re-group to [box, cls, sum] per scale.
        outs = []
        for name in self.output_names:
            o = np.asarray(results[name], dtype=np.float32)
            if o.ndim == 4:                      # NHWC -> NCHW
                o = np.transpose(o, (0, 3, 1, 2))
            outs.append(np.ascontiguousarray(o))
        return _regroup_rockchip_outputs(outs)

    def release(self):
        # Close the infer pipeline; keep the shared VDevice alive for other streams /
        # process lifetime.
        try:
            if getattr(self, "_pipe_cm", None) is not None:
                self._pipe_cm.__exit__(None, None, None)
        except Exception:
            pass
        self._pipe = None
        self._pipe_cm = None
        self.network_group = None
