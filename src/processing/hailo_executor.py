# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
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
from collections import deque

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

# Power-measurement enums (for the continuous averaged measurement). Guarded — location varies by
# build; absent entirely on boards without an INA231 sensor (power then reports None).
try:
    from hailo_platform import (DvmTypes, PowerMeasurementTypes, AveragingFactor,
                                SamplingPeriod, MeasurementBufferIndex)
except Exception:
    try:
        from hailo_platform.pyhailort.pyhailort import (DvmTypes, PowerMeasurementTypes,
            AveragingFactor, SamplingPeriod, MeasurementBufferIndex)
    except Exception:
        DvmTypes = PowerMeasurementTypes = AveragingFactor = SamplingPeriod = MeasurementBufferIndex = None


# One shared VDevice for the whole process, with the round-robin scheduler enabled
# so multiple containers (camera streams) time-share the single Hailo-8 fairly.
_VDEVICE = None
_VDEVICE_LOCK = threading.Lock()

# ---- Live device stats for the web System Monitor + performance report ------
# HailoRT 4.24 exposes NO Python device-utilization API (no query_performance_stats /
# nnc_utilization; verified against the pyhailort binary). So "Load" is a DEVICE-OCCUPANCY
# busy-fraction: the sum of device inference wall-time over a trailing ~1 s window / that window
# (0-100%). It is computed from ONE rolling event log fed by every stream's run(), and the SAME
# hailo_device_occupancy() value is (a) shown live on the monitor AND (b) recorded per-frame into
# each stream's CSV -> the live card and the report AGREE, and it is a DEVICE-LEVEL figure (like the
# RKNPU/GPU sysfs loads), independent of stream count. (The earlier per-stream detect/frame ratio in
# the report read ~1/N of the live device value — N streams each ~19% vs the ~55% aggregate.)
# It reads well below 100% because the single-process, GIL-bound host pipeline can't keep the Hailo's
# queue full: the accelerator is HOST-BOUND, not compute-bound (measured ~1.3 W, ~102 fps nano) ->
# the low power + headroom are REAL, not a measurement error.
#
# Temperature + power are single-shot control reads off the shared VDevice's physical device, taken
# ONLY at the ~500 ms monitor cadence (NEVER per-frame — PCIe control round-trips contend with
# inference and tank FPS). Power is throttled to >= ~3 s between reads and cached, to limit the
# "overcurrent protection" warning HailoRT prints on each overcurrent-DVM measurement.
# (A CONTINUOUS averaged measurement was tried but set/start_power_measurement reconfigures the
# sensor into a mode where get_chip_temperature() ALSO fails -> temp and power both went blank;
# the single-shot read auto-restores protection, so temperature stays readable.)
_STATS_LOCK = threading.Lock()
_OCC_WINDOW = 1.0                       # trailing window (s) for occupancy / fps / latency
_infer_events = deque()                 # (end_time, duration_s) of recent device inferences (all streams)
_last_temp = None
_last_power = None
_last_power_t = 0.0
_POWER_MIN_INTERVAL = 3.0               # min seconds between single-shot power reads (caps the warning)


def _record_infer(dt):
    """Log one device inference (called by every stream's run())."""
    now = time.time()
    with _STATS_LOCK:
        _infer_events.append((now, float(dt)))
        cutoff = now - _OCC_WINDOW
        while _infer_events and _infer_events[0][0] < cutoff:
            _infer_events.popleft()


def _device_stats(now=None):
    """(occupancy%, fps, latency_ms) over the trailing window across ALL streams; (0,0,None) idle."""
    if now is None:
        now = time.time()
    cutoff = now - _OCC_WINDOW
    with _STATS_LOCK:
        while _infer_events and _infer_events[0][0] < cutoff:
            _infer_events.popleft()
        evts = list(_infer_events)
    if not evts:
        return 0.0, 0.0, None
    busy = sum(d for _, d in evts)
    occ = min(100.0, busy / _OCC_WINDOW * 100.0)
    fps = len(evts) / _OCC_WINDOW
    latency_ms = busy / len(evts) * 1000.0
    return occ, fps, latency_ms


def hailo_device_occupancy():
    """Current DEVICE occupancy % over the trailing window (cheap; NO device I/O). Recorded per-frame
    by the stream workers so the report matches the live monitor. 0.0 when idle."""
    return _device_stats()[0]


def hailo_env():
    """Last (temperature_C, power_W) the monitor sampled (cached; NO device I/O). (None, None) until
    the first sample. Recorded per-frame by the workers so the report can show avg temp/power."""
    return _last_temp, _last_power


def get_hailo_stats():
    """Stats for the System Monitor:
      {'utilization': %|None, 'latency_ms': ms|None, 'temperature': degC|None, 'power': W|None, 'fps': float}.
    utilization = device-occupancy busy-fraction (HOST-BOUND -> well below 100). Sampled at ~500 ms;
    NEVER per-frame (temp/power are PCIe control reads). None for util when idle."""
    global _last_temp, _last_power, _last_power_t
    now = time.time()
    occ, fps, latency_ms = _device_stats(now)
    util = occ if fps > 0 else None
    temp = power = None
    # Read temp/power INLINE, holding the physical Device in LOCAL scope ONLY for the duration of the
    # reads, then releasing it (del). This is critical:
    #  * The Device MUST stay referenced WHILE its .control is used, or HailoRT raises "The device in
    #    use has been released" (that blanked temp/power when a helper returned phys[0].control and let
    #    `phys` go out of scope).
    #  * But the Device must NOT be CACHED/held across calls: keeping the physical-device handle open
    #    blocks the inference output (D2H) vstreams -> HAILO_TIMEOUT and inference dies. So we acquire
    #    it fresh each ~500 ms sample and release it immediately (the original working pattern).
    if _VDEVICE is not None:
        try:
            phys = _VDEVICE.get_physical_devices()
            if phys:
                ctrl = phys[0].control
                try:
                    temp = round(float(ctrl.get_chip_temperature().ts0_temperature), 1)
                    _last_temp = temp
                except Exception:
                    temp = _last_temp
                # Single-shot power (whole-module W via AUTO/overcurrent DVM), throttled + cached.
                # ~1 W is REAL (host-bound). Each read warns once + auto-restores overcurrent protection.
                if now - _last_power_t >= _POWER_MIN_INTERVAL:
                    try:
                        _last_power = round(float(ctrl.power_measurement()), 2)
                    except Exception:
                        _last_power = None
                    _last_power_t = now
                power = _last_power
                del ctrl, phys   # release the physical-device handle immediately (do NOT hold it)
        except Exception:
            pass
    return {'utilization': util,
            'latency_ms': round(latency_ms, 2) if latency_ms is not None else None,
            'temperature': temp, 'power': power,
            'fps': round(fps, 1)}


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

        self.last_infer_s = 0.0   # device infer time of the most recent run() (per-frame latency)
        logging.info("[Hailo] configured %s | inputs=%s outputs=%d",
                     os.path.basename(self.model_path), self.input_name, len(self.output_names))

    def run(self, input_datas):
        # input_datas[0]: NHWC uint8 (1, H, W, 3) from preprocess_frame ('hailo' path).
        data = np.ascontiguousarray(input_datas[0])
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)

        t0 = time.time()
        results = self._pipe.infer({self.input_name: data})
        dt = time.time() - t0
        self.last_infer_s = dt
        _record_infer(dt)

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
        # Close the infer pipeline; keep the shared VDevice alive for other streams / process lifetime.
        try:
            if getattr(self, "_pipe_cm", None) is not None:
                self._pipe_cm.__exit__(None, None, None)
        except Exception:
            pass
        self._pipe = None
        self._pipe_cm = None
        self.network_group = None
