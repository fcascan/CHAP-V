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
# =============================================================================

import numpy as np
import cv2


class OpenCV_OpenCL_model_container:
    def __init__(self, model_path, use_opencl=True):
        if not str(model_path).endswith('.onnx'):
            raise ValueError(
                f"OpenCV GPU executor requires an .onnx model, got: {model_path}")
        self.model_path = str(model_path)
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        target = cv2.dnn.DNN_TARGET_CPU
        if use_opencl and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            target = cv2.dnn.DNN_TARGET_OPENCL  # fp32; Mali OpenCL_FP16 unsupported
            try:
                d = cv2.ocl.Device_getDefault()
                print(f"[INFO] OpenCV-DNN GPU target: OpenCL on {d.name()} "
                      f"({d.vendorName()}, OpenCL {d.OpenCLVersion()})")
            except Exception:
                print("[INFO] OpenCV-DNN GPU target: OpenCL")
        else:
            print("[WARNING] OpenCL not available — OpenCV-DNN falling back to CPU target")
        self.net.setPreferableTarget(target)
        self._out_names = self.net.getUnconnectedOutLayersNames()

    def run(self, input_datas):
        inp = np.ascontiguousarray(np.asarray(input_datas[0], dtype=np.float32))
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
        self.net = None
