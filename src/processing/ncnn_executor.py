# =============================================================================
# ncnn_executor.py
# NCNN inference executor with Vulkan GPU acceleration (Mali-G610 / RK3588).
# Interface mirrors ONNX_model_container so it can be used as a drop-in.
# by fcascan 2026
#
# ─── RK3588 / Mali-G610 NCNN notes ──────────────────────────────────────────
#
# [1] Vulkan device auto-selection bug
#     NCNN ranks Vulkan devices by an internal r-score and auto-selects the
#     highest.  On RK3588 the llvmpipe software renderer scores ~14 while the
#     real Mali-G610 hardware scores ~9, so NCNN defaults to llvmpipe (CPU) even
#     when use_vulkan_compute=True.  Fix: enumerate devices with get_gpu_info()
#     and call net.set_vulkan_device(idx) before load_param().  The call order
#     matters — setting the device after load_param() has no effect.
#
# [2] fp16 DFL precision bug (Mali-G610)
#     Enabling any fp16 path (packed / storage / arithmetic) causes the DFL
#     (Distribution Focal Loss) box regression to degenerate on Mali-G610.
#     DFL decodes box edges by running a 16-bin softmax distribution per side;
#     in fp16 the softmax saturates and all predicted distances collapse to the
#     maximum bin (15), which maps to full-frame coordinates (0, 0, W, H).
#     The class score branch is unaffected — detections still fire — but box
#     regression is completely wrong.  Fix: force fp32 by setting:
#         net.opt.use_fp16_packed = False
#         net.opt.use_fp16_storage = False
#         net.opt.use_fp16_arithmetic = False
#     Cost: roughly 20 % lower throughput vs fp16.
#     These options must be set before load_param() to take effect.
#
# [3] NCNN output format for YOLO11 (pnnx export)
#     The exported model produces a single blob named "out0" with shape
#     [nc+4, 8400] where:
#       rows 0-3   : box coords cx, cy, w, h in pixel space (already DFL-decoded)
#       rows 4-end : per-class confidences, already sigmoided
#     The 8400 anchors are ordered by stride:
#       0    – 6399  : stride-8  (P3, 80×80 grid, small objects)
#       6400 – 7999  : stride-16 (P4, 40×40 grid, medium objects)
#       8000 – 8399  : stride-32 (P5, 20×20 grid, large objects)
#     The blob name and layout are fixed by the pnnx NCNN export and match
#     what post_process_ncnn() in src/rockchip/yolo11_infer.py expects.
#
# [4] NCNN model export bug — broken class head for stride-8/16 (pnnx issue)
#     When exporting YOLO11 with `yolo export format=ncnn`, pnnx converts through
#     a PNNX IR stage.  On some Ultralytics versions this breaks the Concat node
#     that assembles class scores from all three FPN levels: the cv3 class branches
#     for P3 (stride-8) and P4 (stride-16) are disconnected from the output while
#     cv3 for P5 (stride-32) connects correctly.  Result: stride-8 and stride-16
#     class scores are always ~0; only stride-32 anchors ever trigger.  This makes
#     the model detect only huge-box objects regardless of actual object size.
#     Diagnosis: watch for "stride band breakdown: stride-8=0  stride-16=0
#     stride-32=N" in post_process_ncnn() logs.
#     Fix options (in order of preference):
#       A. Re-export with recent Ultralytics (>=8.2.0):
#              yolo export model=X.pt format=ncnn imgsz=640 half=False
#          Verify with netron that all three cv3 branches reach the output Concat.
#       B. Export via ONNX then convert:
#              yolo export model=X.pt format=onnx imgsz=640 opset=17
#              onnx2ncnn X.onnx X.param X.bin
#              ncnnoptimize X.param X.bin X_opt.param X_opt.bin 0
#          Note: onnx2ncnn produces per-scale blobs (not the single "out0"
#          tensor), so post_process_ncnn() would need to be adapted.
#
# [5] pyncnn API compatibility
#     Older pyncnn builds (pre-2023) may not expose set_vulkan_device() or
#     get_gpu_info().  The constructor handles this with AttributeError fallbacks.
#     Minimum recommended version: ncnn>=1.0.20230223 (installable via pip).
#
# [6] Input tensor layout
#     Input must be [3, H, W] float32 (CHW, NOT batch-first) when passed to
#     pyncnn.Mat().  The batch dimension [1, 3, H, W] must be removed before
#     constructing the Mat.  pyncnn.Mat(chw_array).clone() is used to ensure
#     the Mat owns its data rather than holding a numpy view that could be
#     garbage-collected mid-inference.
# =============================================================================

import os
import numpy as np

# One-time diagnostic flag: print raw output shape/stats on first inference call.
_ncnn_diag_done = False


class NCNN_model_container:
    """
    NCNN model container with optional Vulkan GPU acceleration.

    Expected input:  list containing one numpy array of shape [1, 3, H, W]
                     float32 normalised to [0, 1] (CHW, batch-first).
    Expected output: list containing one numpy array of shape [nc+4, 8400]
                     — first 4 rows are decoded box coords (cx, cy, w, h) in
                     pixel space; remaining rows are class confidences
                     (already sigmoided, one row per class).
    """

    def __init__(self, model_dir, use_vulkan=True):
        try:
            import ncnn as pyncnn
        except ImportError:
            raise ImportError(
                "ncnn Python package is not installed. "
                "Install it with:  pip install ncnn"
            )

        self._pyncnn = pyncnn
        self.model_path = str(model_dir)

        # Locate .param / .bin files
        if os.path.isdir(model_dir):
            param_files = [f for f in os.listdir(model_dir) if f.endswith(".param")]
            if not param_files:
                raise FileNotFoundError(f"No .param file found in {model_dir}")
            param_file = os.path.join(model_dir, sorted(param_files)[0])
        elif str(model_dir).endswith(".param"):
            param_file = str(model_dir)
        else:
            raise ValueError(
                f"model_dir must be a directory containing a .param file "
                f"or a direct path to a .param file; got: {model_dir}"
            )

        bin_file = param_file[:-len(".param")] + ".bin"
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"NCNN .bin file not found: {bin_file}")

        self.net = pyncnn.Net()
        self.net.opt.use_vulkan_compute = use_vulkan

        # Force Mali-G610 (Vulkan device 0) instead of letting NCNN auto-select.
        # NCNN ranks devices by r-score; on RK3588 llvmpipe scores higher than Mali,
        # so without explicit selection NCNN defaults to the software renderer.
        if use_vulkan:
            try:
                gpu_count = pyncnn.get_gpu_count()
                print(f"[INFO] NCNN Vulkan GPU count: {gpu_count}")

                # Find the first non-llvmpipe device (prefer real hardware).
                target_idx = 0
                for i in range(gpu_count):
                    try:
                        name = pyncnn.get_gpu_info(i).device_name()
                        print(f"[INFO] NCNN Vulkan device {i}: {name}")
                        if "llvmpipe" not in name.lower() and "software" not in name.lower():
                            target_idx = i
                            break
                    except Exception:
                        pass

                self.net.set_vulkan_device(target_idx)
                print(f"[INFO] NCNN: using Vulkan device {target_idx} (Mali-G610)")
            except AttributeError:
                # Older pyncnn builds may not expose set_vulkan_device or get_gpu_info.
                try:
                    self.net.set_vulkan_device(0)
                    print("[INFO] NCNN: forced Vulkan device 0")
                except Exception as e:
                    print(f"[WARNING] NCNN: cannot force Vulkan device: {e}")

        # Force fp32 precision — Mali-G610 fp16 causes DFL softmax to degenerate,
        # producing full-frame boxes instead of precise small boxes.  The class
        # scores branch is unaffected so detections trigger, but box regression
        # is wrong.  fp32 gives correct boxes at the cost of ~20 % throughput.
        try:
            self.net.opt.use_fp16_packed = False
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_arithmetic = False
        except AttributeError:
            pass

        ret = self.net.load_param(param_file)
        if ret != 0:
            raise RuntimeError(f"Failed to load NCNN param: {param_file} (ret={ret})")
        ret = self.net.load_model(bin_file)
        if ret != 0:
            raise RuntimeError(f"Failed to load NCNN model: {bin_file} (ret={ret})")

    def run(self, input_datas):
        """
        Run NCNN inference.

        Args:
            input_datas: list with one numpy array, shape [1, 3, H, W] float32.

        Returns:
            list with one numpy array of shape [nc+4, 8400].
        """
        global _ncnn_diag_done

        pyncnn = self._pyncnn
        inp = input_datas[0]

        # Remove batch dimension: [1, 3, H, W] → [3, H, W]
        if inp.ndim == 4:
            inp = inp[0]
        if inp.dtype != np.float32:
            inp = inp.astype(np.float32)

        # ncnn.Mat(chw_array) creates a 3-D Mat (c, h, w) from a CHW numpy array
        mat_in = pyncnn.Mat(inp).clone()

        with self.net.create_extractor() as ex:
            ex.input("in0", mat_in)
            ret, out0 = ex.extract("out0")
            if ret != 0:
                raise RuntimeError(f"NCNN inference failed (ret={ret})")

        output = np.array(out0)

        # One-time diagnostic: input sanity-check + raw output shape/ranges.
        if not _ncnn_diag_done:
            _ncnn_diag_done = True

            # ── Input stats ──────────────────────────────────────────────────
            print(f"[NCNN DIAG] Input: shape={inp.shape}  dtype={inp.dtype}  "
                  f"range=[{inp.min():.4f}, {inp.max():.4f}]  "
                  f"mean={inp.mean():.4f}  non-zero={int((inp > 0).sum())}")

            # ── Output stats ─────────────────────────────────────────────────
            print(f"[NCNN DIAG] Output: shape={output.shape}  dtype={output.dtype}  "
                  f"global range=[{output.min():.4f}, {output.max():.4f}]")
            if output.ndim == 2:
                print(f"[NCNN DIAG] Per-row statistics (row = feature channel):")
                for i in range(output.shape[0]):
                    row_label = (
                        ["cx", "cy", "w", "h"] + [f"score_cls{j}" for j in range(output.shape[0]-4)]
                    )[i] if i < output.shape[0] else f"row{i}"
                    print(f"[NCNN DIAG]   row[{i}] {row_label:12s}: "
                          f"min={output[i].min():.4f}  max={output[i].max():.4f}  "
                          f"mean={output[i].mean():.4f}  "
                          f"non-zero={int((output[i] != 0).sum())}")
            elif output.ndim == 3:
                print(f"[NCNN DIAG] 3-D output (unexpected) shape={output.shape}:")
                for i in range(min(output.shape[0], 8)):
                    print(f"[NCNN DIAG]   ch[{i}]: "
                          f"min={output[i].min():.4f}  max={output[i].max():.4f}")
            else:
                print(f"[NCNN DIAG] Unexpected ndim={output.ndim}")

        return [output]

    def release(self):
        if self.net is not None:
            del self.net
            self.net = None
