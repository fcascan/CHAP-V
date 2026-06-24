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
# [3] NCNN output format — two variants depending on conversion tool
#
#     Variant A — pnnx export (Ultralytics default):
#       Single blob "out0", shape [nc+4, 8400]:
#         rows 0-3   : cx, cy, w, h in pixel space (DFL-decoded inside model)
#         rows 4-end : per-class confidences, sigmoided
#       Anchor ordering: stride-8 [0–6399], stride-16 [6400–7999], stride-32 [8000–8399]
#       Handled by: post_process_ncnn() in src/rockchip/yolo11_infer.py
#       Detected by: "out0" appears in the .param file
#
#     Variant B — onnx2ncnn conversion (recommended, avoids bugs in [4] and [5]):
#       9 output blobs matching the ONNX model graph.output tensors:
#         [1, 64, 80, 80]  stride-8  DFL raw (16 bins × 4 sides)
#         [1,  2, 80, 80]  stride-8  class scores
#         [1,  1, 80, 80]  stride-8  objectness
#         [1, 64, 40, 40]  stride-16 DFL raw
#         [1,  2, 40, 40]  stride-16 class scores
#         [1,  1, 40, 40]  stride-16 objectness
#         [1, 64, 20, 20]  stride-32 DFL raw
#         [1,  2, 20, 20]  stride-32 class scores
#         [1,  1, 20, 20]  stride-32 objectness
#       Input blob name: "images"  (from ONNX graph.input)
#       Handled by: post_process() in src/rockchip/yolo11_infer.py (same as CPU/ONNX)
#       Detected by: "out0" absent; input blob is "images"
#
#     NCNN_model_container auto-detects the variant from the .param file at load
#     time and exposes it via self.model_format ('pnnx' or 'onnx2ncnn').
#     postprocess_outputs() in yolo11_inference.py routes accordingly.
#
# [4] NCNN model export bug — wrong anchor_x encoding (pnnx issue)
#     Confirmed by inspecting model.ncnn.param: the dist2bbox section uses blobs
#     309/310 as anchor x-coordinates.  After export, the stored values have a
#     wrong step size (~64px per grid cell instead of the correct 32px for
#     stride-32).  As a result, cx for stride-32 anchors exceeds 640px
#     (observed: 870–1057 for anchors whose correct cx is 304–400).  When
#     post_process_ncnn() computes x1=cx−w/2, both x1 and x2 exceed the frame
#     width, clip to the right edge, and produce zero-area boxes.  The cy / h
#     values are unaffected (anchor_y is stored correctly).
#     Fix: the zero-area boxes are now discarded in post_process_ncnn() before
#     NMS.  But detections are effectively suppressed until the model is
#     re-exported correctly.
#
# [5] NCNN model export bug — broken class head for stride-8/16 (pnnx issue)
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
# [6] pyncnn API compatibility
#     Older pyncnn builds (pre-2023) may not expose set_vulkan_device() or
#     get_gpu_info().  The constructor handles this with AttributeError fallbacks.
#     Minimum recommended version: ncnn>=1.0.20230223 (installable via pip).
#
# [7] Input tensor layout
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


def _parse_ncnn_model_format(param_file):
    """
    Parse an NCNN .param file and return (model_format, input_blob, output_blobs).

    model_format:  'pnnx'      — Ultralytics pnnx export; single 'out0' blob
                   'onnx2ncnn' — onnx2ncnn conversion; 9 FPN output blobs
    input_blob:    name of the network's input blob ('in0' or 'images')
    output_blobs:  ordered list of network output blob names
    """
    all_outputs = []
    all_inputs_seen = set()
    input_layer_blobs = []

    with open(param_file, 'r') as f:
        lines = f.readlines()

    for line in lines[2:]:       # skip magic + counts header
        parts = line.split()
        if len(parts) < 5:
            continue
        layer_type = parts[0]
        try:
            num_in  = int(parts[2])
            num_out = int(parts[3])
        except ValueError:
            continue

        in_start  = 4
        out_start = in_start + num_in

        for i in range(num_in):
            idx = in_start + i
            if idx < len(parts) and '=' not in parts[idx]:
                all_inputs_seen.add(parts[idx])
        for i in range(num_out):
            idx = out_start + i
            if idx < len(parts) and '=' not in parts[idx]:
                all_outputs.append(parts[idx])

        if layer_type == 'Input':
            for i in range(num_out):
                idx = out_start + i
                if idx < len(parts) and '=' not in parts[idx]:
                    input_layer_blobs.append(parts[idx])

    # Network outputs are blobs that are produced but never consumed.
    net_outputs = [b for b in all_outputs if b not in all_inputs_seen]

    # PNNX names network outputs out0, out1, ... regardless of how many there
    # are, so the presence of 'out0' does NOT distinguish the two formats.
    # Detect by output COUNT instead:
    #   1 output  -> Ultralytics-style single decoded blob ([nc+4, 8400])
    #   9 outputs -> Rockchip/raw multi-tensor (box/cls/sum per FPN scale)
    # The old 'out0' name check misclassified the 9-blob PNNX-from-ONNX model as
    # single-blob 'pnnx', so only out0 (a raw [1,64,H,W] box-reg tensor) was
    # extracted and fed to post_process_ncnn() as if decoded -> garbage boxes.
    if len(net_outputs) == 1:
        fmt = 'pnnx'
        in_blob = input_layer_blobs[0] if input_layer_blobs else 'in0'
    else:
        fmt = 'onnx2ncnn'
        in_blob = input_layer_blobs[0] if input_layer_blobs else 'images'

    return fmt, in_blob, net_outputs


class NCNN_model_container:
    """
    NCNN model container with optional Vulkan GPU acceleration.

    Supports two model formats (auto-detected from the .param file):

    pnnx (Ultralytics export):
      Input:  list[ndarray[1,3,H,W] float32]
      Output: list[ndarray[nc+4, 8400]] — decoded boxes + sigmoided class scores

    onnx2ncnn (ONNX → onnx2ncnn conversion, recommended):
      Input:  list[ndarray[1,3,H,W] float32]
      Output: list of 9 ndarray[1,ch,H,W] — raw FPN tensors matching ONNX output;
              compatible with post_process() in yolo11_infer.py (same as CPU mode)

    Use self.model_format ('pnnx' or 'onnx2ncnn') to pick the right postprocess call.
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

        # Detect model format from param topology.
        self.model_format, self._input_blob, self._output_blobs = \
            _parse_ncnn_model_format(param_file)
        print(f"[INFO] NCNN model format: {self.model_format}  "
              f"input={self._input_blob}  outputs={self._output_blobs}")

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

    def _prepare_input(self, input_datas):
        """Remove batch dim and ensure float32. Returns (inp_chw, mat)."""
        pyncnn = self._pyncnn
        inp = input_datas[0]
        if inp.ndim == 4:
            inp = inp[0]                    # [1,3,H,W] → [3,H,W]
        if inp.dtype != np.float32:
            inp = inp.astype(np.float32)
        mat_in = pyncnn.Mat(inp).clone()    # Mat owns its data (no dangling view)
        return inp, mat_in

    def run(self, input_datas):
        """
        Run NCNN inference.  Dispatches to the correct format handler.

        Args:
            input_datas: list with one numpy array, shape [1, 3, H, W] float32.

        Returns:
            pnnx:      list[ndarray[nc+4, 8400]]  — decoded boxes + class scores
            onnx2ncnn: list of 9 ndarray[1,ch,H,W] — raw FPN tensors (same as
                       onnx_executor.run() output, compatible with post_process())
        """
        if self.model_format == 'pnnx':
            return self._run_pnnx(input_datas)
        else:
            return self._run_onnx2ncnn(input_datas)

    def _run_pnnx(self, input_datas):
        """Handle the pnnx single-blob output format."""
        global _ncnn_diag_done

        inp, mat_in = self._prepare_input(input_datas)

        with self.net.create_extractor() as ex:
            ex.input(self._input_blob, mat_in)
            ret, out0 = ex.extract("out0")
            if ret != 0:
                raise RuntimeError(f"NCNN inference failed (ret={ret})")

        output = np.array(out0)

        # One-time diagnostic.
        if not _ncnn_diag_done:
            _ncnn_diag_done = True
            print(f"[NCNN DIAG] Input: shape={inp.shape}  dtype={inp.dtype}  "
                  f"range=[{inp.min():.4f}, {inp.max():.4f}]  "
                  f"mean={inp.mean():.4f}  non-zero={int((inp > 0).sum())}")
            print(f"[NCNN DIAG] Output (pnnx): shape={output.shape}  "
                  f"dtype={output.dtype}  "
                  f"global range=[{output.min():.4f}, {output.max():.4f}]")
            if output.ndim == 2:
                for i in range(output.shape[0]):
                    row_label = (
                        ["cx", "cy", "w", "h"]
                        + [f"score_cls{j}" for j in range(output.shape[0] - 4)]
                    )[i]
                    print(f"[NCNN DIAG]   row[{i}] {row_label:12s}: "
                          f"min={output[i].min():.4f}  max={output[i].max():.4f}  "
                          f"mean={output[i].mean():.4f}  "
                          f"non-zero={int((output[i] != 0).sum())}")

        return [output]

    def _run_onnx2ncnn(self, input_datas):
        """
        Handle the onnx2ncnn 9-blob output format.

        Returns a list of 9 numpy arrays shaped [1, ch, H, W], matching the
        output of onnx_executor.run() so that post_process() works unchanged.
        """
        global _ncnn_diag_done

        inp, mat_in = self._prepare_input(input_datas)
        results = []

        with self.net.create_extractor() as ex:
            ex.input(self._input_blob, mat_in)
            for blob_name in self._output_blobs:
                ret, out = ex.extract(blob_name)
                if ret != 0:
                    raise RuntimeError(
                        f"NCNN failed to extract blob '{blob_name}' (ret={ret})"
                    )
                arr = np.array(out)
                results.append(arr[np.newaxis])  # [ch,H,W] → [1,ch,H,W]

        # One-time diagnostic.
        if not _ncnn_diag_done:
            _ncnn_diag_done = True
            print(f"[NCNN DIAG] Input: shape={inp.shape}  dtype={inp.dtype}  "
                  f"range=[{inp.min():.4f}, {inp.max():.4f}]  "
                  f"mean={inp.mean():.4f}  non-zero={int((inp > 0).sum())}")
            print(f"[NCNN DIAG] Output (onnx2ncnn): {len(results)} blobs")
            for name, arr in zip(self._output_blobs, results):
                print(f"[NCNN DIAG]   {name}: shape={arr.shape}  "
                      f"range=[{arr.min():.4f}, {arr.max():.4f}]")

        return results

    def release(self):
        if self.net is not None:
            del self.net
            self.net = None
