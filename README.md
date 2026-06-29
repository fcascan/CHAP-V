# Python YOLO RKNN/NPU

Real-time object detection using YOLO v11 models on RK3588 (with Orange Pi 5 Max).
A project for comparison between CPU, GPU, RKNPU & Hailo-8 inference with **Web Interface** support.

## Features

- **Web Interface**: Modern web UI with real-time video streaming and console output
- **Multi-camera support**: Up to 3 USB cameras with RKNPU core assignment
- **Auto-installation**: Intelligent dependency detection and installation
- **RKNPU & Hailo-8 acceleration**: RKNN toolkit / HailoRT with CPU fallback
- **Real-time inference**: Live object detection with statistics
- **Web Configuration**: Change settings and control processing via web interface

## Requirements

- RK3588 device (Orange Pi 5/5+/5 Max)
- Python 3.7+ (aarch64)
- RKNN Toolkit Lite 2.3.2
- OpenCV, NumPy
- USB cameras (optional when using benchmark mode)
- Web browser for web interface

## Quick Start

### Installation (Recommended)
To avoid conflicts with operating system packages (PEP 668), this project uses a virtual environment with Python 3.12.
Simply run the automatic installation script:

```bash
git clone https://github.com/fcascan/PythonYoloRKNPU.git
cd PythonYoloRKNPU
chmod +x setup.sh
./setup.sh
```

### Console Mode (Traditional)
```bash
git clone https://github.com/fcascan/PythonYoloRKNPU.git
cd PythonYoloRKNPU
sudo python3 main.py
```

### Web Interface Mode
```bash
# Install web dependencies
sudo pip3 install Flask Flask-SocketIO eventlet

# Start web interface
sudo python3 main.py --web

# Or use the dedicated web starter
sudo python3 start_web.py
```

Then open your browser to: **http://your-device-ip:8080**

## Web Interface Features

- **Real-time Video Streaming**: See live detection results in your browser
- **Console Output**: All terminal messages displayed in web interface
- **Configuration Panel**: Change inference device, mode, and camera settings
- **Control Buttons**: Start/stop processing remotely
- **System Information**: Real-time status and performance metrics
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Configuration

All settings live in [`config.ini`](config.ini). A subset can also be changed at runtime from the web
UI's *Configuration* panel (which writes them back to `config.ini`); the rest are file-only. The
**Web** column marks which is which.

| Section | Key | Values / format | Web |
|---|---|---|:---:|
| `[MODE]` | `benchmark_mode` | `true` (video file) / `false` (camera) | ✅ |
| `[INFERENCE]` | `inference_device` | `RKNPU-Auto` / `RKNPU-Distributed` / `CPU` / `CPU-50%` / `GPU-OpenCV-OpenCL` / `GPU-MNN` / `NPU-Hailo8` | ✅ |
| | `max_inference_instances` | `1`–`3` parallel streams/cameras (RKNPU-Distributed pins stream N → RKNN core N) | ✅ |
| | `debug_mode` | `true` / `false` (verbose logging) | ✅ |
| | `rockchip_target` | RKNPU platform, e.g. `rk3588` | — |
| | `obj_threshold`, `nms_threshold` | `0.0`–`1.0` | — |
| | `max_detections_per_frame` | int; `0` disables the corrupt-frame guard | — |
| | `cpu50_threads`, `cpu50_affinity` | int · CSV core ids — **CPU-50%** mode | — |
| | `mnn_precision`, `mnn_backend` | `low`/`high`/`normal` · `OPENCL`/`CPU` — **GPU-MNN** mode | — |
| `[PATHS]` | `model_rknn` | `.rknn` path — used by **RKNPU** | ✅ |
| | `model_onnx` | `.onnx` path — used by **CPU / CPU-50% / GPU-OpenCV-OpenCL** | ✅ |
| | `model_mnn` | `.mnn` path — used by **GPU-MNN** | ✅ |
| | `model_hailo8` | `.hef` path — used by **NPU-Hailo8** | ✅ |
| | `model_labels` | class-names file path | — |
| | `benchmark_video_0..N` | benchmark video path, one per stream | — |
| `[WEB]` | `enabled`, `host`, `port` | `true`/`false` · bind IP · port | — |
| `[IMAGE]` | `img_width`, `img_height` | inference size, must match the model (`640`) | — |
| | overlay/text style | `show_overlay`, `fps_text_size`, `label_text_size`, `overlay_text_color`, `save_debug_frames` | — |
| `[DETECTION]` | box/label style | `box_color`, `label_text_color`, `label_background_color`, `box_thickness`, `label_text_size`, `label_text_thickness` | — |
| `[CLASSES]` | `default_labels` | comma-separated class names (fallback if no labels file) | — |

> **Web UI** (*Configuration* panel → **Save**) edits only the ✅ rows; everything else is set in
> `config.ini` directly. Colors accept `B,G,R` or `#RRGGBB`. Comments must be on their own line
> (`;` prefix — not after a value), and a literal `%` is allowed (e.g. `CPU-50%`).

## Usage Options

### Command Line Arguments
```bash
# Console mode
sudo python3 main.py

# Web interface mode
sudo python3 main.py --web
sudo python3 main.py --web --web-port 8080 --web-host 0.0.0.0

# Web interface with custom settings
sudo python3 start_web.py --port 8080 --host 0.0.0.0
```

### Inference Devices
- **RKNPU Mode**: `inference_device = RKNPU-Auto` / `RKNPU-Distributed` - Uses RKNN Lite API on the RK3588 on-chip NPU (3 cores; *Distributed* pins one camera stream per core, *Auto* uses Core 0)
- **GPU-OpenCV-OpenCL Mode**: `inference_device = GPU-OpenCV-OpenCL` - Runs the ONNX model on the Mali-G610 via OpenCV-DNN + OpenCL (numerically correct but slow; see [GPU-OpenCV-OpenCL Inference](#gpu-opencv-opencl-inference-mali-g610) below)
- **GPU-MNN Mode**: `inference_device = GPU-MNN` - Runs a dedicated `.mnn` model on the Mali-G610 via MNN + OpenCL (fp16) - the fast, numerically-correct GPU path (see [GPU-MNN Inference](#gpu-mnn-inference-mali-g610) below)
- **NPU-Hailo8 Mode**: `inference_device = NPU-Hailo8` - Runs a dedicated `.hef` model on the external **Hailo-8** (26 TOPS) M.2/PCIe accelerator via HailoRT (see [NPU-Hailo8 Inference](#npu-hailo8-inference-hailo-8) below)
- **CPU Mode**: `device = CPU` - Uses ONNX Runtime with the CPU backend

### Processing Modes
- **Camera mode**: `benchmark_mode = false` - Live camera processing
- **Video mode**: `benchmark_mode = true` - Benchmark video file processing

**Console Mode**: Press 'q' to exit. Results displayed on terminal.
**Web Mode**: Use web interface controls to start/stop processing.

## Web Interface Usage

1. **Start the server**: `sudo python3 main.py --web` or `sudo python3 start_web.py`
2. **Open browser**: Navigate to `http://your-device-ip:8080`
3. **Configure settings**: Use the configuration panel to set device and mode
4. **Start processing**: Click "Start Processing" to begin inference
5. **Monitor results**: Watch live video stream and console output
6. **Control remotely**: Stop/start processing from any device on your network

### Web Interface Sections

- **Video Display**: Real-time video with detection overlays and performance metrics
- **Control Panel**: Start/stop processing and refresh system status
- **Configuration**: Change processing mode, inference device, and camera settings
- **System Info**: Current status, processing mode, and frame availability
- **Console Output**: Real-time logging with color-coded message levels

## GPU-OpenCV-OpenCL Inference (Mali-G610)

> **Mode name:** `inference_device = GPU-OpenCV-OpenCL` (the legacy value `GPU` still works and
> maps to this mode). Named explicitly because a separate, faster GPU backend (MNN) may be added
> later as its own mode — this one is the OpenCV-DNN / OpenCL path.

**This mode runs the ONNX model on the Mali-G610 via OpenCV-DNN with the OpenCL target**
(`DNN_BACKEND_OPENCV` + `DNN_TARGET_OPENCL`), decoded by the same `post_process()` as the
CPU and NPU paths. It uses the **same ONNX** as CPU mode (`PATHS.model_onnx`), so all three
processors run identical weights — no separate GPU model or conversion needed.

Observed (benchmark.mp4, `detectS2`, obj 0.5): **correct detections matching CPU/NPU to within
~2 px**, ~2.3–2.5 s/frame (~0.4 FPS). It is slow, but it is a genuine, numerically-correct
Mali-GPU data point — Mali load sits at ~70–100% during inference. The mode is kept on purpose:
it offloads inference to the GPU, leaving most of the CPU free for other work.

> **Why this mode is slow.** The bottleneck is OpenCV-DNN's OpenCL backend (`ocl4dnn`), not the
> Mali hardware: it does not fuse YOLO11's SiLU activation (so each SiLU runs as a separate,
> memory-bandwidth-bound element-wise kernel — these dominate the runtime), its fast convolution
> kernels are Intel-only and fall back to a generic kernel on Mali (no Winograd), and OpenCV
> forces fp32 on non-Intel GPUs. The net effect is that OpenCV's *own* CPU backend is ~4× faster
> than its OpenCL backend on the same graph, and ONNX Runtime on CPU (fused + multithreaded) is
> faster still.

> **Why not ncnn + Vulkan?** ncnn was the first GPU candidate but numerically mis-computes YOLO11
> on this Mali-G610: the stride-8/16 class heads collapse to 0.0 (only stride-32 fires), so
> detections are garbage. The decisive test was that ncnn's **CPU and Vulkan backends produce the
> *same* garbage** (they agree with each other), while the **same model is correct on the NPU and
> on ONNX Runtime CPU** — proving the fault is ncnn's computation of this YOLO11 graph, not the
> Mali driver and not the model. No conversion path or ncnn version produced stable, correct
> boxes, so ncnn + Vulkan was dropped in favour of OpenCV-DNN / OpenCL.

### System dependencies (one-time setup)

GPU mode needs an OpenCL stack on top of the Python packages. On this OrangePi it is already
installed; on a fresh device set it up as follows.

**1. OpenCL ICD loader** (generic `libOpenCL.so` that apps link against):

```bash
sudo apt install ocl-icd-libopencl1
```

**2. Mali OpenCL userspace driver** — the `libmali` *valhall-g610* blob, which implements
OpenCL 3.0 for the Mali-G610. (Installed manually; it is not in the Ubuntu repos.)

```bash
# Download the g610 valhall blob (x11-wayland-gbm variant includes OpenCL) from
#   https://github.com/tsukumijima/libmali-rockchip/releases
sudo cp libmali-valhall-g610-g24p0-x11-wayland-gbm.so /usr/lib/aarch64-linux-gnu/
sudo ln -sf /usr/lib/aarch64-linux-gnu/libmali-valhall-g610-g24p0-x11-wayland-gbm.so \
            /usr/lib/aarch64-linux-gnu/libmali.so.1
```

**3. Register the Mali OpenCL ICD** so the loader finds the blob:

```bash
sudo mkdir -p /etc/OpenCL/vendors
echo "/usr/lib/aarch64-linux-gnu/libmali-valhall-g610-g24p0-x11-wayland-gbm.so" \
  | sudo tee /etc/OpenCL/vendors/mali.icd
```

**4. OpenCV with OpenCL** — `opencv-python` from PyPI already includes the OpenCL backend
(no rebuild needed).

> Note: this is the **OpenCL** path. The old ncnn experiment used Vulkan
> (`/etc/vulkan/icd.d/`); that is no longer used by this project and is not required.

### Verify the GPU is available

```bash
venv/bin/python -c "
import cv2
print('haveOpenCL:', cv2.ocl.haveOpenCL())
cv2.ocl.setUseOpenCL(True)
d = cv2.ocl.Device_getDefault()
print('Device:', d.name(), '| vendor:', d.vendorName(), '| OpenCL', d.OpenCLVersion())
"
# Expected: haveOpenCL: True   Device: Mali-G610 r0p0 | vendor: ARM | OpenCL OpenCL 3.0 ...
```

### Observed performance (benchmark.mp4, 1 stream, detectS2, obj 0.5)

| Mode | Backend | Inference (ms/frame) | FPS | Saturated unit |
|------|---------|----------------------|-----|----------------|
| RKNPU             | RKNN (rknnlite) | ~33 | ~29 | RKNPU |
| CPU               | ONNX Runtime (all 8 cores) | ~127 | ~7.6 | CPU 100% |
| CPU-50%           | ONNX Runtime (4 A76 threads) | ~340 | ~3 | CPU ~48% (A55 cores idle) |
| GPU-OpenCV-OpenCL | OpenCV-DNN / OpenCL (Mali) | ~2400 | ~0.4 | GPU ~70–100% |
| GPU-MNN (fp16)    | MNN / OpenCL (Mali) | ~140 | ~7 | GPU |
| GPU-MNN (fp32)    | MNN / OpenCL (Mali) | ~260 | ~4 | GPU |
| NPU-Hailo8        | HailoRT (Hailo-8, 26 TOPS) | _pending_ | _pending_ | Hailo-8 |

All modes produce correct, matching detections. GPU-OpenCV-OpenCL is slowest; **GPU-MNN is the fast,
correct GPU path** (~18× faster than OpenCV-OpenCL, ~CPU-parity in speed while offloading the CPU —
see [GPU-MNN Inference](#gpu-mnn-inference-mali-g610)). Figures are approximate; the first GPU-MNN run
adds a one-time ~50 s OpenCL kernel auto-tuning.

> The first GPU run auto-tunes the OpenCL convolution kernels (slower first frame); the tuned
> configs are cached under `~/.cache/PythonYoloRKNPU/ocl4dnn`, so later runs start faster.

---

## CPU-50% mode (`inference_device = CPU-50%`)

CPU-50% is **plain CPU inference, just capped so it does not saturate the whole CPU** — the device
stays usable for other things while it runs. It is identical to CPU mode except the ONNX Runtime
session is limited to `cpu50_threads` (default **4**) threads, pinned to the **A76 big cores**
(`cpu50_affinity = 4,5,6,7`), so the **A55 little cores stay free** for the OS/desktop. Same
`detectS2.onnx` — no extra model.

Measured on benchmark.mp4: **~340 ms/frame (~3 FPS)**, whole-device CPU **~48 %** (A76 ~85–88 %,
A55 ~9 %) — versus full CPU mode which hits ~127 ms/frame but pins all 8 cores at 100 %. The trade
is throughput for headroom; lower `cpu50_threads` (e.g. `2–3`) leaves even more CPU free.

> This bounds *core placement and count* (the A55 cores stay free), not *utilization* — the A76
> cluster can still run near 100 % while inferring. The point is the device as a whole is not
> saturated and stays responsive.
>
> A true CPU+GPU split for one stream was evaluated and dropped: with the current GPU ~19× slower
> than the CPU, it cannot beat CPU-alone (its frames arrive stale). A genuinely useful CPU+GPU mode
> would first require a faster GPU backend (e.g. MNN) — tracked separately.

Enable it by selecting **CPU-50%** in the web UI's *Inference Device* dropdown (or `inference_device =
CPU-50%`); tune the cap with `cpu50_threads` and `cpu50_affinity` — see [Configuration](#configuration).

---

## GPU-MNN Inference (Mali-G610)

> **Mode name:** `inference_device = GPU-MNN`. The fast, numerically-correct GPU path: it runs a
> dedicated `.mnn` model on the Mali-G610 via **MNN** (Alibaba) with the **OpenCL** backend
> (operator fusion + fp16 + kernel auto-tuning).

Unlike GPU-OpenCV-OpenCL (correct but ~2.4 s/frame), MNN's OpenCL backend computes the same YOLO11
graph **correctly and ~18× faster** (~140 ms/frame at fp16). Verified against the CPU path on
benchmark.mp4: every output head — including the stride-8/16 class heads that ncnn+Vulkan corrupted —
matches CPU at **cosine ≥ 0.9993 (fp16)** / **≥ 0.9999 (fp32)**, and per-frame detections match CPU.

**Precision is a runtime knob** (one `.mnn` file serves both fp16 and fp32): enable with
`inference_device = GPU-MNN` and set `mnn_precision` (default `low` = fp16) / `mnn_backend` — see
[Configuration](#configuration).

### Model
GPU-MNN uses its own model (`PATHS.model_mnn`, default `assets/models/detectS2.mnn`), converted from
the same ONNX in Google Colab. Convert to a plain fp32 `.mnn` (no `--fp16`) — the compute precision is
chosen at runtime via `mnn_precision`:

```bash
mnnconvert -f ONNX --modelFile detectS2.onnx --MNNModel detectS2.mnn --bizCode biz
```

### System dependencies (one-time setup)
The PyPI `mnn` wheel is **CPU-only** (no OpenCL backend). MNN with OpenCL must be **built from source**
(already done on this device):

```bash
git clone --depth 1 https://github.com/alibaba/MNN.git
cd MNN/pymnn/pip_package
# build_deps.py: set MNN_ARM82=OFF (gcc 15 ICEs on the ARM82 fp16 backend; not needed for GPU)
CMAKE_POLICY_VERSION_MINIMUM=3.5 venv/bin/python build_deps.py opencl   # cmake 4 policy compat
venv/bin/python setup.py install --deps opencl                         # installs into the project venv
```

MNN's OpenCL loader dlopens a bare `libOpenCL.so`, but this device ships only `libOpenCL.so.1`, so a
symlink is required (the app provisions it at startup when run as root, or set it once manually):

```bash
sudo ln -sf /usr/lib/aarch64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so
```

> **Notes.** GPU-MNN disables OpenCV's OpenCL (T-API) within its process so it does not contend with
> MNN for the single Mali OpenCL context (they conflict, producing NaN). The first run auto-tunes the
> OpenCL kernels (~50 s, first frame only). This mode is independent of GPU-OpenCV-OpenCL (unchanged).

---

## NPU-Hailo8 Inference (Hailo-8)

> **Mode name:** `inference_device = NPU-Hailo8`. Runs a dedicated `.hef` model on the external
> **Hailo-8** (26 TOPS) accelerator on the M.2/PCIe slot, via **HailoRT** — a separate dedicated NPU,
> independent of the RK3588's on-chip RKNPU.

It uses the **same Rockchip 9-head ONNX** as the other modes, compiled to a `.hef` **without on-chip
NMS (raw FPN)**, so the existing `post_process()` decodes it unchanged. Model: `PATHS.model_hailo8`
(point it at your `<model>.hef`, e.g. `assets/models/<model>.hef`). Input is fed as **NHWC uint8**
(normalization baked into the HEF); outputs are transposed to NCHW and re-grouped to the per-scale
`[box, cls, sum]` layout.

**Multi-stream:** the Hailo-8 is a *single* accelerator (not 3 cores like the RKNPU). Up to 3 camera
streams share one VDevice via HailoRT's round-robin scheduler (time-shared), so there is **no
Auto/Distributed split** for this mode.

### Model conversion (ONNX → HEF)
The `.hef` is produced **off-device** with the Hailo **Dataflow Compiler** (DFC) from the same
Rockchip 9-head ONNX as the other modes — keep the 9 output heads, no on-chip NMS. The pipeline is
**parse** (ONNX → `.har`) → **optimize** (INT8 calibration with ~100–300 representative images) →
**compile** (`performance_param(compiler_optimization_level=max)`) → `.hef`; then place your
`<model>.hef` in `assets/models/` and point `PATHS.model_hailo8` at it.

**Recommended:** run the DFC inside Hailo's **AI Software Suite Docker** on an x86_64 machine (Linux,
or Windows via Docker Desktop, ≥16 GB RAM). That image bundles **`hailo_dataflow_compiler-3.34.0`**,
which version-matches HailoRT 4.24.0 for Hailo-8. See the *Conversion to Hailo HEF* section of the
conversion notebook for the full steps.

> ⚠️ **A free, time-limited Google Colab session cannot finish the conversion.** Parse/optimize run,
> but the final compile times out — YOLO11's C2PSA attention block forces a multi-context split whose
> per-context allocator watchdog (~1h) Colab's 2-core CPU can't beat ("Resolver didn't find possible
> solution / Watchdog expired"), or the session disconnects first (true even for a YOLO11n nano). Use
> Colab only to validate the pipeline; build the real `.hef` in the Docker image on a multi-core x86
> machine. (`3.33.1` from the Developer Zone → **Archive** also works; source page:
> <https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l>.)

> ⚠️ **Match the DFC to the runtime.** For **Hailo-8 + HailoRT 4.24.0** (this device) use **DFC 3.34.0 or
> 3.33.1** — both pin `tensorflow==2.18.0` so they install on Colab's Python 3.12, and both emit a HEF
> that HailoRT 4.24.0 can load. The *main* download page's **DFC 5.3.0 targets Hailo-10H**; its HEF will
> **not load** on HailoRT 4.24.0, so use 5.3.0 only to dry-run the pipeline. The notebook's setup cell
> auto-prefers a 3.x wheel.
>
> _Colab note:_ a 3.x DFC pins `pyparsing==2.4.7`, but it imports TensorFlow whose `httplib2` needs
> `pyparsing>=3.1`; the notebook upgrades it (otherwise `hailo parser` crashes with `AttributeError:
> ... 'set_name'`).

### Runtime install (one-time, on the device)
The Hailo-8 needs the HailoRT stack (PCIe driver + runtime + pyhailort), matching the **HailoRT 4.24.0**
used here (account-gated downloads from the Hailo Developer Zone):

```bash
sudo apt install -y dkms linux-headers-$(uname -r)
sudo dpkg -i hailort-pcie-driver_4.24.0_all.deb     # DKMS builds the kernel module
sudo modprobe hailo_pci                              # creates /dev/hailo0 (no reboot needed)
sudo dpkg -i hailort_4.24.0_arm64.deb                # hailortcli + libhailort
venv/bin/pip install hailort-4.24.0-cp312-cp312-linux_aarch64.whl
hailortcli fw-control identify                       # verify the Hailo-8 is detected
```

### System Monitor
The web System Monitor includes a **Hailo-8** card: utilization (a busy-fraction % — inference time ÷
wall-time, since HailoRT 4.24 exposes no direct utilization counter) and chip temperature. Values
populate while NPU-Hailo8 inference is running.

---

## Troubleshooting

- **No cameras**: Check `ls /dev/video*`
- **Permission denied**: Use `sudo` to run this program
- **Web interface not accessible**: Check firewall settings and use correct IP
- **Video stream not loading**: Ensure processing is started and frames are available
- **GPU-OpenCV-OpenCL falling back to CPU**: Check the OpenCL stack — `cv2.ocl.haveOpenCL()` must be `True` and `/etc/OpenCL/vendors/mali.icd` must point to the libmali blob (see [GPU-OpenCV-OpenCL Inference](#gpu-opencv-opencl-inference-mali-g610) above)

## Model Compatibility
Ensure the RKNN model is compatible with your Rockchip device and matches the input size (640, 640).

## Benchmark Videos
The benchmark clips are **public YouTube surveillance videos**, chosen deliberately: they are publicly
available, show a **clear real-weapon situation in which nobody is harmed** (so the footage is not
graphic or sensitive), and **nobody is identifiable** (no privacy concerns). Each is security-camera
footage of two full-body subjects in a room with a typical, everyday setup where such incidents can
occur — a realistic fit for evaluating weapon detection.

- **benchmark.mp4** — *Caught On Camera: Gunpoint Robbery Inside Nail Salon*
  (https://www.youtube.com/watch?v=apxdeD32kAk)
- **benchmark2.mp4** — *Surveillance video: Boy robs gas station, fires shot*
  (https://www.youtube.com/watch?v=y-QXYbd4Zb0)

## License
This project is licensed under the MIT License.

## Acknowledgments
- libmali (https://github.com/tsukumijima/libmali-rockchip/releases)
- RKNN Toolkit2 (https://github.com/airockchip/rknn-toolkit2)
- RKNN Model Zoo (https://github.com/airockchip/rknn_model_zoo)
- YOLO Vision (https://github.com/ultralytics/ultralytics)
- OpenCV (https://opencv.org/)
- ONNX Runtime (https://github.com/microsoft/onnxruntime)
- MNN (https://github.com/alibaba/MNN)
- HailoRT (https://github.com/hailo-ai/hailort)
- Flask-SocketIO (https://github.com/miguelgrinberg/Flask-SocketIO)
- rknputop (https://github.com/ramonbroox/rknputop)
- myrktop (https://github.com/mhl221135/myrktop)
