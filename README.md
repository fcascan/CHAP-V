# Python YOLO RKNN/NPU

Real-time object detection using YOLO v11 models on RK3588 (with Orange Pi 5 Max).
A project for comparison between CPU, GPU & NPU inference with **Web Interface** support.

## Features

- **Web Interface**: Modern web UI with real-time video streaming and console output
- **Multi-camera support**: Up to 3 USB cameras with NPU core assignment
- **Auto-installation**: Intelligent dependency detection and installation
- **NPU acceleration**: RKNN toolkit with CPU fallback
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

Edit [`config.ini`](config.ini):
- `benchmark_mode`: Video file vs camera mode
- `device`: NPU, GPU or CPU inference
- `enabled`: Enable/disable web interface by default
- Model paths and detection parameters

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
- **NPU Mode**: `device = NPU` - Uses RKNN Lite API with the RK3588 Neural Processing Unit
- **GPU Mode**: `device = GPU` - Runs the ONNX model on the Mali-G610 GPU via OpenCV-DNN + OpenCL (see [GPU Inference](#gpu-inference-mali-g610) below)
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

## GPU Inference (Mali-G610)

**GPU mode runs the ONNX model on the Mali-G610 via OpenCV-DNN with the OpenCL target**
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
| NPU  | RKNN (rknnlite) | ~33 | ~29 | NPU |
| CPU  | ONNX Runtime | ~127 | ~7.6 | CPU 100% |
| GPU  | OpenCV-DNN / OpenCL (Mali) | ~2400 | ~0.4 | GPU ~70–100% |

All three produce correct, matching detections; the GPU is slowest despite genuinely using the
Mali (see "Why this mode is slow" above).

> The first GPU run auto-tunes the OpenCL convolution kernels (slower first frame); the tuned
> configs are cached under `~/.cache/PythonYoloRKNPU/ocl4dnn`, so later runs start faster.

---

## Troubleshooting

- **No cameras**: Check `ls /dev/video*`
- **Permission denied**: Use `sudo` to run this program
- **Web interface not accessible**: Check firewall settings and use correct IP
- **Video stream not loading**: Ensure processing is started and frames are available
- **GPU mode falling back to CPU**: Check the OpenCL stack — `cv2.ocl.haveOpenCL()` must be `True` and `/etc/OpenCL/vendors/mali.icd` must point to the libmali blob (see [GPU Inference](#gpu-inference-mali-g610) above)

## Model Compatibility
Ensure the RKNN model is compatible with your Rockchip device and matches the input size (640, 640).

## License
This project is licensed under the MIT License.

## Acknowledgments
- RKNN Toolkit (https://github.com/rockchip-linux/rknn-toolkit)
- YOLO Vision (https://github.com/ultralytics/ultralytics)
- OpenCV (https://opencv.org/)
- rknputop (https://github.com/ramonbroox/rknputop)
- myrktop (https://github.com/mhl221135/myrktop)
