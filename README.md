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
- **NPU Mode**: `device = NPU` - Uses RKNN Lite API with RK3588 Neural Processing Unit
- **GPU Mode**: `device = GPU` - Uses ncnn + Vulkan for Mali-G610 GPU (requires Mali Vulkan blob, see below)
- **CPU Mode**: `device = CPU` - Uses ONNX Runtime with CPU backend

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
CPU and NPU paths. It uses the same ONNX as CPU mode (`PATHS.model_onnx`), so the three
processors run identical weights — no separate GPU model or conversion needed. OpenCL is
provided by the Mali blob (`/etc/OpenCL/vendors/mali.icd`); nothing else to install
(OpenCV is already a dependency).

Observed: correct detections matching CPU/NPU, ~2.3–2.5 s/frame (~0.4 FPS) — slow, but a
valid Mali-GPU data point. Confirmed running on the GPU (Mali load ~70–100% during inference).

> **Why not ncnn+Vulkan for GPU?** ncnn numerically mis-computes YOLO11 on this Mali-G610:
> the stride-8/16 class heads collapse to 0.0 on most frames and saturate to garbage on
> others — reproduced identically on ncnn's CPU **and** Vulkan backends and across multiple
> ncnn versions, while the same model runs correctly on the NPU and CPU. So the ncnn path is
> retained only as a (non-functional) reference; the working GPU path is OpenCV-DNN/OpenCL.
> See `src/processing/opencv_executor.py`.

---

### Legacy / experimental: ncnn + Vulkan (does NOT produce correct detections — see note above)

The ncnn path needs two system-level prerequisites beyond the Python packages:

### 1. Mali Vulkan blob

The default Mali blob shipped with Ubuntu for RK3588 is OpenCL-only and does **not**
include Vulkan. You need the GBM variant with Vulkan:

```bash
# The .deb is included in the installation/ directory
sudo dpkg -i installation/libmali-valhall-g610-g24p0-gbm_1.9-1_arm64.deb
```

Or download from [tsukumijima/libmali-rockchip releases](https://github.com/tsukumijima/libmali-rockchip/releases)
(file: `libmali-valhall-g610-g24p0-gbm_*_arm64.deb`).

### 2. Vulkan ICD registration

```bash
sudo mkdir -p /etc/vulkan/icd.d
sudo tee /etc/vulkan/icd.d/mali.json << 'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "/usr/lib/aarch64-linux-gnu/libmali.so",
        "api_version": "1.2.204"
    }
}
EOF
```

### 3. ncnn model conversion (requires a PC with ultralytics)

Use the **Ultralytics-native** ncnn export. It produces the standard YOLO11 head as a
single decoded output blob (`out0`, shape `[1, 4+nc, 8400]`), which is what
`post_process_ncnn()` consumes.

```bash
# On a PC with Python + ultralytics installed:
python3 -c "from ultralytics import YOLO; YOLO('your_model.pt').export(format='ncnn', imgsz=640, half=False)"
# Copy the generated *_ncnn_model/ directory to assets/models/
# Update config.ini: model_ncnn = assets/models/your_model_ncnn_model
```

> Use `half=False` — fp16 weights degrade DFL/box regression on the Mali-G610.
>
> **Do NOT** feed the Rockchip RKNN-optimized ONNX (the 3-tensor-per-scale, 9-output
> head) through `pnnx` for ncnn. That graph is mis-computed by stock ncnn on this
> device (dead stride-8/16 class heads on Mali, and the CPU backend overflows). The
> 9-output head is for the **NPU/RKNN** path only. For ncnn/GPU, always use the
> native export above.

> **setup.sh** handles steps 1 and 2 automatically if the .deb is present in `installation/`.

### ⚠️ Required ncnn version (pinned)

GPU mode is **pinned to `ncnn==1.0.20250503`** (see `requirements.txt`).

`ncnn 1.0.20260526` contains a regression that **zeros the stride-8 / stride-16 class
heads** of YOLO11 on this device: only large (stride-32) objects produce any class
score, detections collapse, and bounding boxes look wrong. Confirmed on both the
Mali-Vulkan and CPU backends with identical model files — i.e. it is a runtime bug, not
a conversion or model problem. Version `1.0.20250503` computes all three FPN scales
correctly. This only affects the Python `ncnn` package; it does **not** touch the
system Mali/Vulkan blob.

```bash
# If GPU detections look wrong / only fire on huge boxes, check the version:
venv/bin/python -c "import ncnn; print(ncnn.__version__)"   # must be 1.0.20250503
venv/bin/pip install --no-cache-dir --no-deps 'ncnn==1.0.20250503'
```

### Verify Vulkan detection

```bash
python3 -c "
import ncnn
ncnn.create_gpu_instance()
count = ncnn.get_gpu_count()
print('GPU count:', count)
if count > 0:
    print('Device:', ncnn.get_gpu_info(0).device_name())
ncnn.destroy_gpu_instance()
"
# Expected: GPU count: 2   Device: Mali-G610
```

### Observed performance (benchmark.mp4, 1 stream)

| Mode | Inference (ms) | FPS | CPU % | GPU % |
|------|---------------|-----|-------|-------|
| NPU  | ~15           | ~35 | ~15   | 0     |
| GPU (ncnn+Vulkan) | ~67–74 | ~13 | ~21 | ~58 |
| GPU (TIMVX/CPU fallback) | ~397 | ~2.5 | ~94 | ~5 |

---

## Troubleshooting

- **No cameras**: Check `ls /dev/video*`
- **Permission denied**: Use `sudo` to run this program
- **Web interface not accessible**: Check firewall settings and use correct IP
- **Video stream not loading**: Ensure processing is started and frames are available
- **GPU mode falling back to CPU**: Install Mali Vulkan blob (see GPU Inference Setup above)

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
