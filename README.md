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
- **GPU Mode**: `device = GPU` - Uses OpenCV DNN with OpenCL backend for Mali G610 GPU  
- **CPU Mode**: `device = CPU` - Uses OpenCV DNN with CPU backend

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

## Troubleshooting

- **No cameras**: Check `ls /dev/video*`
- **Permission denied**: Use `sudo` to run this program
- **Web interface not accessible**: Check firewall settings and use correct IP
- **Video stream not loading**: Ensure processing is started and frames are available

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
