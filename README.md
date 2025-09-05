# Python YOLO RKNN/NPU

Real-time object detection using YOLO v11 models on RK3588 (with Orange Pi 5 Max).
A project for comparison between CPU, GPU & NPU inference.

## Features

- **Multi-camera support**: Up to 3 USB cameras with NPU core assignment
- **Auto-installation**: Intelligent dependency detection and installation
- **NPU acceleration**: RKNN toolkit with CPU fallback
- **Real-time inference**: Live object detection with statistics

## Requirements
- Python 3.x
- RKNN Toolkit Lite 2.3.2
- OpenCV
- NumPy

## Quick Start

```bash
git clone https://github.com/fcascan/PythonYoloRKNPU.git
cd PythonYoloRKNPU
sudo python3 main.py
```

## Requirements

- RK3588 device (Orange Pi 5/5+/5 Max)
- Python 3.7+ (aarch64)
- USB cameras (optional when using benchmark mode)

## Configuration

Edit [`config.ini`](config.ini):
- `benchmark_mode`: Video file vs camera mode
- `device`: NPU or CPU inference
- Model paths and detection parameters

## Usage

**Camera mode**: `benchmark_mode = false`
**Video mode**: `benchmark_mode = true`

Press 'q' to exit. Results saved to `images/` directory.

## Troubleshooting

- **No cameras**: Check `ls /dev/video*`
- **Permission denied**: Use `sudo`
- **RKNN fails**: Program auto-switches to CPU mode
## Model Compatibility:
Ensure the RKNN model is compatible with your NPU and matches the input size (640, 640).

## License
This project is licensed under the MIT License.

## Acknowledgments
- RKNN Toolkit Lite (https://github.com/rockchip-linux/rknn-toolkit)
- YOLO Model (https://github.com/ultralytics/yolov5)
- OpenCV (https://opencv.org/)
- rknputop (https://github.com/ramonbroox/rknputop)
- myrktop (https://github.com/mhl221135/myrktop)
