# Python YOLO with RK3588 and Multiple USB Cameras

## Overview

This project demonstrates how to use the **RKNN Toolkit Lite** to perform object detection using a YOLO model on an **RKNN NPU** (Neural Processing Unit). The program supports multiple USB cameras (up to 3) and assigns each camera to a specific NPU core for parallel inference. The detected objects are displayed in real-time, and the processed frames are saved as images in the `images` directory.

## Features

- **Multiple USB Camera Support**: The program supports between 1 and 3 USB cameras connected simultaneously.
- **NPU Core Assignment**: Each camera is assigned to a specific NPU core (`NPU_CORE_0`, `NPU_CORE_1`, `NPU_CORE_2`) for efficient parallel processing.
- **Real-Time Object Detection**: The program performs real-time object detection using a YOLO model optimized for RKNN.
- **Custom Object Classes**: The model is configured to detect a variety of objects, including weapons and other items.
- **Output Images**: The processed frames with bounding boxes and labels are saved in the `images` directory.
- **Average FPS Calculation**: The program calculates and displays the average FPS (frames per second) for each camera.

## Requirements

### Hardware
- A device with an RKNN-compatible NPU (e.g., RK3588 from Orange Pi 5).
- 1 to 3 USB cameras.

### Software
- Python 3.x
- RKNN Toolkit Lite 2.3.2 or later
- OpenCV
- NumPy

## Installation

1. Clone this repository:
```bash
   git clone https://github.com/your-repo/PythonYoloRKNPU.git
   cd PythonYoloRKNPU
```
2. Install the required Python packages:
```bash
   pip install opencv-python-headless numpy
```
3. Ensure the RKNN Toolkit Lite is installed on your device. Follow the official RKNN Toolkit Lite installation guide.  
  https://github.com/rockchip-linux/rknn-toolkit
4. Connect 1 to 3 USB cameras to your device.

## Usage

1. Run the program with superuser permissions:
```bash
   sudo python3 inference_test.py
```
2. The program will:
- Detect connected USB cameras.
- Assign each camera to an NPU core.
- Perform real-time object detection and display the results in separate windows.
- Save the processed frames in the images directory.
3. Press q to exit the program.

## Configuration

### Model Path
The YOLO model file is located in assets/models/yolo_quant_int8.rknn. You can replace this file with your own RKNN model.

### Object Classes
The object classes are defined in the CLASSES variable. You can modify this list to match your custom dataset.

### Thresholds
OBJ_THRESH: Object detection confidence threshold (default: 0.25).
NMS_THRESH: Non-Maximum Suppression (NMS) threshold (default: 0.45).

### Image Size
The input image size is set to (640, 640) by default. You can adjust this in the IMG_SIZE variable.

## Project Structure

PythonYoloRKNPU/
├── assets/
│   └── models/
│       └── yolo_quant_int8.rknn  # YOLO model file
├── images/                       # Directory for output images
├── [inference_test.py](http://_vscodecontentref_/0)             # Main Python script
└── README.md                     # Project documentation

## Example Output

```bash
  Permisos verificados. Iniciando el programa...
  Cámara 0 inicializada.
  Cámara 1 inicializada.
  Se detectaron 2 cámara(s).
  RKNN para cámara 0 inicializado en NPU_CORE_0.
  RKNN para cámara 1 inicializado en NPU_CORE_1.
  Imagen de la cámara 0 guardada en: /path/to/images/inference_output_cam0.jpg
  Imagen de la cámara 1 guardada en: /path/to/images/inference_output_cam1.jpg
  Average FPS para la cámara 0: 25.34
  Average FPS para la cámara 1: 24.87
```

## Saved Images
The processed images will be saved in the images directory with bounding boxes and labels for detected objects.

## Troubleshooting
1. Permission Denied:
- Ensure you run the program with sudo to access the USB cameras and save files.
  
2. Camera Not Detected:
- Verify the connected cameras using:
  ```bash
    ls /dev/video*
  ```
  Ensure the cameras are properly connected and supported by OpenCV.

## Model Compatibility:
Ensure the RKNN model is compatible with your NPU and matches the input size (640, 640).

## License
This project is licensed under the MIT License.

## Acknowledgments
- RKNN Toolkit Lite (https://github.com/rockchip-linux/rknn-toolkit)
- YOLO Model (https://github.com/ultralytics/yolov5)
- OpenCV (https://opencv.org/)
- rknputop (https://github.com/ramonbroox/rknputop)
