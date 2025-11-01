# RKNN Model Converter

Convert PyTorch YOLO models to RKNN format for Rockchip devices.

## Basic Usage

```bash
# Complete conversion (PyTorch → ONNX → RKNN)
python main.py

# Specify model and platform
python main.py --model model/yolo11_model.pt --platform rk3588 --dtype i8

# Only convert to ONNX
python main.py --onnx-only
```

## Arguments

### Model and Conversion
- `--model`: Path to PyTorch model (default: `model/my_latest_yolo11n.pt`)
- `--platform`: Target platform (default: `rk3588`)
  - Options: `rk3562`, `rk3566`, `rk3568`, `rk3576`, `rk3588`, `rv1126b`, `rv1109`, `rv1126`, `rk1808`
- `--dtype`: Quantization type (default: `i8`)
  - For rk3588/rk3576/rk3568/rk3566/rk3562: `i8`, `fp`
  - For rv1109/rv1126/rk1808: `u8`, `fp`
- `--output-rknn`: Custom output path for RKNN model

### Conversion Control
- `--skip-pytorch`: Skip PyTorch→ONNX conversion (use existing ONNX)
- `--skip-rknn`: Skip ONNX→RKNN conversion (use existing RKNN)
- `--onnx-only`: Only convert to ONNX (skip RKNN conversion)

## Usage Examples

### Different Platforms and Quantization
```bash
# For RK3566 with int8 quantization
python main.py --model model/yolo11n.pt --platform rk3566 --dtype i8

# For RK3576 with floating point
python main.py --model model/yolo11n.pt --platform rk3576 --dtype fp

# Custom output path
python main.py --output-rknn model/custom_model.rknn
```

### Step-by-Step Conversion
```bash
# Only PyTorch → ONNX
python main.py --onnx-only

# Only ONNX → RKNN (using existing ONNX)
python main.py --skip-pytorch
```

## Conversion Process

1. **PyTorch → ONNX**: Loads the `.pt` model and exports to ONNX format with 640x640 input
2. **ONNX → RKNN**: Uses `rknn_model_zoo/convert.py` to apply quantization and generate RKNN model

## Generated Files

- `{model_name}.onnx` - Model in ONNX format
- `yolo11.rknn` - RKNN-optimized model (or custom path)
- `conversion.log` - Process log

## File Structure
```
rknn_converter/
├── main.py                    # Main conversion script
├── model/
│   └── *.pt                   # Input PyTorch models
│   └── *.onnx                 # Generated ONNX models
│   └── *.rknn                 # Generated RKNN models
├── dataset/                   # Quantization images
│   ├── *.jpg
│   └── imgs.txt               # Image list for quantization
└── rknn_model_zoo/            # Conversion scripts
    └── convert.py
```

## Troubleshooting

1. **Error "torch not found"**: Install PyTorch
2. **Error "RKNN files not found"**: Verify `rknn_model_zoo/convert.py` exists
3. **Conversion error**: Check `conversion.log`
4. **Platform error**: Verify `--dtype` is compatible with `--platform`

## Notes

- The script handles Ultralytics YOLO models automatically
- Model input size is fixed at 640x640 pixels
- Files in `rknn_model_zoo/` should not be modified
- Full RKNN toolkit (`rknn-toolkit2`) is required for conversion
