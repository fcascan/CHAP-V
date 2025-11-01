#!/usr/bin/env python3
"""
Model Converter: PyTorch to RKNN

This script converts PyTorch YOLO models to RKNN format for Rockchip devices:
1. Converts PyTorch model (.pt) to ONNX
2. Converts ONNX to RKNN using convert.py

by fcascan 2025
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelConverter:
    def __init__(self, model_path, target_platform="rk3588", dtype="i8"):
        """
        Initialize the model converter
        
        Args:
            model_path (str): Path to PyTorch model (.pt)
            target_platform (str): Target platform (e.g. rk3588)
            dtype (str): Data type for quantization (i8, u8, fp)
        """
        self.model_path = Path(model_path)
        self.target_platform = target_platform
        self.dtype = dtype
        self.project_root = Path(__file__).parent
        self.model_dir = self.project_root / "model"
        self.rknn_zoo_dir = self.project_root / "rknn_model_zoo"
        
        # Validate that necessary directories exist
        self._validate_directories()
    
    def _validate_directories(self):
        """Validate that necessary directories and files exist"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if not self.rknn_zoo_dir.exists():
            raise FileNotFoundError(f"rknn_model_zoo directory not found: {self.rknn_zoo_dir}")
        
        if not (self.rknn_zoo_dir / "convert.py").exists():
            raise FileNotFoundError(f"convert.py script not found in {self.rknn_zoo_dir}")
        
        logger.info("[OK] Directory validation completed")
    
    def pytorch_to_onnx(self):
        """
        Convert PyTorch model to ONNX
        
        Returns:
            Path: Path to generated ONNX file
        """
        try:
            import torch
            import torch.onnx
        except ImportError:
            raise ImportError("PyTorch is not installed. Install with: pip install torch torchvision")
        
        logger.info(f"Converting {self.model_path} to ONNX...")
        
        # Try loading with Ultralytics YOLO first
        try:
            return self._convert_ultralytics_to_onnx()
        except Exception as e:
            logger.warning(f"Ultralytics conversion failed: {e}")
            logger.info("Attempting manual conversion...")
            return self._convert_manual_to_onnx()
    
    def _convert_ultralytics_to_onnx(self):
        """Conversion using Ultralytics YOLO directly"""
        try:
            from ultralytics import YOLO
            
            # Load model with Ultralytics
            model = YOLO(str(self.model_path))
            
            # Generate ONNX file name
            onnx_path = self.model_dir / f"{self.model_path.stem}.onnx"
            
            # Export to ONNX using Ultralytics native method with fixed shapes for RKNN
            model.export(format='onnx', imgsz=640, dynamic=False)
            
            # File is saved in the same directory as the original model
            # with the same name but .onnx extension
            expected_onnx = self.model_path.with_suffix('.onnx')
            if expected_onnx.exists():
                # Move to desired location if different
                if expected_onnx != onnx_path:
                    import shutil
                    shutil.move(str(expected_onnx), str(onnx_path))
                    
                logger.info(f"[OK] ONNX model saved at: {onnx_path}")
                return onnx_path
            else:
                raise FileNotFoundError("Expected ONNX file was not generated")
                
        except ImportError:
            raise ImportError("Ultralytics is not installed. Install with: pip install ultralytics")
    
    def _convert_manual_to_onnx(self):
        """Manual conversion for generic PyTorch models"""
        import torch
        import torch.onnx
        
        # Load PyTorch model
        try:
            # For Ultralytics YOLO models, use weights_only=False
            # This is safe if you trust the model source
            try:
                # Try secure loading first
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
            except Exception:
                # If it fails, use full loading for Ultralytics models
                logger.warning("Secure loading failed, using weights_only=False for Ultralytics model")
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Extract model from checkpoint
            if hasattr(checkpoint, 'model'):
                # Object with model attribute
                model = checkpoint.model
            elif isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # Dictionary with 'model' key
                    model = checkpoint['model']
                elif 'ema' in checkpoint:
                    # Checkpoint with EMA
                    model = checkpoint['ema'].model if hasattr(checkpoint['ema'], 'model') else checkpoint['ema']
                else:
                    # The dictionary IS the model
                    logger.warning("Checkpoint is a state dict, trying to load as state_dict")
                    raise ValueError("File appears to be a state_dict, not a complete model")
            else:
                # The loaded object IS the model
                model = checkpoint
            
            # Verify we have a valid model
            if not hasattr(model, 'eval'):
                raise ValueError(f"Loaded object is not a valid PyTorch model: {type(model)}")
                
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            logger.error("Suggestion: Make sure the .pt file contains a complete Ultralytics YOLO model")
            raise
        
        model.eval()
        
        # Define dummy input (common for YOLO11)
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Generate ONNX file name
        onnx_path = self.model_dir / f"{self.model_path.stem}.onnx"
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                (dummy_input,),  # Arguments must be in a tuple
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],  # Use 'images' name like Ultralytics
                output_names=['output'],
                # Remove dynamic_axes for RKNN compatibility - use fixed shapes
            )
            logger.info(f"[OK] ONNX model saved at: {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            raise
    
    def onnx_to_rknn(self, onnx_path, output_rknn_path=None):
        """
        Convert ONNX model to RKNN using convert.py
        
        Args:
            onnx_path (Path): Path to ONNX file
            output_rknn_path (str, optional): Custom output path for RKNN model
            
        Returns:
            Path: Path to generated RKNN file
        """
        logger.info(f"Converting {onnx_path} to RKNN...")
        
        # Change to rknn_model_zoo directory to run convert.py
        original_cwd = os.getcwd()
        
        try:
            os.chdir(self.rknn_zoo_dir)
            
            # Build command with optional output path
            cmd = [
                sys.executable, "convert.py",
                str(onnx_path.absolute()),
                self.target_platform,
                self.dtype
            ]
            
            if output_rknn_path:
                cmd.append(str(output_rknn_path))
                expected_rknn_path = Path(output_rknn_path)
            else:
                expected_rknn_path = onnx_path.parent / "yolo11.rknn"
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Execute convert.py
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("convert.py output:")
            logger.info(result.stdout)
            
            if result.stderr:
                logger.warning(f"Warnings: {result.stderr}")
            
            # Check if RKNN file was generated
            if not expected_rknn_path.exists():
                # Try to find generated .rknn files
                rknn_files = list(onnx_path.parent.glob("*.rknn"))
                if rknn_files:
                    expected_rknn_path = rknn_files[0]
                    logger.info(f"RKNN file found: {expected_rknn_path}")
                else:
                    raise FileNotFoundError("No RKNN file was generated")
            
            logger.info(f"[OK] RKNN model saved at: {expected_rknn_path}")
            return expected_rknn_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing convert.py: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise
        finally:
            os.chdir(original_cwd)
    
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to RKNN for Rockchip devices")
    
    # Model and conversion arguments
    parser.add_argument("--model", default="model/yolo11n.pt", 
                       help="Path to PyTorch model (default: model/yolo11n.pt)")
    parser.add_argument("--platform", default="rk3588", 
                       choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588", "rv1126b", "rv1109", "rv1126", "rk1808"],
                       help="Target platform (default: rk3588)")
    parser.add_argument("--dtype", default="i8", choices=["i8", "u8", "fp"],
                       help="Data type for quantization (default: i8)")
    parser.add_argument("--output-rknn", default=None,
                       help="Custom output path for RKNN model (default: model/yolo11.rknn)")
    
    # Conversion control arguments
    parser.add_argument("--skip-pytorch", action="store_true",
                       help="Skip PyTorch->ONNX conversion (use existing ONNX)")
    parser.add_argument("--skip-rknn", action="store_true",
                       help="Skip ONNX->RKNN conversion (use existing RKNN)")
    parser.add_argument("--onnx-only", action="store_true",
                       help="Only convert to ONNX (skip RKNN conversion)")
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = ModelConverter(args.model, args.platform, args.dtype)
        
        onnx_path = None
        rknn_path = None
        
        # 1. Convert PyTorch to ONNX
        if not args.skip_pytorch:
            logger.info("=== PyTorch to ONNX Conversion ===")
            onnx_path = converter.pytorch_to_onnx()
        else:
            # Look for existing ONNX file
            onnx_files = list(converter.model_dir.glob("*.onnx"))
            if onnx_files:
                onnx_path = onnx_files[0]
                logger.info(f"Using existing ONNX: {onnx_path}")
            else:
                logger.error("No existing ONNX file found. Use --skip-pytorch only if ONNX exists.")
                return 1
        
        # 2. Convert ONNX to RKNN (unless --onnx-only)
        if not args.onnx_only and not args.skip_rknn:
            logger.info("=== ONNX to RKNN Conversion ===")
            rknn_path = converter.onnx_to_rknn(onnx_path, args.output_rknn)
        elif args.skip_rknn:
            logger.info("Skipping RKNN conversion (--skip-rknn)")
        elif args.onnx_only:
            logger.info("ONNX-only conversion completed (--onnx-only)")
        
        # Summary
        logger.info("=== Conversion Summary ===")
        if onnx_path:
            logger.info(f"ONNX model: {onnx_path}")
        if rknn_path:
            logger.info(f"RKNN model: {rknn_path}")
        
        logger.info("[SUCCESS] Model conversion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
