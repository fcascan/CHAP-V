# -*- coding: utf-8 -*-
"""rknn_executor.py
Modified version from rknn_model_zoo -> py_utils
https://github.com/airockchip/rknn_model_zoo
RKNNLite executor for NPU inference on Rockchip devices
by fcascan 2025
"""
import os
import sys

# Add the project root to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.config import DEBUG_MODE
from rknnlite.api import RKNNLite


def is_debug_enabled():
    """Check if debug mode is enabled dynamically."""
    return DEBUG_MODE or os.environ.get('RKNN_LOG_LEVEL', '0') == '1'


def setup_rknn_environment():
    """Set RKNN environment variables when debug mode is enabled."""
    if is_debug_enabled():
        os.environ['RKNN_LOG_LEVEL'] = '1'
        os.environ['RKNN_VERBOSE'] = '1'
    else:
        os.environ.pop('RKNN_LOG_LEVEL', None)
        os.environ.pop('RKNN_VERBOSE', None)


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        # Setup environment variables dynamically
        setup_rknn_environment()
        
        if is_debug_enabled():
            print(f"[RKNN_EXECUTOR] Initializing model: {model_path}")
            print(f"[RKNN_EXECUTOR] Target: {target}, Device ID: {device_id}")
        
        # Initialize RKNNLite
        rknn = RKNNLite(
            verbose=is_debug_enabled(),
            verbose_file="./rknn_verbose.log",  # str: Path to save verbose log file, None to disable
        )
        
        # Load RKNN model
        if is_debug_enabled():
            print(f"[RKNN_EXECUTOR] Loading RKNN model from: {model_path}")
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print(f"[RKNN_EXECUTOR] ERROR: Failed to load model, return code: {ret}")
            raise Exception(f"Failed to load RKNN model: {model_path}")
        
        if is_debug_enabled():
            print(f"[RKNN_EXECUTOR] Model loaded successfully")
            print('--> Init runtime environment')
        
        # Initialize runtime environment
        try:
            if device_id is not None:
                # Use specific core if device_id is provided
                core_mask = getattr(RKNNLite, f'NPU_CORE_{device_id}', RKNNLite.NPU_CORE_0)
                ret = rknn.init_runtime(
                    # target=target,      # str: 'RK3562'/'RK3566'/'RK3568'/'RK3588' or None for NPU inside
                    # device_id=None,     # str: adb device id when multiple devices connected
                    # async_mode=False,   # bool: enable/disable async mode
                    core_mask=core_mask,  # int: NPU_CORE_AUTO(0)/NPU_CORE_0(1)/NPU_CORE_1(2)/NPU_CORE_2(4)/NPU_CORE_0_1(3)/NPU_CORE_0_1_2(7)/NPU_CORE_ALL(0xffff)
                )
            else:
                # Default initialization
                ret = rknn.init_runtime(
                    # target=target,      # str: 'RK3562'/'RK3566'/'RK3568'/'RK3588' or None for NPU inside
                    # device_id=None,     # str: adb device id when multiple devices connected
                    # async_mode=False,   # bool: enable/disable async mode
                    # core_mask=NPU_CORE_AUTO,  # int: NPU_CORE_AUTO(0)/NPU_CORE_0(1)/NPU_CORE_1(2)/NPU_CORE_2(4)/NPU_CORE_0_1(3)/NPU_CORE_0_1_2(7)/NPU_CORE_ALL(0xffff)
                )
                
            if ret != 0:
                print(f'[RKNN_EXECUTOR] Init runtime environment failed with code: {ret}')
                raise Exception(f"Failed to initialize RKNNLite runtime: {ret}")
            
            if is_debug_enabled():
                print('[RKNN_EXECUTOR] Runtime environment initialized successfully')
            
        except Exception as e:
            print(f'[RKNN_EXECUTOR] Init runtime environment failed with exception: {e}')
            raise e
        
        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs):
        if self.rknn is None:
            print("[RKNN_EXECUTOR] ERROR: RKNNLite instance has been released")
            return []

        # Ensure inputs is a list
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # Debug logging for inference only in debug mode
        if is_debug_enabled():
            print(f"[RKNN_EXECUTOR] Running inference with {len(inputs)} input(s)")
            for i, inp in enumerate(inputs):
                print(f"[RKNN_EXECUTOR] Input {i} shape: {inp.shape}, dtype: {inp.dtype}")
        
        try:
            result = self.rknn.inference(
                inputs=inputs,
                # data_type=None,           # str: 'int8'/'uint8'/'int16'/'float16'/'float32', default 'uint8'
                # data_format=None,         # str: 'nhwc'/'nchw', default None
                # inputs_pass_through=None, # list: pass_through flag (0/1) for each input
                # get_frame_id=False,       # bool: get frame id for async mode
            )
            
            if is_debug_enabled():
                print(f"[RKNN_EXECUTOR] Inference completed, got {len(result)} output(s)")
                for i, out in enumerate(result):
                    print(f"[RKNN_EXECUTOR] Output {i} shape: {out.shape}, dtype: {out.dtype}")
        
            return result
        except Exception as e:
            print(f"[RKNN_EXECUTOR] ERROR during RKNNLite inference: {e}")
            return []

    def release(self):
        if self.rknn is not None:
            if is_debug_enabled():
                print("[RKNN_EXECUTOR] Releasing RKNNLite resources")
            self.rknn.release()
            self.rknn = None