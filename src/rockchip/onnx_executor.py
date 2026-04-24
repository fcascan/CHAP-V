# =============================================================================
# Origin: https://github.com/airockchip/rknn_model_zoo
# Path:   py_utils/onnx_executor.py
# License: Apache 2.0
# Copied and included in this repository to avoid runtime git-clone dependency.
# Modified from original:
#   - ONNX_model_container_cpp stub class commented out (empty, unused)
#   - reset_onnx_shape() commented out (onnxsim utility, not used in inference)
# =============================================================================

import numpy as np
import onnxruntime as rt


type_map = {
    'tensor(int32)':   np.int32,
    'tensor(int64)':   np.int64,
    'tensor(float32)': np.float32,
    'tensor(float64)': np.float64,
    'tensor(float)':   np.float32,
    'tensor(bool)':    bool,
}


def _ignore_dim_with_zero(shape, shape_target):
    shape = [v for v in shape if v != 1]
    shape_target = [v for v in shape_target if v != 1]
    return shape == shape_target


class ONNX_model_container():
    def __init__(self, model_path) -> None:
        opts = rt.SessionOptions()
        opts.log_severity_level = 3  # 3 = error only
        self.sess = rt.InferenceSession(
            model_path, sess_options=opts,
            providers=['CPUExecutionProvider'],
        )
        self.model_path = model_path

    def run(self, input_datas):
        if self.sess is None:
            print('ERROR: session has been released')
            return []

        if len(input_datas) < len(self.sess.get_inputs()):
            assert False, 'input count does not match model {} inputs'.format(self.model_path)
        elif len(input_datas) > len(self.sess.get_inputs()):
            print('WARNING: more inputs provided than model input nodes')

        input_dict = {}
        for i, inp in enumerate(self.sess.get_inputs()):
            if inp.type in type_map and type_map[inp.type] != input_datas[i].dtype:
                input_datas[i] = input_datas[i].astype(type_map[inp.type])
            if inp.shape != list(input_datas[i].shape):
                if _ignore_dim_with_zero(input_datas[i].shape, inp.shape):
                    input_datas[i] = input_datas[i].reshape(inp.shape)
                else:
                    assert False, 'input shape {} does not match data shape {}'.format(inp.shape, input_datas[i].shape)
            input_dict[inp.name] = input_datas[i]

        output_names = [o.name for o in self.sess.get_outputs()]
        return self.sess.run(output_names, input_dict)

    def release(self):
        del self.sess
        self.sess = None


# --- ONNX_model_container_cpp: empty stub, not used in this project ---
# class ONNX_model_container_cpp:
#     def __init__(self, model_path) -> None:
#         pass
#     def run(self, input_datas):
#         pass

# --- reset_onnx_shape: onnxsim shape-fixing utility, not used in inference ---
# def reset_onnx_shape(onnx_model_path, output_path, input_shapes):
#     ...
