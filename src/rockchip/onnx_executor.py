# =============================================================================
# Origin: https://github.com/airockchip/rknn_model_zoo
# Path:   py_utils/onnx_executor.py
# License: Apache 2.0
# Copied and included in this repository to avoid runtime git-clone dependency.
# Modified from original:
#   - ONNX_model_container_cpp stub class commented out (empty, unused)
#   - reset_onnx_shape() commented out (onnxsim utility, not used in inference)
#   - ONNX_model_container.__init__(): optional intra_op_num_threads + cpu_affinity
#     params for CPU-50% mode — caps the onnxruntime intra-op thread pool, sets
#     inter_op=1 and disables thread spinning; default None keeps the original
#     all-cores behavior (plain CPU mode unchanged).
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
    def __init__(self, model_path, intra_op_num_threads=None, cpu_affinity=None) -> None:
        opts = rt.SessionOptions()
        opts.log_severity_level = 3  # 3 = error only
        # CPU-50% mode: cap the CPU thread pool so CPU inference does not pin all 8 cores
        # (the "don't saturate" dial). intra_op_num_threads=None keeps onnxruntime's default
        # (all cores) so plain CPU mode is byte-for-byte unchanged.
        if intra_op_num_threads is not None:
            opts.intra_op_num_threads = int(intra_op_num_threads)
            opts.inter_op_num_threads = 1
            # Stop idle intra-op threads from busy-spinning (burning a core) between the
            # ~7-8 fps frames — they would otherwise read as ~100% on the capped cores.
            try:
                opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            except Exception:
                pass
        self.sess = rt.InferenceSession(
            model_path, sess_options=opts,
            providers=['CPUExecutionProvider'],
        )
        self.model_path = model_path
        # The actual thread-to-core pinning is applied by the worker THREAD that calls run()
        # (os.sched_setaffinity is per-thread); stored here only for reference/inspection.
        self.cpu_affinity = cpu_affinity

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
