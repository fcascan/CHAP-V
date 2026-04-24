# =============================================================================
# Origin: https://github.com/airockchip/rknn_model_zoo
# Path:   py_utils/pytorch_executor.py
# License: Apache 2.0
# Copied and included in this repository to avoid runtime git-clone dependency.
# Modified from original:
#   - multi_list_unfold() commented out (unused helper with a latent bug)
# =============================================================================

import torch
torch.backends.quantized.engine = 'qnnpack'


# --- multi_list_unfold: unused helper (also contains a bug: unfold called without 'target') ---
# def multi_list_unfold(tl):
#     def unfold(_inl, target):
#         if not isinstance(_inl, list) and not isinstance(_inl, tuple):
#             target.append(_inl)
#         else:
#             unfold(_inl)   # bug: missing 'target' arg


def _flatten_list(in_list):
    flatten = lambda x: [sub for item in x for sub in flatten(item)] if type(x) is list else [x]
    return flatten(in_list)


class Torch_model_container():
    def __init__(self, model_path, qnnpack=False) -> None:
        if qnnpack:
            torch.backends.quantized.engine = 'qnnpack'
        self.pt_model = torch.jit.load(model_path)
        self.pt_model.eval()

    def run(self, input_datas):
        if self.pt_model is None:
            print('ERROR: pt_model has been released')
            return []

        assert isinstance(input_datas, list), 'input_datas must be a list of np.ndarray'

        tensors = [torch.tensor(d) for d in input_datas]
        tensors = [t.float() if t.dtype == torch.float64 else t for t in tensors]

        result = self.pt_model(*tensors)

        if isinstance(result, tuple):
            result = list(result)
        if not isinstance(result, list):
            result = [result]

        result = _flatten_list(result)
        result = [torch.dequantize(r).cpu().detach().numpy() for r in result]
        return result

    def release(self):
        del self.pt_model
        self.pt_model = None
