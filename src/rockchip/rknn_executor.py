"""RKNNLite executor for on-device Rockchip inference."""

from rknnlite.api import RKNNLite


class RKNN_model_container:
    def __init__(self, model_path, target=None, device_id=None):
        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN model: {} (ret={})".format(model_path, ret))

        print("--> Init runtime environment")

        if device_id is not None:
            core_name = "NPU_CORE_{}".format(device_id)
            core_mask = getattr(RKNNLite, core_name, None)
            if core_mask is not None:
                ret = self.rknn.init_runtime(core_mask=core_mask)
            else:
                ret = self.rknn.init_runtime()
        else:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError("Init runtime environment failed, ret={}".format(ret))

        print("done")

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return self.rknn.inference(inputs=inputs)

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
