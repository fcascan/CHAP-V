# =============================================================================
# Origin: https://github.com/airockchip/rknn_model_zoo
# Path:   py_utils/coco_utils.py
# License: Apache 2.0
# Copied and included in this repository to avoid runtime git-clone dependency.
# Modified from original:
#   - coco_eval_with_json() commented out (requires pycocotools, only needed for
#     COCO mAP benchmarking; uncomment if you need mAP evaluation)
#   - letter_box() and direct_resize(): replaced list.append() with direct assignment
#     so letter_box_info_list stays at size 1 — prevents unbounded memory growth in
#     long-running inference sessions (only [-1] is ever accessed)
#   - get_real_box(): cached letter_box_info_list[-1] into a local variable to avoid
#     8 repeated list lookups per call
# =============================================================================

from copy import copy
import os
import cv2
import numpy as np
import json


class Letter_Box_Info():
    def __init__(self, shape, new_shape, w_ratio, h_ratio, dw, dh, pad_color) -> None:
        self.origin_shape = shape
        self.new_shape = new_shape
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.dw = dw
        self.dh = dh
        self.pad_color = pad_color


# --- Uncomment to enable COCO mAP evaluation (requires: pip install pycocotools) ---
# def coco_eval_with_json(anno_json, pred_json):
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval
#     anno = COCO(anno_json)
#     pred = anno.loadRes(pred_json)
#     eval = COCOeval(anno, pred, 'bbox')
#     eval.evaluate()
#     eval.accumulate()
#     eval.summarize()
#     map, map50 = eval.stats[:2]
#     print('map  --> ', map)
#     print('map50--> ', map50)
#     print('map75--> ', eval.stats[2])
#     print('map85--> ', eval.stats[-2])
#     print('map95--> ', eval.stats[-1])
# ------------------------------------------------------------------------------------


class COCO_test_helper():
    def __init__(self, enable_letter_box=False) -> None:
        self.record_list = []
        self.enable_ltter_box = enable_letter_box
        if self.enable_ltter_box is True:
            self.letter_box_info_list = []
        else:
            self.letter_box_info_list = None

    def letter_box(self, im, new_shape, pad_color=(0, 0, 0), info_need=False):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

        if self.enable_ltter_box is True:
            self.letter_box_info_list = [Letter_Box_Info(shape, new_shape, ratio, ratio, dw, dh, pad_color)]
        if info_need is True:
            return im, ratio, (dw, dh)
        else:
            return im

    def direct_resize(self, im, new_shape, info_need=False):
        shape = im.shape[:2]
        h_ratio = new_shape[0] / shape[0]
        w_ratio = new_shape[1] / shape[1]
        if self.enable_ltter_box is True:
            self.letter_box_info_list = [Letter_Box_Info(shape, new_shape, w_ratio, h_ratio, 0, 0, (0, 0, 0))]
        im = cv2.resize(im, (new_shape[1], new_shape[0]))
        return im

    def get_real_box(self, box, in_format='xyxy'):
        bbox = copy(box)
        if self.enable_ltter_box is True:
            if in_format == 'xyxy':
                info = self.letter_box_info_list[-1]
                bbox[:, 0] -= info.dw
                bbox[:, 0] /= info.w_ratio
                bbox[:, 0] = np.clip(bbox[:, 0], 0, info.origin_shape[1])

                bbox[:, 1] -= info.dh
                bbox[:, 1] /= info.h_ratio
                bbox[:, 1] = np.clip(bbox[:, 1], 0, info.origin_shape[0])

                bbox[:, 2] -= info.dw
                bbox[:, 2] /= info.w_ratio
                bbox[:, 2] = np.clip(bbox[:, 2], 0, info.origin_shape[1])

                bbox[:, 3] -= info.dh
                bbox[:, 3] /= info.h_ratio
                bbox[:, 3] = np.clip(bbox[:, 3], 0, info.origin_shape[0])
        return bbox

    def add_single_record(self, image_id, category_id, bbox, score, in_format='xyxy'):
        if self.enable_ltter_box is True:
            if in_format == 'xyxy':
                bbox[0] -= self.letter_box_info_list[-1].dw
                bbox[0] /= self.letter_box_info_list[-1].w_ratio
                bbox[1] -= self.letter_box_info_list[-1].dh
                bbox[1] /= self.letter_box_info_list[-1].h_ratio
                bbox[2] -= self.letter_box_info_list[-1].dw
                bbox[2] /= self.letter_box_info_list[-1].w_ratio
                bbox[3] -= self.letter_box_info_list[-1].dh
                bbox[3] /= self.letter_box_info_list[-1].h_ratio

        if in_format == 'xyxy':
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
        else:
            assert False, "only xyxy format is supported"

        self.record_list.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [round(x, 3) for x in bbox],
            'score': round(score, 5),
        })

    def export_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.record_list, f)
