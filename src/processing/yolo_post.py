# -*- coding: utf-8 -*-
"""yolo_post.py
by fcascan 2025
"""
import numpy as np
import cv2
from ..core.config import IMG_SIZE

def yolo_onnx_postprocess(outputs, img_shape, conf_thres=0.25, iou_thres=0.45):
    if isinstance(outputs, (list, tuple)):
        preds = outputs[0]
    else:
        preds = outputs
    while preds.ndim > 2:
        preds = preds[0]
    if preds.shape[1] < 6:
        return None, None, None
    boxes = preds[:, :4]
    scores = preds[:, 4:5] * preds[:, 5:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    mask = confidences > conf_thres
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    h0, w0 = img_shape[:2]
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= w0 / IMG_SIZE[0]
        boxes[:, [1, 3]] *= h0 / IMG_SIZE[1]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thres, iou_thres) if len(boxes) > 0 else []
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], class_ids[indices], confidences[indices]
    else:
        return None, None, None
