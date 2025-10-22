# -*- coding: utf-8 -*-
"""rknn_post_processing.py
by fcascan 2025
"""
import numpy as np
from ..core.config import *
from rknnlite.api import RKNNLite

# Adjust for tasks (taken from yolov8 default cfg)
OBJ_THRESH = 0.25
NMS_THRESH = 0.45 

# Post processing functions taken from https://github.com/airockchip/rknn_model_zoo
def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    x = np.array(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)

    max_values = np.max(y, axis=2, keepdims=True)
    exp_values = np.exp(y - max_values)
    y = exp_values / np.sum(exp_values, axis=2, keepdims=True)

    acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y * acc_matrix, axis=2)

    return y

def box_process(position):
    # print(f"[DEBUG] box_process: position shape: {position.shape}, dtype: {position.dtype}")
    
    # Check if position has at least 4 dimensions
    if len(position.shape) < 4:
        print(f"[ERROR] box_process: position shape {position.shape} has less than 4 dimensions!")
        print(f"[ERROR] Expected shape format: (batch, channels, height, width)")
        raise ValueError(f"Position tensor shape {position.shape} is not compatible. Expected at least 4 dimensions.")
    
    try:
        grid_h, grid_w = position.shape[2:4]
        # print(f"[DEBUG] box_process: grid_h: {grid_h}, grid_w: {grid_w}")
    except ValueError as e:
        print(f"[ERROR] box_process: Failed to unpack grid dimensions from shape[2:4]: {position.shape[2:4]}")
        print(f"[ERROR] Full shape: {position.shape}")
        raise e
    
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process_yolov11(input_data):
    """
    Post-processing function for YOLOv11 models with flattened output format.
    YOLOv11 might use a different structure without explicit objectness.
    Expected input: (1, features, anchors) where features = bbox(4) + classes(n)
    """
    # print(f"[DEBUG] post_process_yolov11: input_data type: {type(input_data)}")
    # print(f"[DEBUG] post_process_yolov11: len(input_data): {len(input_data)}")
    
    if len(input_data) != 1:
        print(f"[WARNING] Expected 1 output tensor, got {len(input_data)}. Using the first one.")
    
    output = input_data[0]  # Shape: (1, features, anchors)
    # print(f"[DEBUG] post_process_yolov11: output shape: {output.shape}")
    
    batch_size, features, num_anchors = output.shape
    
    # Let's try both interpretations:
    # 1. YOLOv11 with objectness: bbox(4) + objectness(1) + classes(n)
    # 2. YOLOv11 without objectness: bbox(4) + classes(n)
    
    # print(f"[DEBUG] post_process_yolov11: Trying different interpretations...")
    
    # Method 1: Try without objectness (bbox + classes only)
    if features == 8:  # 4 bbox + 4 classes
        # print(f"[DEBUG] Trying interpretation: 4 bbox + 4 classes (no objectness)")
        num_classes = 4
        
        # Extract components from the output tensor
        output = output.transpose(0, 2, 1)  # Now shape: (1, 8400, 8)
        output = output.squeeze(0)  # Remove batch dimension: (8400, 8)
        
        # Split the tensor components
        bbox_coords = output[:, :4]  # (8400, 4) - x_center, y_center, width, height
        class_probs = output[:, 4:8]  # (8400, 4) - class probabilities
        
        # Debug raw values before activation
        # print(f"[DEBUG] Raw bbox_coords range: [{bbox_coords.min():.6f}, {bbox_coords.max():.6f}]")
        # print(f"[DEBUG] Raw class_probs range: [{class_probs.min():.6f}, {class_probs.max():.6f}]")
        
        # Check if all values are the same
        class_probs_unique = np.unique(class_probs.flatten())
        print(f"[DEBUG] Unique class_probs values count: {len(class_probs_unique)}")
        
        if len(class_probs_unique) < 10:
            print(f"[WARNING] Very few unique values detected in no-objectness mode.")
            if np.allclose(class_probs, class_probs[0, 0]):
                print(f"[ERROR] All class prob values are identical ({class_probs[0, 0]:.6f}). Model inference failed.")
                return None, None, None
        
        # Apply sigmoid activation to class probabilities
        class_probs = 1 / (1 + np.exp(-class_probs))  # sigmoid activation
        
        # print(f"[DEBUG] post_process_yolov11: bbox_coords shape: {bbox_coords.shape}")
        # print(f"[DEBUG] post_process_yolov11: class_probs shape: {class_probs.shape}")
        # print(f"[DEBUG] post_process_yolov11: class_probs range: [{class_probs.min():.6f}, {class_probs.max():.6f}]")
        
        # Get class scores (no objectness, use class confidence directly)
        class_max_scores = np.max(class_probs, axis=1)  # (8400,)
        classes = np.argmax(class_probs, axis=1)  # (8400,)
        confidence_scores = class_max_scores  # Use class confidence directly
        
    elif features == 9:  # 4 bbox + 1 objectness + 4 classes
        # print(f"[DEBUG] Trying interpretation: 4 bbox + 1 objectness + 4 classes")
        num_classes = 4
        
        # Extract components from the output tensor
        output = output.transpose(0, 2, 1)  # Now shape: (1, 8400, 9)
        output = output.squeeze(0)  # Remove batch dimension: (8400, 9)
        
        # Split the tensor components  
        bbox_coords = output[:, :4]  # (8400, 4) - x_center, y_center, width, height
        objectness = output[:, 4:5]  # (8400, 1) - objectness score
        class_probs = output[:, 5:9]  # (8400, 4) - class probabilities
        
        # Debug raw values before activation
        print(f"[DEBUG] Raw objectness range: [{objectness.min():.6f}, {objectness.max():.6f}]")
        print(f"[DEBUG] Raw class_probs range: [{class_probs.min():.6f}, {class_probs.max():.6f}]")
        
        # Check if all values are the same (which would indicate a problem)
        objectness_unique = np.unique(objectness)
        class_probs_unique = np.unique(class_probs.flatten())
        print(f"[DEBUG] Unique objectness values count: {len(objectness_unique)}")
        print(f"[DEBUG] Unique class_probs values count: {len(class_probs_unique)}")
        
        if len(objectness_unique) < 10 and len(class_probs_unique) < 10:
            print(f"[WARNING] Very few unique values detected. This suggests model inference issues.")
            print(f"[WARNING] Objectness unique values: {objectness_unique[:5]}...")  
            print(f"[WARNING] Class_probs unique values: {class_probs_unique[:5]}...")
            
            # If all values are effectively the same (likely 0), return no detections
            if np.allclose(objectness, objectness[0, 0]) and np.allclose(class_probs, class_probs[0, 0]):
                print(f"[ERROR] All tensor values are identical ({objectness[0, 0]:.6f}). Model inference failed.")
                return None, None, None
        
        # Apply sigmoid activation
        objectness = 1 / (1 + np.exp(-objectness))  # sigmoid activation
        class_probs = 1 / (1 + np.exp(-class_probs))  # sigmoid activation
        
        print(f"[DEBUG] post_process_yolov11: bbox_coords shape: {bbox_coords.shape}")
        print(f"[DEBUG] post_process_yolov11: objectness shape: {objectness.shape}")
        print(f"[DEBUG] post_process_yolov11: class_probs shape: {class_probs.shape}")
        print(f"[DEBUG] post_process_yolov11: objectness range: [{objectness.min():.6f}, {objectness.max():.6f}]")
        print(f"[DEBUG] post_process_yolov11: class_probs range: [{class_probs.min():.6f}, {class_probs.max():.6f}]")
        
        # Combined confidence score (objectness * class_confidence)
        class_max_scores = np.max(class_probs, axis=1)  # (8400,)
        classes = np.argmax(class_probs, axis=1)  # (8400,)
        confidence_scores = objectness.squeeze() * class_max_scores  # (8400,)
    else:
        print(f"[ERROR] Unexpected feature count: {features}. Expected 8 or 9.")
        return None, None, None
    
    print(f"[DEBUG] post_process_yolov11: bbox_coords range: [{bbox_coords.min():.6f}, {bbox_coords.max():.6f}]")
    print(f"[DEBUG] post_process_yolov11: confidence_scores range: [{confidence_scores.min():.6f}, {confidence_scores.max():.6f}]")
    
    # Count how many are above different thresholds
    counts = {}
    for thresh in [0.01, 0.05, 0.1, 0.25, 0.5]:
        count = np.sum(confidence_scores >= thresh)
        counts[thresh] = count
    print(f"[DEBUG] post_process_yolov11: detections above thresholds: {counts}")
    
    # Show top 5 confidence scores
    top_indices = np.argsort(confidence_scores)[-5:][::-1]
    print(f"[DEBUG] post_process_yolov11: top 5 confidence scores:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. idx={idx}, confidence={confidence_scores[idx]:.6f}, class={classes[idx]}")
    
    # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    cx, cy, w, h = bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2], bbox_coords[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)  # (8400, 4)
    
    # Filter by confidence threshold
    valid_mask = confidence_scores >= OBJ_THRESH
    
    if not np.any(valid_mask):
        print(f"[DEBUG] post_process_yolov11: No detections above threshold {OBJ_THRESH}")
        # Try with a lower threshold
        low_thresh = 0.01
        valid_mask = confidence_scores >= low_thresh
        if not np.any(valid_mask):
            print(f"[DEBUG] post_process_yolov11: No detections above threshold {low_thresh} either")
            return None, None, None
        else:
            print(f"[INFO] Using lower threshold {low_thresh} instead of {OBJ_THRESH}")
    
    filtered_boxes = boxes[valid_mask]
    filtered_classes = classes[valid_mask]
    filtered_scores = confidence_scores[valid_mask]
    
    print(f"[DEBUG] post_process_yolov11: {np.sum(valid_mask)} detections above threshold")
    
    # Apply Non-Maximum Suppression (NMS) per class
    final_boxes, final_classes, final_scores = [], [], []
    
    for class_id in np.unique(filtered_classes):
        class_mask = filtered_classes == class_id
        class_boxes = filtered_boxes[class_mask]
        class_scores = filtered_scores[class_mask]
        
        # Apply NMS
        keep_indices = nms_boxes(class_boxes, class_scores)
        
        if len(keep_indices) > 0:
            final_boxes.append(class_boxes[keep_indices])
            final_classes.append(np.full(len(keep_indices), class_id))
            final_scores.append(class_scores[keep_indices])
    
    if not final_boxes:
        print("[DEBUG] post_process_yolov11: No detections after NMS")
        return None, None, None
    
    # Concatenate results
    final_boxes = np.concatenate(final_boxes, axis=0)
    final_classes = np.concatenate(final_classes, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    
    print(f"[DEBUG] post_process_yolov11: Final detections: {len(final_boxes)}")
    
    return final_boxes, final_classes, final_scores


def post_process(input_data):
    """
    Main post-processing function that auto-detects the format and routes to appropriate handler.
    """
    # Debug: Print information about input_data structure
    # print(f"[DEBUG] post_process: input_data type: {type(input_data)}")
    # print(f"[DEBUG] post_process: len(input_data): {len(input_data)}")
    # for i, data in enumerate(input_data):
        # print(f"[DEBUG] post_process: input_data[{i}] shape: {data.shape}, dtype: {data.dtype}")
    
    # Auto-detect format based on tensor shapes
    if len(input_data) == 1 and len(input_data[0].shape) == 3:
        # YOLOv11 format: single tensor with shape (batch, features, anchors)
        # print("[INFO] Detected YOLOv11 format, using YOLOv11 post-processing")
        return post_process_yolov11(input_data)
    else:
        # YOLOv8 format: multiple tensors with 4D shapes
        # print("[INFO] Detected YOLOv8 format, using original post-processing")
        return post_process_yolov8(input_data)


def post_process_yolov8(input_data):
    """
    Original post-processing function for YOLOv8 format (renamed for clarity).
    """
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    # print(f"[DEBUG] post_process_yolov8: defualt_branch: {defualt_branch}, pair_per_branch: {pair_per_branch}")
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores