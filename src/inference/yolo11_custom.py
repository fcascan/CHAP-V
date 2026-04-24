# -*- coding: utf-8 -*-
"""yolo11_custom.py
Modified version from rknn_model_zoo -> examples/yolo11/python
https://github.com/airockchip/rknn_model_zoo
YOLO11 object detection with custom configuration
by fcascan 2025
"""
import os
import cv2
import sys
import numpy as np

# Import project configuration instead of hardcoded values
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import COCO utilities
try:
    from .coco_utils import COCO_test_helper
    print("[YOLO11_CUSTOM] COCO utilities imported successfully")
except ImportError:
    try:
        from coco_utils import COCO_test_helper
        print("[YOLO11_CUSTOM] COCO utilities imported (absolute import)")
    except ImportError as e:
        print(f"[YOLO11_CUSTOM] Could not import COCO utilities: {e}")


# ============================================================================
# CORE YOLO11 FUNCTIONS (Previously from yolo11_og.py)
# ============================================================================

def filter_boxes(boxes, box_confidences, box_class_probs):
    """
    Filter boxes with object threshold using project configuration.
    
    Args:
        boxes: Detection boxes
        box_confidences: Box confidence scores
        box_class_probs: Class probabilities
        
    Returns:
        Tuple of filtered (boxes, classes, scores)
    """
    # Get current thresholds from config
    from src.core.config import OBJ_THRESHOLD
    
    print(f"[DEBUG][FILTER_BOXES] Input shapes:")
    print(f"[DEBUG][FILTER_BOXES] - boxes: {boxes.shape}")
    print(f"[DEBUG][FILTER_BOXES] - box_confidences: {box_confidences.shape}")
    print(f"[DEBUG][FILTER_BOXES] - box_class_probs: {box_class_probs.shape}")
    print(f"[DEBUG][FILTER_BOXES] - OBJ_THRESHOLD: {OBJ_THRESHOLD}")
    
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape
    print(f"[DEBUG][FILTER_BOXES] After reshape - candidates: {candidate}, classes: {class_num}")

    print(f"[DEBUG][FILTER_BOXES] Box confidences range: [{np.min(box_confidences):.6f}, {np.max(box_confidences):.6f}]")
    print(f"[DEBUG][FILTER_BOXES] Box class probs range: [{np.min(box_class_probs):.6f}, {np.max(box_class_probs):.6f}]")

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    
    print(f"[DEBUG][FILTER_BOXES] Class max scores range: [{np.min(class_max_score):.6f}, {np.max(class_max_score):.6f}]")
    
    combined_scores = class_max_score * box_confidences
    print(f"[DEBUG][FILTER_BOXES] Combined scores range: [{np.min(combined_scores):.6f}, {np.max(combined_scores):.6f}]")
    print(f"[DEBUG][FILTER_BOXES] Boxes above threshold: {np.sum(combined_scores >= OBJ_THRESHOLD)}")

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESHOLD)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    
    print(f"[DEBUG][FILTER_BOXES] Final filtered results: {len(boxes)} boxes")

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """
    Apply non-maximum suppression using project NMS threshold.
    
    Args:
        boxes: Bounding boxes in format [x1, y1, x2, y2]
        scores: Confidence scores for each box
        
    Returns:
        List of indices to keep after NMS
    """
    from src.core.config import NMS_THRESHOLD
    
    if len(boxes) == 0:
        return []
    
    # Validate input boxes
    if not np.isfinite(boxes).all():
        print(f"[WARNING] NMS: Found non-finite values in boxes, filtering them out")
        valid_mask = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
        if not np.any(valid_mask):
            return []
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if len(boxes) == 0:
            return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas, adding small epsilon to avoid division by zero
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Filter out boxes with zero or negative area
    valid_area_mask = areas > 0
    if not np.any(valid_area_mask):
        return []
    
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # Compute IoU with protection against division by zero
        union = areas[i] + areas[order[1:]] - inter
        # Add small epsilon to avoid division by zero
        union = np.maximum(union, 1e-6)
        ovr = inter / union
        
        # Keep boxes with IoU less than threshold
        inds = np.where(ovr <= NMS_THRESHOLD)[0]
        order = order[inds + 1]
    
    return keep


def dfl(position):
    """
    Distribution Focal Loss (DFL) processing for bounding box regression.
    
    Args:
        position: Raw position predictions from model
        
    Returns:
        Processed position values
    """
    x = np.array(position, dtype=np.float32)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    
    # Apply softmax along the mc dimension
    y_exp = np.exp(y)
    y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
    
    # Create range tensor
    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y_softmax * acc_metrix, axis=2)
    
    return y


def box_process(position):
    """
    Process position predictions to generate bounding boxes.
    
    Args:
        position: Position tensor from model output
        
    Returns:
        Processed bounding boxes
    """
    from src.core.config import IMG_SIZE
    
    grid_h, grid_w = position.shape[-2:]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_w, IMG_SIZE[0] // grid_h]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def img_check(path):
    """
    Check if file path is a valid image format.
    
    Args:
        path: File path to check
        
    Returns:
        bool: True if valid image format, False otherwise
    """
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def original_setup_model(args):
    """
    Setup model with proper executor based on file extension.
    
    Args:
        args: Arguments containing model_path, target, device_id
        
    Returns:
        Tuple of (model, platform)
    """
    model_path = args.model_path
    platform = None
    
    if model_path.endswith('.rknn'):
        # Use RKNN executor for .rknn models
        from .rknn_executor import RKNN_model_container
        model = RKNN_model_container(model_path, target=args.target, device_id=args.device_id)
        platform = "RKNN"
    elif model_path.endswith('.onnx'):
        # Use ONNX executor for .onnx models  
        from .onnx_executor import ONNX_model_container_py
        model = ONNX_model_container_py(model_path)
        platform = "ONNX"
    elif model_path.endswith('.pt'):
        # Use PyTorch executor for .pt models
        try:
            from .pytorch_executor import Torch_model_container
            model = Torch_model_container(model_path)
            platform = "PyTorch"
        except ImportError as e:
            print(f"[WARNING] PyTorch not available: {e}")
            print(f"[WARNING] Cannot load .pt model: {model_path}")
            raise ValueError(f"PyTorch is required for .pt models but is not available: {e}")
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    
    return model, platform


def load_current_config():
    """
    Load current configuration values from config module.
    This function ensures we always get the most up-to-date configuration.
    """
    from src.core.config import reload_config
    
    # Force reload of configuration
    config_data = reload_config()
    
    # Import updated values
    from src.core.config import CLASSES, IMG_SIZE, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD
    
    return CLASSES, IMG_SIZE, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD

# Load initial configuration and display info
CLASSES, IMG_SIZE, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD = load_current_config()

# Use configuration values from config.ini
OBJ_THRESH = OBJ_THRESHOLD
NMS_THRESH = NMS_THRESHOLD

# Display current configuration
print(f"[YOLO11_CUSTOM] Using {len(CLASSES)} custom classes: {CLASSES[:3]}...")
print(f"[YOLO11_CUSTOM] Using image size: {IMG_SIZE}")
print(f"[YOLO11_CUSTOM] Using Rockchip target: {ROCKCHIP_TARGET}")
print(f"[YOLO11_CUSTOM] Using thresholds: OBJ={OBJ_THRESH}, NMS={NMS_THRESH}")


def post_process_modern_yolo11(input_data):
    """
    Modern YOLO11 post-processing for single concatenated output.
    Expected input shape: (1, 84, 8400) where 84 = 4_bbox + 80_classes
    """    
    # For modern YOLO11, we expect a single output with shape (1, 84, 8400)
    if len(input_data) == 1:
        output = input_data[0]  # Shape: (1, 84, 8400)
        
        # Transpose to (1, 8400, 84) for easier processing
        output = output.transpose(0, 2, 1)  # Now (1, 8400, 84)
        
        # Remove batch dimension
        output = output.squeeze(0)  # Now (8400, 84)
        if DEBUG_MODE:
            print(f"[DEBUG][MODERN_POST_PROCESS] Reshaped output: {output.shape}")
        
        # Split into bounding boxes and class probabilities
        # Dynamically determine number of classes from config and output shape
        from src.core.config import CLASSES
        expected_classes = len(CLASSES)
        total_channels = output.shape[1]
        
        if DEBUG_MODE:
            print(f"[DEBUG][MODERN_POST_PROCESS] Total channels: {total_channels}, Expected classes: {expected_classes}")
        
        # Check if model uses DFL (Distribution Focal Loss) format
        if total_channels == 4 + expected_classes:  # Standard format: 4 bbox + N classes
            boxes = output[:, :4]  # First 4 columns are bbox coordinates
            class_probs = output[:, 4:]  # Remaining columns are class probabilities
            if DEBUG_MODE:
                print(f"[DEBUG][MODERN_POST_PROCESS] Using standard format: 4 bbox + {expected_classes} classes")
        elif total_channels > 4 + expected_classes:  # Possible DFL format with expanded box predictions
            if DEBUG_MODE:
                print(f"[DEBUG][MODERN_POST_PROCESS] Potential DFL format detected with {total_channels} channels")
            # For DFL format, typically first N channels are for bbox and remaining are for classes
            box_channels = total_channels - expected_classes
            boxes = output[:, :box_channels]
            class_probs = output[:, box_channels:]
            if DEBUG_MODE:
                print(f"[DEBUG][MODERN_POST_PROCESS] DFL format: {box_channels} box channels + {expected_classes} classes")
        else:
            raise ValueError(f"Unexpected output format: {total_channels} channels for {expected_classes} classes")
        
        # Process box coordinates - handle both standard and DFL formats        
        if boxes.shape[1] == 4:  # Standard format: cx, cy, w, h
            # Extract coordinates - these are already in absolute pixel values
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
        elif boxes.shape[1] > 4:  # DFL format - need to process using DFL function            
            # Check if we need to use DFL processing or if it's already processed
            raw_coords = boxes[:, :4] if boxes.shape[1] >= 4 else boxes
            cx, cy, w, h = raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2], raw_coords[:, 3]
            
            # Use the first 4 columns as coordinates for now
            boxes = raw_coords
            
        else:
            raise ValueError(f"Unexpected box format with {boxes.shape[1]} channels")
            
        # Common processing for both formats
        from src.core.config import IMG_SIZE
        input_width, input_height = IMG_SIZE[1], IMG_SIZE[0]
        
        # Extract final coordinates (assuming cx, cy, w, h format at this point)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Clamp coordinates to reasonable bounds to avoid infinity issues
        # Allow some margin beyond image boundaries for partially visible objects
        margin_factor = 1.5
        max_coord = max(input_width, input_height) * margin_factor
        
        cx = np.clip(cx, 0, max_coord)
        cy = np.clip(cy, 0, max_coord) 
        w = np.clip(w, 1, max_coord)  # Minimum width of 1 pixel
        h = np.clip(h, 1, max_coord)  # Minimum height of 1 pixel
        
        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        x1 = cx - w / 2
        y1 = cy - h / 2  
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # For modern YOLO11, class probabilities might be raw logits
        # Apply sigmoid to get probabilities with numerical stability        
        # Clamp extreme values to prevent overflow in sigmoid
        class_probs = np.clip(class_probs, -50, 50)
        
        # Apply sigmoid activation: 1 / (1 + exp(-x))
        class_probs = 1 / (1 + np.exp(-class_probs))
        
        # Check for problematic sigmoid results (all values around 0.5 indicates bad logits)
        if DEBUG_MODE:
            unique_probs = np.unique(np.round(class_probs, 3))
            if len(unique_probs) < 10 and np.all(np.abs(unique_probs - 0.5) < 0.1):
                print(f"[WARNING][MODERN_POST_PROCESS] Suspicious sigmoid results - most values near 0.5")
        
        # Get the best class and its probability for each detection
        class_ids = np.argmax(class_probs, axis=1)
        class_scores = np.max(class_probs, axis=1)
        
        # Apply confidence threshold
        from src.core.config import OBJ_THRESHOLD
        
        mask = class_scores >= OBJ_THRESHOLD
        if not np.any(mask):
            return None, None, None
            
        filtered_boxes = boxes[mask]
        filtered_classes = class_ids[mask]
        filtered_scores = class_scores[mask]
        
        # Filter out invalid boxes (infinite or NaN values) before NMS
        valid_mask = (
            np.isfinite(filtered_boxes).all(axis=1) &  # All coordinates are finite
            (filtered_boxes[:, 2] > filtered_boxes[:, 0]) &  # x2 > x1
            (filtered_boxes[:, 3] > filtered_boxes[:, 1]) &  # y2 > y1
            (filtered_boxes[:, 2] - filtered_boxes[:, 0] > 0) &  # width > 0
            (filtered_boxes[:, 3] - filtered_boxes[:, 1] > 0)    # height > 0
        )
        
        if not np.any(valid_mask):
            return None, None, None
            
        valid_boxes = filtered_boxes[valid_mask]
        valid_classes = filtered_classes[valid_mask]
        valid_scores = filtered_scores[valid_mask]
        
        # Apply NMS
        final_boxes, final_classes, final_scores = [], [], []
        for class_id in np.unique(valid_classes):
            class_mask = valid_classes == class_id
            class_boxes = valid_boxes[class_mask]
            class_scores_for_nms = valid_scores[class_mask]
            
            # Apply NMS for this class
            keep_indices = nms_boxes(class_boxes, class_scores_for_nms)
            
            if len(keep_indices) > 0:
                final_boxes.append(class_boxes[keep_indices])
                final_classes.append(np.full(len(keep_indices), class_id))
                final_scores.append(class_scores_for_nms[keep_indices])
        
        if not final_boxes:
            return None, None, None
            
        final_boxes = np.concatenate(final_boxes)
        final_classes = np.concatenate(final_classes)
        final_scores = np.concatenate(final_scores)
        
        # Always show detection results with class breakdown - important information
        from src.core.config import CLASSES
        class_counts = {}
        for cls in final_classes:
            class_name = CLASSES[cls] if cls < len(CLASSES) else f"Unknown({cls})"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            print(f"[DETECTIONS] Found {len(final_boxes)} objects: {dict(class_counts)}")
        else:
            print(f"[DETECTIONS] Found {len(final_boxes)} objects")
        return final_boxes, final_classes, final_scores
    
    else:
        print(f"[DEBUG][MODERN_POST_PROCESS] Unexpected number of outputs: {len(input_data)}")
        print(f"[DEBUG][MODERN_POST_PROCESS] Falling back to legacy post-processing")
        return post_process_legacy(input_data)


def post_process_legacy(input_data):
    """Legacy post-processing for multi-branch YOLO11 outputs.""" 
    print(f"[DEBUG][LEGACY_POST_PROCESS] Using legacy multi-branch processing")
    return post_process_original(input_data)


def post_process_original(input_data):
    """
    Original post-process YOLO11 model outputs.
    Uses the same logic as original but with project configuration.
    """
    print(f"[DEBUG][POST_PROCESS] Input data type: {type(input_data)}")
    print(f"[DEBUG][POST_PROCESS] Number of outputs: {len(input_data) if hasattr(input_data, '__len__') else 'N/A'}")
    
    # Debug each output tensor
    for i, output in enumerate(input_data):
        print(f"[DEBUG][POST_PROCESS] Output {i} shape: {output.shape}, dtype: {output.dtype}")
        if output.size > 0:
            print(f"[DEBUG][POST_PROCESS] Output {i} value range: [{np.min(output):.6f}, {np.max(output):.6f}]")
            print(f"[DEBUG][POST_PROCESS] Output {i} mean: {np.mean(output):.6f}, std: {np.std(output):.6f}")
    
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    print(f"[DEBUG][POST_PROCESS] Expected branches: {defualt_branch}, pair per branch: {pair_per_branch}")
    
    for i in range(defualt_branch):
        box_idx = pair_per_branch*i
        class_idx = pair_per_branch*i+1
        print(f"[DEBUG][POST_PROCESS] Processing branch {i}: box_idx={box_idx}, class_idx={class_idx}")
        
        if box_idx < len(input_data) and class_idx < len(input_data):
            boxes.append(box_process(input_data[box_idx]))
            classes_conf.append(input_data[class_idx])
            scores.append(np.ones_like(input_data[class_idx][:,:1,:,:], dtype=np.float32))
            print(f"[DEBUG][POST_PROCESS] Branch {i} processed successfully")
        else:
            print(f"[DEBUG][POST_PROCESS] Branch {i} indices out of range: max_idx={len(input_data)-1}")

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
    
    print(f"[DEBUG][POST_PROCESS] After concatenation:")
    print(f"[DEBUG][POST_PROCESS] Boxes shape: {boxes.shape}")
    print(f"[DEBUG][POST_PROCESS] Classes conf shape: {classes_conf.shape}")
    print(f"[DEBUG][POST_PROCESS] Scores shape: {scores.shape}")
    print(f"[DEBUG][POST_PROCESS] Max class confidence: {np.max(classes_conf)}")
    print(f"[DEBUG][POST_PROCESS] Using thresholds - OBJ: {OBJ_THRESH}, NMS: {NMS_THRESH}")

    # Filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    print(f"[DEBUG][POST_PROCESS] After filtering: {len(boxes) if boxes is not None else 0} boxes")

    # NMS
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
        print(f"[DEBUG][POST_PROCESS] No detections after NMS")
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    
    print(f"[DEBUG][POST_PROCESS] Final results: {len(boxes)} detections")
    for i, (box, cls, score) in enumerate(zip(boxes[:5], classes[:5], scores[:5])):
        print(f"[DEBUG][POST_PROCESS] Detection {i}: class={cls}, score={score:.3f}, box={box}")

    return boxes, classes, scores


def post_process(input_data):
    """
    Main post-processing function that tries modern format first, then falls back to legacy.
    """
    print(f"[DEBUG][POST_PROCESS] Determining post-processing method...")
    
    # Check if this looks like modern YOLO11 format (single output)
    if len(input_data) == 1 and len(input_data[0].shape) == 3:
        output_shape = input_data[0].shape
        print(f"[DEBUG][POST_PROCESS] Single output detected with shape: {output_shape}")
        
        # Modern YOLO11 typically has shape (1, 84, 8400) or similar
        if output_shape[0] == 1 and output_shape[1] > 4:
            print(f"[DEBUG][POST_PROCESS] Using modern YOLO11 post-processing")
            return post_process_modern_yolo11(input_data)
    
    # Fall back to legacy multi-branch processing
    print(f"[DEBUG][POST_PROCESS] Using legacy multi-branch post-processing")
    return post_process_original(input_data)


def draw(image, boxes, scores, classes):
    """
    Draw detection results on image using project's custom classes.
    """
    # Get current classes to ensure up-to-date values
    from src.core.config import CLASSES as current_classes
    
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        
        # Use custom classes instead of hardcoded COCO classes
        if cl < len(current_classes):
            class_name = current_classes[cl]
            print("%s @ (%d %d %d %d) %.3f" % (class_name, top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(class_name, score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            print("Unknown class %d @ (%d %d %d %d) %.3f" % (cl, top, left, right, bottom, score))


def setup_model(args):
    """
    Setup model with project configuration.
    Uses ROCKCHIP_TARGET from config instead of hardcoded values.
    """
    # Get current configuration to ensure up-to-date values
    global CLASSES, IMG_SIZE, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD, OBJ_THRESH, NMS_THRESH
    CLASSES, IMG_SIZE, ROCKCHIP_TARGET, OBJ_THRESHOLD, NMS_THRESHOLD = load_current_config()
    OBJ_THRESH = OBJ_THRESHOLD
    NMS_THRESH = NMS_THRESHOLD
    
    model_path = args.model_path
    
    # Override target with config value if using NPU
    if model_path.endswith('.rknn'):
        args.target = ROCKCHIP_TARGET
        print(f"[YOLO11_CUSTOM] Using Rockchip target from config: {ROCKCHIP_TARGET}")
    
    # Use original setup_model function with updated args
    return original_setup_model(args)


class MockArgs:
    """Mock arguments class for setup_model function."""
    def __init__(self, model_path, target=None, device_id=None):
        self.model_path = model_path
        self.target = target or ROCKCHIP_TARGET
        self.device_id = device_id


def create_model_from_path(model_path, device_id=None, target=None):
    """
    Create model from path using project configuration.
    
    Args:
        model_path: Path to model file
        device_id: Device ID for multi-device setups
        target: Target platform (e.g., 'rk3588', 'rk3566'). If None, uses ROCKCHIP_TARGET from config.
        
    Returns:
        Tuple of (model, platform)
    """
    target_platform = target if target is not None else ROCKCHIP_TARGET
    mock_args = MockArgs(model_path, target_platform, device_id)
    return setup_model(mock_args)


def validate_classes_compatibility(model_output_shape):
    """
    Validate that the number of classes in config matches model output.
    
    Args:
        model_output_shape: Shape of model's class output tensor
        
    Returns:
        bool: True if compatible, False otherwise
    """
    # For YOLO models, the class output typically has shape [batch, num_classes, height, width]
    if len(model_output_shape) >= 2:
        model_num_classes = model_output_shape[1]
        config_num_classes = len(CLASSES)
        
        if model_num_classes != config_num_classes:
            print(f"[WARNING] Model expects {model_num_classes} classes but config has {config_num_classes} classes")
            print(f"[WARNING] Model output shape: {model_output_shape}")
            print(f"[WARNING] Config classes: {CLASSES}")
            return False
    
    return True


def get_class_name(class_id):
    """
    Get class name from class ID using project configuration.
    
    Args:
        class_id: Integer class ID
        
    Returns:
        str: Class name or "Unknown" if ID is out of range
    """
    # Get current classes to ensure up-to-date values
    from src.core.config import CLASSES as current_classes
    
    if 0 <= class_id < len(current_classes):
        return current_classes[class_id]
    else:
        return f"Unknown({class_id})"


def get_detection_summary(boxes, classes, scores, score_threshold=0.5):
    """
    Get a summary of detections using custom class names.
    
    Args:
        boxes: Detection bounding boxes
        classes: Detection class IDs
        scores: Detection scores
        score_threshold: Minimum score threshold for summary
        
    Returns:
        dict: Summary with class counts and high-confidence detections
    """
    if boxes is None or classes is None or scores is None:
        return {"total_detections": 0, "class_counts": {}, "high_confidence": []}
    
    # Count detections by class
    class_counts = {}
    high_confidence = []
    
    for box, cls, score in zip(boxes, classes, scores):
        class_name = get_class_name(cls)
        
        # Count all detections
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
        
        # Track high-confidence detections
        if score >= score_threshold:
            high_confidence.append({
                "class": class_name,
                "score": float(score),
                "bbox": [int(x) for x in box]
            })
    
    return {
        "total_detections": len(boxes),
        "class_counts": class_counts,
        "high_confidence": high_confidence
    }


# Export the functions that the main system needs
__all__ = [
    'post_process', 'draw', 'setup_model', 'create_model_from_path',
    'validate_classes_compatibility', 'get_class_name', 'get_detection_summary',
    'OBJ_THRESH', 'NMS_THRESH', 'CLASSES', 'IMG_SIZE', 'ROCKCHIP_TARGET'
]