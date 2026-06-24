# =============================================================================
# Origin: https://github.com/airockchip/rknn_model_zoo
# Path:   examples/yolo11/python/yolo11.py
# License: Apache 2.0
#
# Modified from original:
#   - OBJ_THRESH, NMS_THRESH, IMG_SIZE converted to CLI arguments
#   - CLASSES loaded from --classes_file (text file) instead of hardcoded COCO list
#   - coco_id_list replaced with sequential IDs for non-COCO datasets
#   - --img_folder and --anno_json defaults updated for standalone use
#   - sys.path manipulation removed (executors are co-located in the same directory)
#   - --video_source added for video file and camera inference
#   - press 'q' in the display window to stop video/camera inference
#   - draw(): optional frame_label parameter added for per-frame log tagging
#   - preprocess_frame(): 'ncnn' added to the CHW float32 branch
#   - setup_model(): directory detection branch added for NCNN model loading
#     (uses src.processing.ncnn_executor.NCNN_model_container; import is relative
#     to the project root, not the rknn_zoo examples directory)
#
# Project additions (not part of the original rknn_zoo library):
#   - Imports: from src.core import config as app_config
#              from src.utils.frame_overlay import calculate_recent_average_ms,
#                  calculate_recent_fps, draw_processing_overlay
#   - post_process_ncnn(): full NCNN/Vulkan GPU post-processing for the
#     Ultralytics pnnx single-blob output format ([nc+4, 8400]).
#     Includes degenerate-box filter: discards boxes with zero/negative width or
#     height that arise from the pnnx anchor_x encoding bug (stored step ~64px
#     vs correct 32px → cx > frame width → x1 and x2 both clip to right edge).
#   - _ncnn_stride_info(): maps a flat anchor index (0-8399) to (stride, gx, gy)
#   - Module-level NCNN diagnostic state: _ncnn_det_diag_done, _ncnn_frame_count,
#     _ncnn_det_count, _NCNN_SCORE_LOG_INTERVAL
#   - debug_detection_summary(): prints class counts and top scores when
#     DEBUG_DETECTIONS is enabled; not present in original rknn_zoo script
# =============================================================================

import os
import cv2
import argparse
import configparser
import time

from src.core import config as app_config
from coco_utils import COCO_test_helper
import numpy as np
from src.utils.frame_overlay import calculate_recent_average_ms, calculate_recent_fps, draw_processing_overlay


# Default values; overridden from CLI args in __main__ before any function is called.
OBJ_THRESH: float = 0.25
NMS_THRESH: float = 0.45
IMG_SIZE: tuple = (640, 640)
CLASSES: tuple = ()
coco_id_list: list = []
DEBUG_DETECTIONS: bool = False


def _load_labels_from_file(labels_file):
    """
    Load class labels from a text file.

    Parameters:
    - labels_file: Absolute path to a labels file with one class per line.

    Returns:
    - A tuple of non-empty class names.

    Business rule:
    - Empty lines are ignored.
    """
    with open(labels_file, 'r', encoding='utf-8') as f:
        return tuple(line.strip() for line in f if line.strip())


def resolve_runtime_classes(cli_classes_file):
    """
    Resolve runtime classes using the project fallback chain.

    Parameters:
    - cli_classes_file: Optional labels file path provided by --classes_file.

    Returns:
    - (classes_tuple, source_description)

    Business rules:
    - Priority is config PATHS.model_labels, then project yolo11n.txt, then config CLASSES.default_labels.
    - If --classes_file is provided and exists, it has the highest priority.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config.ini')
    parser = configparser.ConfigParser()
    parser.read(config_path)

    if cli_classes_file:
        cli_path = cli_classes_file
        if not os.path.isabs(cli_path):
            cli_path = os.path.join(project_root, cli_path)
        if os.path.isfile(cli_path):
            labels = _load_labels_from_file(cli_path)
            if labels:
                return labels, f'--classes_file ({cli_path})'
        else:
            print(f'[WARN] classes_file not found: {cli_classes_file}')

    model_labels_cfg = parser.get('PATHS', 'model_labels', fallback='').strip()
    if model_labels_cfg:
        model_labels_path = model_labels_cfg
        if not os.path.isabs(model_labels_path):
            model_labels_path = os.path.join(project_root, model_labels_path)
        if os.path.isfile(model_labels_path):
            labels = _load_labels_from_file(model_labels_path)
            if labels:
                return labels, f'config PATHS.model_labels ({model_labels_path})'

    yolo11n_fallback = os.path.join(project_root, 'yolo11n.txt')
    if os.path.isfile(yolo11n_fallback):
        labels = _load_labels_from_file(yolo11n_fallback)
        if labels:
            return labels, f'project fallback ({yolo11n_fallback})'

    default_labels_cfg = parser.get('CLASSES', 'default_labels', fallback='person')
    default_labels = tuple(label.strip() for label in default_labels_cfg.split(',') if label.strip())
    if default_labels:
        return default_labels, 'config CLASSES.default_labels'

    return ('person',), 'hard fallback (person)'


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes."""
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
    # Distribution Focal Loss (DFL)
    x = np.asarray(position, dtype=np.float32)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = y / np.sum(y, axis=2, keepdims=True)
    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy  = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

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


_ncnn_det_diag_done = False   # print anchor details once on first detection frame
_ncnn_frame_count    = 0      # total frames processed through post_process_ncnn
_ncnn_det_count      = 0      # total detection-frames so far
_NCNN_SCORE_LOG_INTERVAL = 50  # log score distribution every N frames


def _ncnn_stride_info(anch_idx):
    """Return (stride, grid_x, grid_y) for a flat anchor index 0-8399."""
    if anch_idx < 6400:
        stride = 8; base = 0; cols = 80
    elif anch_idx < 8000:
        stride = 16; base = 6400; cols = 40
    else:
        stride = 32; base = 8000; cols = 20
    local = anch_idx - base
    return stride, local % cols, local // cols


def post_process_ncnn(input_data):
    """Post-process the single-tensor NCNN output from a YOLO11 model.

    The NCNN export (Ultralytics) concatenates decoded boxes and class scores
    into one blob named 'out0' with shape [nc+4, 8400]:
      - rows 0-3   : cx, cy, w, h  in pixel coordinates (stride-scaled)
      - rows 4..nc : class confidences (sigmoided, one row per class)

    Returns (boxes, classes, scores) or (None, None, None).
    """
    global _ncnn_det_diag_done, _ncnn_frame_count, _ncnn_det_count

    if not input_data or input_data[0] is None:
        return None, None, None

    output = np.asarray(input_data[0], dtype=np.float32)
    _ncnn_frame_count += 1

    # Squeeze any extra leading dims: [1, nc+4, 8400] → [nc+4, 8400]
    while output.ndim > 2:
        output = output[0]

    # Auto-orient: NCNN may return [8400, nc+4] on some builds; detect by
    # checking which axis is 8400 (all anchors).
    if output.ndim == 2 and output.shape[0] == 8400 and output.shape[1] >= 5:
        output = output.T  # → [nc+4, 8400]

    if output.shape[0] < 5:
        print(f"[NCNN WARN] Unexpected output shape {output.shape} — skipping")
        return None, None, None

    nc = output.shape[0] - 4  # number of classes

    # Boxes: cx, cy, w, h → x1, y1, x2, y2
    cx = output[0]
    cy = output[1]
    w  = output[2]
    h  = output[3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_all = np.stack([x1, y1, x2, y2], axis=1)  # [8400, 4]

    class_scores_all = output[4:].T  # [8400, nc]
    class_ids_all    = np.argmax(class_scores_all, axis=1)
    scores_all       = np.max(class_scores_all, axis=1)

    # ── Periodic score-distribution diagnostic (every N frames) ─────────────
    # Logs the per-FPN-scale (stride 8/16/32) max class score and how many
    # above-threshold anchors are in-frame vs off-frame.  This makes two failure
    # modes visible at a glance:
    #   * a whole stride band reading ~0  → dead class head (e.g. the ncnn
    #     1.0.20260526 regression — see requirements.txt / README pin)
    #   * many off-frame above-threshold anchors → stride-32 anchor-decode garbage
    if _ncnn_frame_count % _NCNN_SCORE_LOG_INTERVAL == 0:
        _wl, _hl = IMG_SIZE[1], IMG_SIZE[0]
        _inframe = (cx >= 0) & (cx <= _wl) & (cy >= 0) & (cy <= _hl)
        _above = scores_all >= OBJ_THRESH
        _s8, _s16, _s32 = scores_all[:6400], scores_all[6400:8000], scores_all[8000:]
        top5_idx = np.argsort(scores_all)[::-1][:5]
        lines = []
        for i in top5_idx:
            stride, gx, gy = _ncnn_stride_info(i)
            cls_name = CLASSES[int(class_ids_all[i])] if 0 <= int(class_ids_all[i]) < len(CLASSES) else str(class_ids_all[i])
            _io = "in" if _inframe[i] else "OFF"
            lines.append(f"    anch={i} stride={stride} grid=({gx},{gy}) "
                         f"cx={cx[i]:.1f} cy={cy[i]:.1f} w={w[i]:.1f} h={h[i]:.1f} "
                         f"cls={cls_name} score={scores_all[i]:.4f} [{_io}-frame]")
        print(f"[NCNN SCORE LOG F#{_ncnn_frame_count}] "
              f"max_score={scores_all.max():.4f}  "
              f"per-stride max: s8={_s8.max():.3f} s16={_s16.max():.3f} s32={_s32.max():.3f}  "
              f"above_thresh={int(_above.sum())} "
              f"(in-frame={int((_above & _inframe).sum())}, off-frame={int((_above & ~_inframe).sum())})  "
              f"top-5 anchors:")
        for l in lines:
            print(l)
    # ─────────────────────────────────────────────────────────────────────────

    # Filter by score threshold.
    mask      = scores_all >= OBJ_THRESH
    boxes     = boxes_all[mask]
    class_ids = class_ids_all[mask]
    scores    = scores_all[mask]

    if len(boxes) == 0:
        return None, None, None

    # Discard off-frame and degenerate boxes.
    # The NCNN/pnnx export of YOLO11 emits spurious stride-32 anchors whose decoded
    # CENTER lands far outside the letterbox (observed cx≈1057 for a 640px input,
    # w≈650).  Their raw width is positive, so a plain area test passes them, but
    # after get_real_box() they collapse onto the frame edge — e.g. (638,0,638,360) —
    # and draw as a garbage line.  Fix: drop any box whose center is outside the
    # letterbox, and any box that is degenerate once clipped to the letterbox bounds.
    W_lb, H_lb = IMG_SIZE[1], IMG_SIZE[0]
    cx_b = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy_b = (boxes[:, 1] + boxes[:, 3]) / 2.0
    cl_x1 = np.clip(boxes[:, 0], 0, W_lb); cl_y1 = np.clip(boxes[:, 1], 0, H_lb)
    cl_x2 = np.clip(boxes[:, 2], 0, W_lb); cl_y2 = np.clip(boxes[:, 3], 0, H_lb)
    center_in = (cx_b >= 0) & (cx_b <= W_lb) & (cy_b >= 0) & (cy_b <= H_lb)
    area_ok   = (cl_x2 - cl_x1 > 1) & (cl_y2 - cl_y1 > 1)
    valid       = center_in & area_ok
    n_drop_off  = int((~center_in).sum())
    n_drop_area = int((center_in & ~area_ok).sum())
    if not valid.any():
        # Throttle this message: with a weak model every frame can be all-garbage,
        # which would flood the console. Log only on the periodic diagnostic tick.
        if (n_drop_off or n_drop_area) and _ncnn_frame_count % _NCNN_SCORE_LOG_INTERVAL == 0:
            print(f"[NCNN WARN] F#{_ncnn_frame_count}: all {len(boxes)} above-threshold "
                  f"box(es) discarded: {n_drop_off} off-frame "
                  f"(stride-32 anchor-decode garbage), {n_drop_area} degenerate.")
        return None, None, None
    if not valid.all():
        print(f"[NCNN WARN] Discarding {int((~valid).sum())} box(es): "
              f"{n_drop_off} off-frame (center outside letterbox), "
              f"{n_drop_area} degenerate (zero area after clip).")
        boxes     = boxes[valid]
        class_ids = class_ids[valid]
        scores    = scores[valid]

    _ncnn_det_count += 1

    # ── First-detection diagnostic: show top-5 firing anchors in detail ─────
    if not _ncnn_det_diag_done:
        _ncnn_det_diag_done = True
        above_idx = np.where(mask)[0]  # original anchor indices that passed threshold
        sorted_order = np.argsort(scores_all[above_idx])[::-1]
        top_n = min(5, len(sorted_order))
        print(f"[NCNN DET DIAG F#{_ncnn_frame_count}] "
              f"{len(above_idx)} candidate(s) above OBJ_THRESH={OBJ_THRESH}  "
              f"(precision mode: fp32)")
        for rank in range(top_n):
            anch_idx = above_idx[sorted_order[rank]]
            stride, gx, gy = _ncnn_stride_info(anch_idx)
            cls_id   = int(class_ids_all[anch_idx])
            cls_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
            # Per-class raw scores for this anchor
            per_cls  = ", ".join(
                f"{(CLASSES[k] if 0 <= k < len(CLASSES) else str(k))}={output[4+k, anch_idx]:.4f}"
                for k in range(nc)
            )
            print(f"[NCNN DET DIAG]  #{rank+1}  anch={anch_idx}  stride={stride}  "
                  f"grid=({gx},{gy})  "
                  f"cx={cx[anch_idx]:.1f}  cy={cy[anch_idx]:.1f}  "
                  f"w={w[anch_idx]:.1f}  h={h[anch_idx]:.1f}  "
                  f"scores=[{per_cls}]")
        # Show box-size distribution for all above-threshold anchors
        w_above = w[above_idx]
        h_above = h[above_idx]
        print(f"[NCNN DET DIAG] box-size distribution for all candidates: "
              f"w∈[{w_above.min():.1f},{w_above.max():.1f}] mean={w_above.mean():.1f}  "
              f"h∈[{h_above.min():.1f},{h_above.max():.1f}] mean={h_above.mean():.1f}")
        # Stride band breakdown
        n8  = int((above_idx < 6400).sum())
        n16 = int(((above_idx >= 6400) & (above_idx < 8000)).sum())
        n32 = int((above_idx >= 8000).sum())
        print(f"[NCNN DET DIAG] stride band breakdown: "
              f"stride-8={n8}  stride-16={n16}  stride-32={n32}")
    # ─────────────────────────────────────────────────────────────────────────

    nboxes, nclasses, nscores = [], [], []
    for c in set(class_ids):
        inds = np.where(class_ids == c)
        b = boxes[inds]
        c_arr = class_ids[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep):
            nboxes.append(b[keep])
            nclasses.append(c_arr[keep])
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def debug_detection_summary(boxes, classes, scores, frame_label='frame'):
    """Print raw detection metadata when debug mode is enabled."""
    if not DEBUG_DETECTIONS:
        return

    if boxes is None or classes is None or scores is None:
        print(f'[DEBUG] {frame_label}: no detections')
        return

    class_ids = [int(c) for c in np.asarray(classes).reshape(-1)]
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    summary = {int(class_id): int(count) for class_id, count in zip(unique_ids, counts)}
    top_scores = [round(float(s), 3) for s in np.asarray(scores).reshape(-1)[:5]]

    print(f'[DEBUG] {frame_label}: detections={len(class_ids)}, class_ids={summary}, top_scores={top_scores}')


def draw(image, boxes, scores, classes, frame_label=None):
    prefix = f"[{frame_label}] " if frame_label else ""
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        class_id = int(cl)
        if 0 <= class_id < len(CLASSES):
            class_name = CLASSES[class_id]
        else:
            class_name = f'class_{class_id}'

        print('%s%s @ (%d %d %d %d) %.3f' % (prefix, class_name, top, left, right, bottom, score))

        box_color = tuple(int(c) for c in app_config.DETECTION_BOX_COLOR)
        label_text_color = tuple(int(c) for c in app_config.DETECTION_LABEL_COLOR)
        label_bg_color = tuple(int(c) for c in app_config.DETECTION_LABEL_BACKGROUND_COLOR)
        box_thickness = max(1, int(app_config.DETECTION_BOX_THICKNESS))
        label_scale = float(app_config.DETECTION_LABEL_TEXT_SIZE)
        label_thickness = max(1, int(app_config.DETECTION_LABEL_TEXT_THICKNESS))

        cv2.rectangle(image, (top, left), (right, bottom), box_color, box_thickness)

        label = '{0} {1:.2f}'.format(class_name, score)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
        label_top = max(0, left - label_height - baseline - 6)
        label_left = top
        label_right = top + label_width + 6
        label_bottom = label_top + label_height + baseline + 6

        if label_top <= 0:
            label_top = min(image.shape[0] - (label_height + baseline + 6), bottom + 6)
            label_bottom = label_top + label_height + baseline + 6

        cv2.rectangle(image, (label_left, label_top), (label_right, label_bottom), label_bg_color, thickness=-1)
        cv2.putText(image, label,
                    (label_left + 3, label_bottom - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_text_color, label_thickness)


def preprocess_frame(frame, platform):
    img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if platform in ['pytorch', 'onnx', 'ncnn']:
        input_data = img.transpose((2, 0, 1))
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        input_data = input_data / 255.
    else:
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)
    return input_data


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('.onnx') and getattr(args, 'gpu_opencl', False):
        # GPU mode: run the ONNX on the Mali-G610 via OpenCV-DNN + OpenCL.
        # (ncnn+Vulkan mis-computes YOLO11 on this GPU — see opencv_executor.py.)
        platform = 'opencv'
        from src.processing.opencv_executor import OpenCV_OpenCL_model_container
        model = OpenCV_OpenCL_model_container(args.model_path, use_opencl=True)
    elif model_path.endswith('.onnx'):
        platform = 'onnx'
        from onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    elif os.path.isdir(model_path):
        platform = 'ncnn'
        from src.processing.ncnn_executor import NCNN_model_container
        model = NCNN_model_container(args.model_path, use_vulkan=True)
    else:
        assert False, '{} is not rknn/pytorch/onnx/ncnn model'.format(model_path)
    print('Model-{} is {} model, starting inference'.format(model_path, platform))
    return model, platform


def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO11 inference script for RKNN/ONNX/PyTorch models.')
    # model params
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path (.pt, .onnx or .rknn)')
    parser.add_argument('--target', type=str, default='rk3588',
                        help='target RKNPU platform (default: rk3588)')
    parser.add_argument('--device_id', type=str, default=None,
                        help='device id for RKNN runtime')

    # display / save
    parser.add_argument('--img_show', action='store_true', default=False,
                        help='show detection results in a window (image mode)')
    parser.add_argument('--img_save', action='store_true', default=False,
                        help='save detection results to ./result/')

    # input source — mutually exclusive modes
    parser.add_argument('--img_folder', type=str, default='.',
                        help='folder with input images (default: current directory)')
    parser.add_argument('--video_source', type=str, default=None,
                        help='video file path, camera index (e.g. "0"), or RTSP URL.\n'
                             'When provided, runs continuous inference; press "q" to stop.')

    # --- COCO mAP evaluation (requires pycocotools — uncomment to enable) ---
    # parser.add_argument('--anno_json', type=str, default=None,
    #                     help='COCO annotation JSON for mAP evaluation')
    # parser.add_argument('--coco_map_test', action='store_true',
    #                     help='enable COCO mAP evaluation (image mode only)')

    # inference parameters
    parser.add_argument('--obj_thresh', type=float, default=0.25,
                        help='object confidence threshold (default: 0.25)')
    parser.add_argument('--nms_thresh', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--img_size', type=int, default=640,
                        help='inference image size, square (default: 640)')
    parser.add_argument('--classes_file', type=str, default=None,
                        help='path to a text file with one class name per line.\n'
                             'Falls back to the 80-class COCO list if not provided.')
    parser.add_argument('--debug_detections', action='store_true', default=False,
                        help='print raw detection class ids and score summaries')

    args = parser.parse_args()

    # --- Set global inference parameters ---
    OBJ_THRESH = args.obj_thresh
    NMS_THRESH = args.nms_thresh
    IMG_SIZE = (args.img_size, args.img_size)
    DEBUG_DETECTIONS = args.debug_detections

    # --- Load class names ---
    CLASSES, classes_source = resolve_runtime_classes(args.classes_file)
    coco_id_list = list(range(len(CLASSES)))
    print(f'Loaded {len(CLASSES)} classes from {classes_source}')

    # --- Init model ---
    model, platform = setup_model(args)

    co_helper = COCO_test_helper(enable_letter_box=True)

    # =========================================================================
    # Video / camera mode
    # =========================================================================
    if args.video_source is not None:
        # Accept "0", "1", etc. as camera indices; everything else as a path/URL
        _src = int(args.video_source) if args.video_source.isdigit() else args.video_source
        cap = cv2.VideoCapture(_src)

        if not cap.isOpened():
            print(f'❌ Could not open video source: {args.video_source}')
            model.release()
            exit(1)

        writer = None
        if args.img_save:
            os.makedirs('./result', exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if isinstance(_src, int):
                _out_name = f'camera_{_src}_detected.mp4'
            else:
                _out_name = os.path.splitext(os.path.basename(_src))[0] + '_detected.mp4'
            _out_path = os.path.join('./result', _out_name)
            writer = cv2.VideoWriter(_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f'Saving output video to: {_out_path}')

        print("Running inference. Press 'q' in the display window to stop.")
        _frame_count = 0
        frame_times = []
        inference_times = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _frame_count += 1
            start_inference = time.time()
            input_data = preprocess_frame(frame, platform)
            outputs = model.run([input_data])
            if outputs is None:
                print('Inference failed for current frame, skipping.')
                continue
            boxes, classes, scores = post_process(outputs)
            debug_detection_summary(boxes, classes, scores, frame_label='video frame')

            end_inference = time.time()
            inference_times.append(end_inference - start_inference)
            frame_times.append(time.time())

            fps_actual = calculate_recent_fps(frame_times[-30:])
            avg_inf_time_ms = calculate_recent_average_ms(inference_times[-30:])

            if boxes is not None:
                draw(frame, co_helper.get_real_box(boxes), scores, classes)

            draw_processing_overlay(
                frame,
                app_config.OVERLAY_ENABLED,
                f'Frame: {_frame_count}',
                inference_time_ms=avg_inf_time_ms,
                fps_value=fps_actual,
                text_size=0.5,
                text_color=app_config.OVERLAY_TEXT_COLOR,
            )

            if writer is not None:
                writer.write(frame)

            # Always show window in video mode so 'q' key can be captured
            cv2.imshow('YOLO11 Inference  [press q to stop]', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopped by user.')
                break

        cap.release()
        if writer is not None:
            writer.release()
            print(f'✅ Output video saved to: {_out_path}')
        else:
            print(f'✅ Inference complete ({_frame_count} frames processed)')
        cv2.destroyAllWindows()

    # =========================================================================
    # Image folder mode
    # =========================================================================
    else:
        file_list = sorted(os.listdir(args.img_folder))
        img_list = [p for p in file_list if img_check(p)]

        for i, img_name in enumerate(img_list):
            print('infer {}/{}'.format(i + 1, len(img_list)), end='\r')

            img_path = os.path.join(args.img_folder, img_name)
            if not os.path.exists(img_path):
                print('{} is not found'.format(img_name))
                continue

            img_src = cv2.imread(img_path)
            if img_src is None:
                continue

            input_data = preprocess_frame(img_src, platform)
            outputs = model.run([input_data])
            if outputs is None:
                print(f'Inference failed for {img_name}, skipping.')
                continue
            boxes, classes, scores = post_process(outputs)
            debug_detection_summary(boxes, classes, scores, frame_label=img_name)

            if args.img_show or args.img_save:
                print('\n\nIMG: {}'.format(img_name))
                img_p = img_src.copy()
                if boxes is not None:
                    draw(img_p, co_helper.get_real_box(boxes), scores, classes)

                if args.img_save:
                    os.makedirs('./result', exist_ok=True)
                    result_path = os.path.join('./result', img_name)
                    cv2.imwrite(result_path, img_p)
                    print('Detection result saved to {}'.format(result_path))

                if args.img_show:
                    cv2.imshow('YOLO11 Inference', img_p)
                    cv2.waitKeyEx(0)

            # --- COCO mAP record (uncomment along with --coco_map_test arg above to enable) ---
            # if args.coco_map_test and boxes is not None and classes is not None and scores is not None:
            #     for j in range(boxes.shape[0]):
            #         co_helper.add_single_record(
            #             image_id=int(img_name.split('.')[0]),
            #             category_id=coco_id_list[int(classes[j])],
            #             bbox=boxes[j],
            #             score=round(scores[j], 5).item(),
            #         )

        print(f'\n✅ Inference complete ({len(img_list)} images)')

        # --- COCO mAP evaluation (uncomment along with --coco_map_test arg above to enable) ---
        # if args.coco_map_test:
        #     pred_json = args.model_path.split('.')[-2] + '_{}'.format(platform) + '.json'
        #     pred_json = pred_json.split('/')[-1]
        #     pred_json = os.path.join('./', pred_json)
        #     co_helper.export_to_json(pred_json)
        #     from coco_utils import coco_eval_with_json   # requires: pip install pycocotools
        #     coco_eval_with_json(args.anno_json, pred_json)

    model.release()
