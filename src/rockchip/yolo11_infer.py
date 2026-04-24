# =============================================================================
# Origin: https://github.com/airockchip/rknn_model_zoo
# Path:   examples/yolo11/python/yolo11.py
# License: Apache 2.0
# Modified from original:
#   - OBJ_THRESH, NMS_THRESH, IMG_SIZE converted to CLI arguments
#   - CLASSES loaded from --classes_file (text file) instead of hardcoded COCO list
#   - coco_id_list replaced with sequential IDs for non-COCO datasets
#   - --img_folder and --anno_json defaults updated for standalone use
#   - sys.path manipulation removed (executors are co-located in the same directory)
#   - --video_source added for video file and camera inference
#   - press 'q' in the display window to stop video/camera inference
# =============================================================================

import os
import cv2
import argparse
import configparser

from coco_utils import COCO_test_helper
import numpy as np


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


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        class_id = int(cl)
        if 0 <= class_id < len(CLASSES):
            class_name = CLASSES[class_id]
        else:
            class_name = f'class_{class_id}'

        print('%s @ (%d %d %d %d) %.3f' % (class_name, top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(class_name, score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def preprocess_frame(frame, platform):
    img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if platform in ['pytorch', 'onnx']:
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
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, '{} is not rknn/pytorch/onnx model'.format(model_path)
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
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _frame_count += 1
            input_data = preprocess_frame(frame, platform)
            outputs = model.run([input_data])
            if outputs is None:
                print('Inference failed for current frame, skipping.')
                continue
            boxes, classes, scores = post_process(outputs)
            debug_detection_summary(boxes, classes, scores, frame_label='video frame')

            if boxes is not None:
                draw(frame, co_helper.get_real_box(boxes), scores, classes)

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
