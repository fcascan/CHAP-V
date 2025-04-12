from rknnlite.api import RKNNLite
import cv2 
import numpy as np
import time
import logging
import os
import sys
import subprocess

# Verificar si el programa se está ejecutando como superusuario
if os.geteuid() != 0:
    print("El programa necesita permisos de superusuario.")
    try:
        subprocess.check_call(["sudo", "python3"] + sys.argv)
    except subprocess.CalledProcessError:
        print("Error: No se pudieron obtener permisos de superusuario.")
    sys.exit(1)

# Continuar con el resto del programa
print("Permisos verificados. Iniciando el programa...")


logger = logging.getLogger()
logger.disabled = True

# Inicializar las cámaras USB
MAX_CAMERAS = 3
cameras = []
rknn_instances = []


# Intentar abrir hasta 3 cámaras
for i in range(MAX_CAMERAS):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        cameras.append(cap)
        print(f"Cámara {i} inicializada.")
    else:
        cap.release()

# Verificar que al menos una cámara esté conectada
if len(cameras) == 0:
    print("Error: No se detectaron cámaras conectadas. Conecta al menos una cámara.")
    exit()

print(f"Se detectaron {len(cameras)} cámara(s).")


# Config
import os

# Obtener la ruta base del archivo .py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración de rutas
MODEL_PATH = os.path.join(BASE_DIR, "assets/models/yolo_quant_int8.rknn")

# Crear la carpeta "images" si no existe
OUTPUT_DIR = os.path.join(BASE_DIR, "images")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OBJ_THRESH = 0.25 # Adjust for your tasks (taken from yolov8 default cfg)
NMS_THRESH = 0.45 # Adjust for your tasks (taken from yolov8 default cfg)
IMG_SIZE = (640, 640)
EXECUTION_TIME = 120 # seconds

# COCO dataset; change for yours (if custom dataset used)
##CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
##            "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
##            "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
##            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
##            "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
##            "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
##            "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

CLASSES = ("person", "gun", "pistol", "revolver", "firearm", "knife", "machete", "katana", "dagger", "sword", "axe", "bat", "club", 
           "bludgeon", "stick", "brass knuckles", "nunchaku", "shuriken", "throwing star", "crossbow", "spear", "harpoon", "boomerang", 
           "slingshot", "catapult", "grenade", "explosive device", "machine gun", "shotgun", "assault rifle", "sniper rifle", "submachine gun",
            "light machine gun", "heavy machine gun", "rocket launcher", "missile launcher", "flamethrower", "taser", "stun gun", "pepper spray")

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
    grid_h, grid_w = position.shape[2:4]
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

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
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

############################################################

# Inicializar una instancia de RKNNLite para cada cámara y asignar un núcleo de NPU
npu_cores = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
for idx, core in enumerate(npu_cores[:len(cameras)]):  # Asignar núcleos según el número de cámaras
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=core)
    rknn_instances.append(rknn)
    print(f"RKNN para cámara {idx} inicializado en NPU_CORE_{idx}.")

# rknn = RKNNLite()
# rknn.load_rknn(MODEL_PATH)
# rknn.init_runtime()


############################################################

# Configuración de FPS y tiempo de ejecución
fps_per_camera = [[] for _ in range(len(cameras))]  # Lista para almacenar FPS por cámara
start_global = time.time()
imgs_to_draw = [None] * len(cameras)  # Almacena las últimas imágenes procesadas de cada cámara

while time.time() - start_global < EXECUTION_TIME:
    for idx, cap in enumerate(cameras):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: No se pudo leer el frame de la cámara {idx}.")
            continue

        # Preprocesar el frame capturado
        start_time = time.time()
        img = cv2.resize(frame, IMG_SIZE)
        img = np.expand_dims(img, 0)  # RKNN espera una imagen de 4 dimensiones
        outputs = rknn_instances[idx].inference(inputs=[img])
        boxes, classes, scores = post_process(outputs)
        end_time = time.time()

        # Calcular FPS para esta cámara
        fps = 1 / (end_time - start_time)
        fps_per_camera[idx].append(fps)

        if boxes is not None and classes is not None and scores is not None:
            imgs_to_draw[idx] = frame.copy()
            for b, label, s in [(box, CLASSES[c], score) for box, c, score in zip(boxes, classes, scores) if c < len(CLASSES)]:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(imgs_to_draw[idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(imgs_to_draw[idx], f"{label}: {s:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostrar la imagen procesada en una ventana
        if imgs_to_draw[idx] is not None:
            cv2.imshow(f"Detections Camera {idx}", imgs_to_draw[idx])

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
for cap in cameras:
    cap.release()
cv2.destroyAllWindows()

# Guardar las últimas imágenes procesadas
for idx, img in enumerate(imgs_to_draw):
    if img is not None:
        output_path = os.path.join(OUTPUT_DIR, f"inference_output_cam{idx}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Imagen de la cámara {idx} guardada en: {output_path}")

# Mostrar el Average FPS por cámara
for idx, fps_list in enumerate(fps_per_camera):
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"Average FPS para la cámara {idx}: {avg_fps:.2f}")
    else:
        print(f"No se pudo calcular el FPS para la cámara {idx}.")
