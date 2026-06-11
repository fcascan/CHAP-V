# GPU Inference Investigation — RK3588 / Mali-G610

## Sistema

| Componente | Detalle |
|---|---|
| SoC | Rockchip RK3588 |
| CPU | Cortex-A76 × 4 + Cortex-A55 × 4 |
| GPU | ARM Mali-G610 MP4 |
| NPU | RKNPU 3 × 1 TOPS (via RKNN Lite) |
| OS | Ubuntu (aarch64) |
| Python | 3.x |

---

## Diagnóstico de los intentos de inferencia GPU

### Hallazgo 1 — ORT no tiene proveedores GPU

```
ORT available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
ORT OpenCLExecutionProvider present: False
ORT CUDAExecutionProvider present:   False
```

El paquete `onnxruntime` instalado vía `pip` para ARM no incluye proveedores GPU.
No hay wheel oficial de `onnxruntime-gpu` para ARM64 + OpenCL.

**Impacto:** ORT solo puede correr en CPU, independientemente de la configuración.

---

### Hallazgo 2 — cv2.dnn + OpenCL → segmentation fault

OpenCV detecta el Mali G610 correctamente:
```
cv2.ocl.haveOpenCL(): True
OpenCL device name:   Mali-G610 r0p0
OpenCL device vendor: ARM
OpenCL driver version: 3.0
```

Pero al intentar ejecutar inferencia DNN con `DNN_TARGET_OPENCL`, el proceso muere:
```
OpenCL program build log: dnn/dummy
Status -43: CL_INVALID_BUILD_OPTIONS
-cl-no-subgroup-ifp
error: unknown OpenCL C option '-cl-no-subgroup-ifp'
Segmentation fault
```

**Causa raíz:** OpenCV compila sus kernels OpenCL con el flag `-cl-no-subgroup-ifp`
que es exclusivo de GPUs Intel. El compilador OpenCL del Mali G610 no lo reconoce,
la compilación del kernel falla, y OpenCV hace un segfault al intentar recuperarse.

**Fix requerido:** Recompilar OpenCV desde fuente, eliminando o condicionando el flag
a plataformas Intel en `modules/dnn/src/opencl/ocl4dnn_conv_spatial.cpp`. No
parcheable desde Python.

---

### Hallazgo 3 — DRM: card1 es el NPU, no la GPU

```
/sys/class/drm/card0/device → DRIVER=rockchip-drm  (display/VOP)
/sys/class/drm/card1/device → DRIVER=RKNPU         (NPU!)
```

ORT busca GPU vía `/sys/class/drm/card1/device/vendor` (error en log):
```
GPU device discovery failed: Failed to open file: "/sys/class/drm/card1/device/vendor"
```

El Mali G610 **no tiene un nodo DRM expuesto** en este sistema. ORT no puede
descubrir el GPU por este mecanismo.

---

### Hallazgo 4 — Mali blob es OpenCL-only, sin soporte Vulkan

```bash
$ find /usr/lib -name "libmali*"
/usr/lib/aarch64-linux-gnu/libmali-x11/libmali-valhall-g610-g13p0-x11-wayland-gbm.so

$ nm -D libmali-valhall-*.so | grep vkCreateInstance
(sin resultados)
```

La librería Mali instalada (`libmali-g610-x11` v1.0.5-2) no exporta símbolos Vulkan.
Es un blob OpenCL-only compilado para entornos X11/Wayland.

**Consecuencia:** `vkCreateInstance failed -9` (VK_ERROR_INCOMPATIBLE_DRIVER).
Ningún framework que use Vulkan (ncnn, MNN, lllamacpp) puede acceder al Mali GPU.

---

### Hallazgo 5 — TIMVX backend funciona, pero usa VIP (no Mali GPU)

cv2.dnn con `DNN_BACKEND_TIMVX` + `DNN_TARGET_NPU` pasó el probe y ejecutó
la inferencia sin crash. Sin embargo:
- **NPU RKNN usage: 0%** — no usa las cores RKNN
- **GPU usage: ~5%** — actividad del VIP (VeriSilicon Image Processor)
- **CPU usage: ~94%** — la mayor carga sigue en CPU

TIMVX en Rockchip accede al coprocesador **VIP** del complejo NPU
(VeriSilicon TIM-VX / OpenVX), distinto del Mali-G610.

**Problema adicional:** TIMVX devuelve 9 outputs en orden diferente al que espera
`post_process()`. Con el fix de remapping en `onnx_executor.py`, las detecciones
funcionan correctamente.

| Output | Shape | Significado |
|---|---|---|
| [0] | (1, 64, 80, 80) | DFL box coords, escala 80 |
| [1] | (1, 1, 80, 80) | Objectness, escala 80 (descartado) |
| [2] | (1, 64, 40, 40) | DFL box coords, escala 40 |
| [3] | (1, 1, 40, 40) | Objectness, escala 40 (descartado) |
| [4] | (1, 64, 20, 20) | DFL box coords, escala 20 |
| [5] | (1, 1, 20, 20) | Objectness, escala 20 (descartado) |
| [6] | (1, 2, 80, 80) | Class probs, escala 80 |
| [7] | (1, 2, 40, 40) | Class probs, escala 40 |
| [8] | (1, 2, 20, 20) | Class probs, escala 20 |

---

### Hallazgo 6 — PyTorch no funciona en este dispositivo

```
python3 -c "import torch" → Illegal instruction
```

Probablemente el wheel instalado fue compilado para x86 o requiere instrucciones
no disponibles. Imposible convertir modelos `.pt` directamente en el dispositivo.

---

## Resumen del estado actual (actualizado 2026-05-13)

| Backend | Estado | Rendimiento |
|---|---|---|
| ORT CPU | ✅ Funciona | ~397ms/frame, ~2.5 FPS, CPU 94% |
| ORT OpenCL | ❌ No disponible | ORT pip sin provider GPU |
| cv2.dnn OpenCL | ❌ Segfault | Flag Intel `-cl-no-subgroup-ifp` |
| cv2.dnn Vulkan | ❌ Sin ICD | Mali blob anterior OpenCL-only |
| cv2.dnn TIMVX | ⚠️ VIP (no Mali) | ~397ms/frame, CPU 94%, GPU 5% |
| ncnn CPU | ✅ Fallback | Similar a ORT CPU |
| **ncnn Vulkan** | ✅ **Funciona** | **~67ms/frame, ~13 FPS, GPU 58%, CPU 21%** |
| RKNN NPU | ✅ Funciona | ~15ms/frame, ~35 FPS (mejor opción) |

---

## Hallazgo 7 — Modelo ncnn convertido; Vulkan sigue siendo el único bloqueador

### Conversión del modelo (completado)

El modelo `april22_2.pt` fue exportado a formato ncnn en PC con ultralytics:

```bash
from ultralytics import YOLO
YOLO("april22_2.pt").export(format="ncnn", imgsz=640)
```

Los archivos resultantes están en:
```
assets/models/april22_2_ncnn_model/
  model.ncnn.param   ← grafo de la red
  model.ncnn.bin     ← pesos
```

Verificación del `.param`:
```
# Capa de entrada
Input    in0    0 1 in0

# Capa de salida (última línea)
Concat   cat_20    2 1 323 324 out0    0=0
```

Los nombres `in0` y `out0` coinciden con lo que usa `ncnn_executor.py`. El tensor
de salida es la concatenación de las coordenadas de caja decodificadas (`323`) y
las probabilidades de clase tras sigmoid (`324`), con axis=0 (dimensión de canales).

**Formato de salida esperado:** `(4+num_classes, 8400)` = `(6, 8400)` para 2 clases.
El post-processor en `_ncnn_post_process()` maneja ambas orientaciones `(C, N)` y
`(N, C)` mediante transposición condicional.

### Error al ejecutar por primera vez

```
[NCNN] GPU mode: using ncnn model .../april22_2_ncnn_model/model.ncnn.param
vkCreateInstance failed -9
[ERROR] Failed to initialize YOLO11 engines: Vulkan GPU not available (no GPU found).
```

El error `vkCreateInstance failed -9` es `VK_ERROR_INCOMPATIBLE_DRIVER`. El blob
Mali instalado (`libmali-valhall-g610-g13p0-x11-wayland-gbm`) es OpenCL-only y no
exporta símbolos Vulkan. ncnn detecta 0 GPUs y `check_vulkan_available()` retorna
`(False, "no GPU found")`.

**Bug**: `NCNN_model_container.__init__()` lanzaba `RuntimeError` en este caso,
abortando toda la inicialización del engine.

**Fix aplicado** (`ncnn_executor.py`): cuando Vulkan no está disponible, el
constructor imprime un warning y cae automáticamente a ncnn CPU mode:

```python
if use_vulkan:
    ok, info = check_vulkan_available()
    if ok:
        ncnn.create_gpu_instance()
        self._gpu_instance_created = True
    else:
        print(f"[NCNN] WARNING: Vulkan not available ({info}). Falling back to ncnn CPU mode.")
        use_vulkan = False   # continúa sin Vulkan
```

### ncnn CPU mode — resultados obtenidos (run 2026-05-13)

Con el fallback a CPU implementado, la primera ejecución confirmó:

```
[NCNN] WARNING: Vulkan not available (no GPU found). Falling back to ncnn CPU mode.
[NCNN] Loaded .../model.ncnn.param  Vulkan=OFF
[NCNN] First inference: raw shape=(6, 8400)  mean_abs=143.1208
[DETECTIONS] Classes found: {'knife': 1}
knife @ (2 0 637 360) 0.332
```

- `raw shape=(6, 8400)` confirma el formato `(4+classes, anchors)` esperado ✅
- `mean_abs=143.1208` indica outputs con valores reales (no all-zero) ✅
- Detecciones aparecen — el modelo fue convertido correctamente ✅

**Bug adicional detectado: threshold no se aplica correctamente**

El umbral configurado es `obj_threshold = 0.5` pero aparecen detecciones con
scores 0.332, 0.283, 0.256 (todos por debajo del umbral).

Causa raíz: `app_launcher.py` agrega `src/rockchip` a `sys.path`. Cuando
`setup_model()` en `yolo11_infer.py` hace `from ncnn_executor import NCNN_model_container`,
Python registra el módulo como `ncnn_executor` en `sys.modules`. Cuando
`_sync_rockchip_runtime_config()` hace `from src.rockchip import ncnn_executor`,
lo registra como `src.rockchip.ncnn_executor`. **Son dos objetos módulo distintos**.
Modificar el atributo en uno no afecta al otro, así que `OBJ_THRESH` nunca se
actualiza a 0.5 en el módulo que realmente ejecuta `_ncnn_post_process()`.

**Fix aplicado** (`ncnn_executor.py::NCNN_model_container.run()`): en vez de leer
la variable del módulo, se importa config directamente en cada llamada:

```python
try:
    from src.core import config as _cfg
    obj_thresh = _cfg.OBJ_THRESHOLD
    nms_thresh = _cfg.NMS_THRESHOLD
except Exception:
    obj_thresh = OBJ_THRESH    # fallback al default del módulo
    nms_thresh = NMS_THRESH
```

Esto evita completamente el problema de doble registro de módulo.

---

---

## Hallazgo 8 — Estado completo del stack Vulkan en el sistema

Investigación exhaustiva del stack Vulkan (2026-05-13):

| Componente | Estado | Detalle |
|---|---|---|
| `libvulkan.so.1` (loader) | ✅ Instalado | v1.3.204, carga bien |
| Vulkan ICD en `/etc/vulkan/icd.d/` | ❌ Vacío | Sin ningún driver ICD configurado |
| Vulkan ICD en `/usr/share/vulkan/icd.d/` | ❌ Vacío | Sin ningún driver ICD |
| `mesa-vulkan-drivers` | ⚠️ Parcial | Solo layers (`VkLayer_MESA_*`), sin driver GPU |
| `panfrost_dri.so` | ✅ Presente | Driver Mesa OpenGL para Mali, sin símbolos Vulkan |
| `panvk` (Panfrost Vulkan) | ❌ Ausente | No incluido en el build panfork Mesa 23.0.5 |
| `libmali-g610-x11` blob | ❌ OpenCL-only | 0 símbolos `vkCreate*` |
| SwiftShader (Chromium) | ⚠️ Disponible | Software-only, no usa GPU |

**Kernel GPU driver:** El módulo `panfrost` está cargado (`lsmod | grep panfrost`),
pero sin usuarios activos (0 referencias). El `renderD128` está ligado al
`display-subsystem` (VOP, controlador de display), no directamente al Mali.

**Conclusión:** No hay ninguna ruta Vulkan activa hacia el Mali-G610 en este sistema.
Para que ncnn acceda al GPU es imprescindible instalar un blob Mali con Vulkan.

---

## Plan para habilitar inferencia real en Mali-G610 via Vulkan

El único bloqueador para ncnn GPU es el blob Mali. Hay dos vías:

### Vía A — Instalar Mali blob con Vulkan (recomendada)

El blob actual es OpenCL-only. Se necesita una variante con Vulkan (formato GBM):

```bash
# Verificar qué está instalado actualmente
dpkg -l | grep libmali

# Buscar en repos Rockchip BSP
apt-cache search libmali
# Candidatos posibles:
#   libmali-valhall-g610-g13p0-gbm          ← GBM + Vulkan
#   libmali-valhall-g610-g13p0-x11-gbm      ← X11 + GBM + Vulkan (preferido)

# Alternativa: descargar desde releases de Rockchip
# https://github.com/tsukumijima/libmali-rockchip/releases
# Buscar: libmali-valhall-g610-g13p0-*gbm*.deb

# Instalación
sudo dpkg -i libmali-valhall-g610-g13p0-gbm_*.deb

# Registrar el driver Vulkan con el loader
sudo mkdir -p /etc/vulkan/icd.d
sudo tee /etc/vulkan/icd.d/mali.json << 'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "/usr/lib/aarch64-linux-gnu/libmali.so",
        "api_version": "1.2.204"
    }
}
EOF

# Verificar
python3 -c "
import ncnn
ncnn.create_gpu_instance()
count = ncnn.get_gpu_count()
print('GPU count:', count)
if count > 0:
    print('Device:', ncnn.get_gpu_info(0).device_name())
ncnn.destroy_gpu_instance()
"
# Resultado esperado: GPU count: 1   Device: Mali-G610
```

---

## Hallazgo 9 — GPU Vulkan funciona; bounding boxes con bajo detection rate

### Resultado final ncnn + Vulkan (2026-05-13)

Con el blob `libmali-valhall-g610-g24p0-gbm_1.9-1_arm64.deb` instalado:
```
[NCNN] Vulkan GPU available: Mali-G610
[NCNN] Loaded .../model.ncnn.param  Vulkan=ON
GPU count: 2   Device: Mali-G610
```
Performance en benchmark (1 stream, 375 frames):
- Inference mean: **74ms** (mediana 67ms), p95 92ms
- FPS: **13 fps** (vs ~2.5 con CPU)
- CPU: **21%** (vs ~94% con CPU)
- GPU: **58%** activo en 86% de frames ✅

### Problema de detecciones: análisis del post-processor ncnn

**Síntoma**: solo 4 detecciones en 375 frames (1.1%) vs NPU que detecta ~80-90% de frames.
Las pocas detecciones que aparecen tienen boxes razonables (`(0,0,198,60)` en frame 640x360),
pero la mayoría de frames queda sin detecciones.

**Formato del output confirmado** (trazando el `.param`):
```
MemoryData pnnx_189: (8400,)           ← strides por ancla (8/16/32)
MemoryData anchor_points: (8400, 2)    ← posiciones de ancla en grid coords
...→ dist2bbox → cx,cy,w,h en grid coords
...→ mul_20: coords × strides → cx,cy,w,h en pixels (640×640)
Sigmoid sigmoid_0                      ← class probs [0,1] ya aplicado
Concat cat_20 → out0: (6, 8400)        ← [cx,cy,w,h,cls0,cls1]
```

El formato `(4+C, 8400)` con `cx,cy,w,h` en pixel coords es correcto.
El post-processor `_ncnn_post_process()` interpreta correctamente.

**Causa raíz del bajo detection rate**: diferencia de arquitectura entre los dos modelos:

| Aspecto | Modelo ncnn | Modelo RKNN |
|---|---|---|
| Head | Standard ultralytics DFL | Rockchip-patched (sin DFL, 9 outputs) |
| Quantización | FP16 (Vulkan) | INT8 (RKNN calibrado) |
| Score calibración | Scores más bajos | Scores más altos (INT8 bias) |
| Export | `YOLO.export(format="ncnn")` | `torch.onnx.export` con head patched |

El modelo ncnn fue exportado con el head estándar ultralytics (scores FP16 "naturales"),
mientras que el RKNN fue calibrado con el head Rockchip (scores INT8 distintos).
El threshold 0.5 que funciona para RKNN es demasiado alto para el modelo ncnn FP16.

**Fix recomendado**: ejecutar con threshold bajo (0.25) para ver la distribución
real de scores del modelo ncnn, y re-calibrar el umbral según resultados:
```ini
# config.ini — probar con threshold más bajo para modo GPU
obj_threshold = 0.25
```

El diagnóstico `[NCNN-DIAG]` que se imprime en la primera inferencia muestra
la distribución de scores y ayuda a determinar el threshold óptimo:
```
[NCNN-DIAG] anchors with score >= 0.10: N
[NCNN-DIAG] anchors with score >= 0.25: N
[NCNN-DIAG] anchors with score >= 0.50: N
[NCNN-DIAG] Top-5 anchors: class, score, cx, cy, w, h
```

### Vía B — Vulkan ICD apuntando al blob actual (puede no funcionar)

Si el blob x11 tiene los símbolos Vulkan compilados pero simplemente no están
registrados, crear el ICD podría ser suficiente sin reinstalar:

```bash
find /usr/lib -name "libmali*.so" -exec nm -D {} \; | grep vkCreateInstance
# Si aparece → el blob tiene Vulkan; solo falta el ICD
# Si no aparece → hay que instalar un blob con Vulkan
```

---

## Flujo de decisión implementado (modo GPU actual)

```
INFERENCE_DEVICE == "GPU"
  └─ create_yolo11_engine("GPU")
       ├─ NCNN_MODEL_PATH existe? → YES
       │    └─ NCNN_model_container(model.ncnn.param)
       │         ├─ check_vulkan_available()
       │         │    ├─ OK → ncnn Vulkan GPU  ← OBJETIVO
       │         │    └─ FAIL → ncnn CPU (fallback actual)
       └─ NCNN_MODEL_PATH no existe → ONNX_MODEL_PATH
            └─ ONNX_model_container() → TIMVX/ORT CPU
```

---

## Notas adicionales

- El `disable_unnecessary_logging()` en `web_server.py` desactiva el root logger
  antes de inicializar el engine. Todos los mensajes críticos usan `print()`.

- El probe de subproceso (TIMVX/Vulkan/OpenCL en `onnx_executor.py`) verifica:
  (1) sin crash, (2) ≥6 outputs, (3) `mean_abs > 1e-6`. Esto detectó que el backend
  Vulkan de cv2.dnn producía outputs all-zero.

- El remapping de outputs de TIMVX identifica tensores por canal:
  64 ch → DFL box, N>1 ch → clases, 1 ch → objectness (descartado).

- `ncnn_executor.py` usa el sentinel `['ncnn_done', boxes, classes, scores]` en
  `run()` para indicar a `postprocess_outputs()` que el NMS ya está hecho y debe
  saltarse el pipeline DFL de `yolo11_infer.post_process()`.
