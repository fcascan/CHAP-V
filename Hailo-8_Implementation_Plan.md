# Plan: Integración de Hailo-8 como backend de inferencia YOLO11

## Contexto

El proyecto actualmente ejecuta un modelo YOLO11 contra tres backends: NPU interno RK3588 (RKNN), GPU Mali-G610 (ncnn+Vulkan) y CPU (ONNX), con selección dinámica vía `config.ini` o la web UI en Flask/SocketIO.

El Hailo-8 ya está conectado físicamente en el slot M.2 (PCIe `0000:01:00.0`) y es detectable por el sistema. El SDK de Hailo (**HailoRT**) **no está instalado** aún. Se necesita:
1. Instalar el SDK en el dispositivo
2. Convertir el modelo existente al formato `.hef` (Hailo Executable Format) — vía Google Colab
3. Agregar el executor `Hailo_model_container` siguiendo el patrón duck-typed existente
4. Enchufarlo en la factory, config y web UI

---

## Premisas de arquitectura

El proyecto usa **duck-typing** para todos los backends: cada executor implementa `__init__`, `run(inputs) -> list[ndarray]`, `release()`. No hay clase base abstracta. El despacho se hace por extensión de archivo/directorio en `setup_model()` de `yolo11_infer.py`, y a un nivel más alto en `create_yolo11_engine()` de `yolo11_inference.py`.

El flujo de datos es:
```
frame BGR → preprocess_frame() → [1,3,H,W] float32 / uint8 → executor.run() → list[ndarray] → post_process() → boxes/classes/scores
```

Para Hailo, el input esperado es `float32 [1,H,W,3]` (NHWC), lo que requiere un branch de preprocessing ligeramente distinto al de RKNN pero similar al de ONNX/ncnn.

### Convención de idioma

Todo el código fuente (nombres de variables, funciones, clases, comentarios inline, docstrings, mensajes de log y print) debe estar **en inglés**, sin excepción. Este plan está redactado en español como documento de referencia, pero el código que se escriba o modifique debe seguir el idioma del proyecto.

### Convención de headers de archivo

Todo archivo **nuevo** que se cree debe incluir el siguiente header al inicio:

```python
# -*- coding: utf-8 -*-
"""<nombre_del_archivo.py>
<Descripción breve en una línea>
by fcascan 2026
"""
```

Todo archivo **existente** que sea modificado debe:
- Conservar su header actual si ya tiene uno con ese formato.
- Actualizar el año de `2025` a `2026` si aparece en el header.
- Si el archivo no tiene header, agregarlo.

Los siguientes archivos del proyecto tienen actualmente `by fcascan 2025` y deben actualizarse a `2026` cuando sean tocados en esta integración:

| Archivo |
|---|
| `src/core/config.py` |
| `src/processing/yolo11_inference.py` |
| `src/web/web_server.py` |
| `src/web/web_video_processing.py` |
| `src/web/web_camera_processing.py` |
| `src/web/video_integration.py` |
| `src/web/console_integration.py` |
| `src/web/__init__.py` |
| `src/utils/my_htop.py` |
| `src/utils/performance_analyzer.py` |
| `src/__init__.py` |

De estos, los que esta integración modifica directamente son: `config.py`, `yolo11_inference.py` y `web_server.py` — todos deben tener el año actualizado en su header.

### Convención de headers en archivos del directorio `src/rockchip/`

Todo archivo dentro de `src/rockchip/` que sea modificado en el marco de esta integración debe registrar el cambio en su bloque de comentario de cabecera existente. El formato a seguir es el ya establecido en `yolo11_infer.py`:

```python
# Project additions (not part of the original rknn_zoo library):
#   - <descripción concisa del cambio>: <detalle de qué hace y por qué>
```

Si el archivo no tiene un header de ese estilo, se agrega una sección `# Hailo-8 integration additions:` al principio del archivo, después de los imports, documentando cada modificación realizada.

---

## Entorno del sistema (relevante para la API de HailoRT)

| Propiedad | Valor |
|---|---|
| **OS** | Armbian 26.8.0-trunk.125 basado en **Ubuntu 26.04** (Resolute Raccoon) |
| **Kernel** | `6.1.115-vendor-rk35xx` — kernel vendor de Rockchip, **no** el kernel genérico de Ubuntu |
| **Arquitectura** | `aarch64` (ARM64) |
| **Python** | 3.12 (venv del proyecto) |
| **SoC** | RK3588 (OrangePi 5 Max) |

### Implicaciones para HailoRT

1. **Compatibilidad del driver PCIe con el kernel vendor.**
   HailoRT instala un módulo de kernel (`hailo_pci.ko`). El `.deb` oficial de Hailo incluye el driver pre-compilado para kernels genéricos de Ubuntu/Debian. El kernel `6.1.115-vendor-rk35xx` es un fork patched de Rockchip — el módulo pre-compilado **probablemente no cargue** en este kernel.
   
   En ese caso se necesita compilar el driver desde fuente contra los headers del kernel vendor:
   ```bash
   # Instalar headers del kernel vendor
   sudo apt install linux-headers-$(uname -r)
   # Si no están en los repos de Armbian, obtenerlos del build de Armbian correspondiente

   # Compilar el módulo manualmente desde el source del HailoRT deb
   dpkg -x hailort_<version>_arm64.deb hailo_extracted/
   cd hailo_extracted/usr/lib/hailo/driver/
   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
   sudo insmod hailo_pci.ko
   sudo depmod -a
   ```

2. **Ubuntu 26.04 es una release muy reciente.** Los paquetes `.deb` oficiales de Hailo típicamente declaran dependencias contra Ubuntu 22.04 o 24.04. Al instalar con `dpkg -i` usar `--ignore-depends` si hay conflictos de versión de librerías del sistema, o forzar con `--force-depends`. Verificar que `libhailort.so` quede accesible con `ldconfig`.

3. **Python 3.12 en ARM64.** El wheel de HailoRT debe ser específicamente `cp312-cp312-linux_aarch64`. Si Hailo no publica ese wheel, puede ser necesario compilar los bindings Python desde el source del SDK (requiere `cmake`, `pybind11`, y los headers de HailoRT). Confirmar disponibilidad en el Developer Zone antes de descargar.

4. **Verificar carga del módulo después de instalar:**
   ```bash
   lsmod | grep hailo
   dmesg | grep -i hailo
   ls /dev/hailo*
   ```
   Si el módulo no carga automáticamente, agregarlo a `/etc/modules-load.d/hailo.conf`.

---

## Fase 0 — Prerequisitos: conversión del modelo en Google Colab

La Hailo Dataflow Compiler (DFC) requiere x86 Linux, por lo que la conversión se realiza en Google Colab donde ya está disponible el entorno de entrenamiento del modelo.

### 0.1 Notebook de conversión en Google Colab

Agregar las siguientes celdas al notebook existente de entrenamiento/exportación:

```python
# Celda 1 — Instalar Hailo Dataflow Compiler y Model Zoo
!pip install hailo-dataflow-compiler
!pip install hailo-model-zoo

# Celda 2 — Verificar instalación
import hailo_sdk_client
print("Hailo DFC disponible:", hailo_sdk_client.__version__)
```

```python
# Celda 3 — Opción A: compilación directa vía Hailo Model Zoo (recomendado para YOLO11)
# Asegurarse de tener april22_2.onnx disponible en el entorno de Colab
!hailomz compile yolo11n \
    --ckpt /content/assets/models/april22_2.onnx \
    --hw-arch hailo8 \
    --classes 2 \
    --output /content/april22_2.hef
```

```python
# Celda 4 — Opción B: conversión manual ONNX → HEF (si Opción A no soporta el modelo exacto)
# Paso 1: parsear ONNX
!hailo parser onnx /content/assets/models/april22_2.onnx \
    --net-name april22_2

# Paso 2: optimizar (cuantización INT8 — requiere dataset de calibración)
# Preparar un directorio /content/calib_images/ con imágenes representativas
!hailo optimize april22_2.hn \
    --hw-arch hailo8 \
    --calib-path /content/calib_images \
    --output-har april22_2_optimized.har

# Paso 3: compilar a HEF
!hailo compiler april22_2_optimized.har \
    --hw-arch hailo8 \
    --output /content/april22_2.hef
```

```python
# Celda 5 — Descargar el .hef generado
from google.colab import files
files.download('/content/april22_2.hef')
```

> **Nota sobre calibración:** La cuantización INT8 de Hailo requiere un dataset de calibración. Usar entre 100 y 300 imágenes representativas de las clases `pistol` y `knife`. Si ya existe un directorio de validación en el entorno de Colab, reutilizarlo.

> **Nota sobre formato de salida del HEF:** Al compilar, controlar si el HEF incluye el decode de bounding boxes integrado (`--end-node-names` con los nodos post-decode) o si exporta los 9 FPN heads raw. La preferencia es **sin decode integrado** (FPN raw), para que el postprocessing existente `post_process()` funcione sin cambios. Documentar el resultado en el notebook.

### 0.2 Copiar el .hef al OrangePi

```bash
# Desde la máquina local o directamente por SCP desde donde se descargó:
scp april22_2.hef orangepi@<ip>:/home/orangepi/Documents/workspace/PythonYoloRKNPU/assets/models/
```

### 0.3 Verificar el dispositivo en el OrangePi (post-instalación SDK)

```bash
hailortcli fw-control identify        # debe mostrar Hailo-8, PCIe, firmware version
hailortcli run assets/models/april22_2.hef --input-files assets/images/bus.jpg  # smoke test
```

---

## Fase 1 — Dependencias en el dispositivo (OrangePi)

### 1.1 HailoRT en tiempo de ejecución

El dispositivo necesita el **runtime de HailoRT** (driver PCIe + Python bindings). Esto **no** se instala vía `pip install -r requirements.txt` como los paquetes ordinarios porque depende de un `.deb` del sistema y de un wheel específico para ARM64.

**Instalación manual (una sola vez):**
```bash
# 1. Descargar desde https://hailo.ai/developer-zone/ → Software Downloads → HailoRT
#    Archivos necesarios para ARM64 / Armbian:
#      - hailort_<version>_arm64.deb    (driver PCIe + libhailort.so)
#      - hailort-<version>-cp312-cp312-linux_aarch64.whl  (Python bindings)

# 2. Instalar el driver de sistema
sudo dpkg -i hailort_<version>_arm64.deb
sudo ldconfig

# 3. Instalar los bindings Python dentro del venv del proyecto
source venv/bin/activate
pip install hailort-<version>-cp312-cp312-linux_aarch64.whl

# 4. Verificar
hailortcli fw-control identify
python3 -c "import hailo; print('HailoRT OK:', hailo.__version__)"
```

### 1.2 Actualización de `requirements.txt`

Agregar una sección comentada al final del archivo, explicando que el wheel debe instalarse manualmente (igual al patrón de rknnlite):

```
# HailoRT Python bindings (for Hailo-8 PCIe inference)
# Note: hailort must be installed from the official ARM64 wheel — see README Hailo section
# Download from https://hailo.ai/developer-zone/ → Software Downloads → HailoRT
# pip install hailort-<version>-cp312-cp312-linux_aarch64.whl
# hailort>=4.19.0
```

### 1.3 Actualización de `setup.sh`

Agregar un bloque después del paso 5 (librknnrt), siguiendo el mismo patrón que el bloque de RKNN:

```bash
# 7. Install HailoRT (optional — for Hailo-8 PCIe NPU)
HAILO_WHEEL=$(ls installation/hailort-*-cp312-cp312-linux_aarch64.whl 2>/dev/null | head -1)
if [ -f "$HAILO_WHEEL" ]; then
    echo "Installing HailoRT Python bindings from local file..."
    pip install --force-reinstall --no-cache-dir "$HAILO_WHEEL"
    echo "HailoRT installed successfully."
else
    echo "INFO: HailoRT wheel not found in installation/"
    echo "      Hailo-8 inference will not be available."
    echo "      Download from https://hailo.ai/developer-zone/ and place in installation/"
fi

HAILO_DEB=$(ls installation/hailort_*_arm64.deb 2>/dev/null | head -1)
if [ -f "$HAILO_DEB" ]; then
    echo "Installing HailoRT system driver..."
    sudo dpkg -i "$HAILO_DEB"
    sudo ldconfig
fi
```

> El usuario debe descargar manualmente los archivos `.deb` y `.whl` de Hailo y colocarlos en el directorio `installation/` antes de correr `setup.sh`.

### 1.4 Actualización del `README.md`

Agregar una sección nueva **"Hailo-8 NPU Setup (PCIe)"** a continuación de la sección de GPU, con el mismo nivel de detalle que esa sección:

```markdown
## Hailo-8 NPU Setup (PCIe M.2)

Hailo-8 mode uses the **HailoRT** runtime to run inference on the Hailo-8 AI processor
connected via PCIe M.2.

### 1. Install HailoRT

Download both files from [hailo.ai/developer-zone](https://hailo.ai/developer-zone/)
(free account required):
- `hailort_<version>_arm64.deb` — PCIe driver and system library
- `hailort-<version>-cp312-cp312-linux_aarch64.whl` — Python bindings

Place them in the `installation/` directory, then run:

```bash
./setup.sh   # handles HailoRT automatically if files are present
```

Or install manually:
```bash
sudo dpkg -i installation/hailort_<version>_arm64.deb
source venv/bin/activate
pip install installation/hailort-<version>-cp312-cp312-linux_aarch64.whl
```

### 2. Convert your model to .hef

Model conversion requires the Hailo Dataflow Compiler, which runs on x86 Linux.
**Use Google Colab** (see the training notebook — conversion cells are included).

The generated `april22_2.hef` file should be placed in `assets/models/`.

### 3. Configure and run

```ini
# config.ini
[INFERENCE]
inference_device = HAILO

[PATHS]
model_hailo = assets/models/april22_2.hef
```

```bash
sudo ./venv/bin/python main.py
```

### Verify device

```bash
hailortcli fw-control identify   # should show Hailo-8, PCIe, firmware version
python3 -c "import hailo; print(hailo.Device.scan())"
```

### Expected performance (to be measured)

| Mode | Inference (ms) | FPS | Notes |
|------|---------------|-----|-------|
| Hailo-8 (PCIe) | TBD | TBD | INT8 quantized via DFC |
```

Also update the **Inference Devices** section:
```markdown
- **Hailo Mode**: `device = HAILO` - Uses HailoRT with Hailo-8 PCIe AI processor
```

And the **Performance table** to add a Hailo row once measured.

---

## Fase 2 — Nuevo executor: `src/processing/hailo_executor.py`

Crear nuevo archivo siguiendo el mismo patrón que `rknn_executor.py`:

```python
"""Hailo-8 executor for PCIe inference via HailoRT.

Hailo-8 integration additions:
  - Hailo_model_container: duck-typed executor (run/release interface) using
    HailoRT VDevice + InferVStreams high-level API. Internally transposes
    CHW float32 input (from preprocess_frame) to NHWC as required by Hailo.
"""
import numpy as np


class Hailo_model_container:
    def __init__(self, model_path, target=None, device_id=None):
        import hailo
        self._vdevice = hailo.VDevice()
        network_group = self._vdevice.configure(model_path)[0]
        self._network_group = network_group
        self._input_vstreams_params  = hailo.InputVStreamParams.make(network_group)
        self._output_vstreams_params = hailo.OutputVStreamParams.make(network_group)
        self._infer_pipeline = hailo.InferVStreams(
            network_group,
            self._input_vstreams_params,
            self._output_vstreams_params,
        )
        info = network_group.get_input_vstream_infos()[0]
        self._input_name  = info.name
        self._input_shape = info.shape   # (H, W, C) NHWC
        self._activated   = network_group.activate()

    def run(self, inputs):
        # inputs[0]: float32 [1,3,H,W] CHW from preprocess_frame()
        # Hailo expects NHWC → transpose internally
        chw  = inputs[0][0]                          # [3,H,W]
        nhwc = np.transpose(chw, (1, 2, 0))[None]    # [1,H,W,3]
        result = self._infer_pipeline.infer({self._input_name: nhwc.astype(np.float32)})
        return [v for v in result.values()]

    def release(self):
        if self._activated:
            self._activated.__exit__(None, None, None)
        if self._vdevice:
            del self._vdevice
            self._vdevice = None
```

> **Nota sobre postprocessing:** Si el HEF exporta los FPN heads raw (recomendado), la salida serán múltiples tensores y `post_process()` ya existente funciona sin cambios. Si el HEF incluye decode integrado (1 tensor `[nc+4, 8400]`), se usa `post_process_ncnn()`. Esto se configura en el branch de `postprocess_outputs()` del engine una vez conocido el formato real del HEF generado.

---

## Fase 3 — Modificaciones al engine principal

### 3.1 `src/rockchip/yolo11_infer.py` — `setup_model()`

**Modificación:** Agregar rama de detección para archivos `.hef` en `setup_model()`.

**Header a agregar** (en el bloque "Project additions" del archivo, líneas ~14-20):
```python
#   - setup_model(): '.hef' branch added to route Hailo-8 HEF files to
#     Hailo_model_container (src/processing/hailo_executor.py); platform = 'hailo'
```

**Código a agregar** dentro de `setup_model(args)`, después del bloque de `.onnx`:
```python
elif model_path.endswith('.hef'):
    from src.processing.hailo_executor import Hailo_model_container
    platform = 'hailo'
    model = Hailo_model_container(model_path, args.target, args.device_id)
```

### 3.2 `src/processing/yolo11_inference.py`

**`preprocess_frame()`** — agregar `'hailo'` al branch CHW float32 (línea ~90):
```python
if self.platform in ('pytorch', 'onnx', 'ncnn', 'hailo'):
```
(La transposición a NHWC se hace internamente en `Hailo_model_container.run()`)

**`postprocess_outputs()`** — agregar rama para `'hailo'` (línea ~135):
```python
elif self.platform == 'hailo':
    if len(outputs) == 1:
        boxes, classes, scores = rockchip_yolo.post_process_ncnn(outputs)
    else:
        boxes, classes, scores = rockchip_yolo.post_process(outputs)
```

**`create_yolo11_engine()`** — agregar rama `"HAILO"` (línea ~294):
```python
elif device_type == "HAILO":
    model_path = app_config.HAILO_MODEL_PATH
    platform = app_config.ROCKCHIP_TARGET
```

### 3.3 `src/core/config.py` — `load_config()`

Agregar `HAILO_MODEL_PATH` al global (línea 43), al bloque de parseo de paths (después de `NCNN_MODEL_PATH`, línea ~67), y al dict de retorno:

```python
# Global (línea 43):
global ..., HAILO_MODEL_PATH

# Bloque de paths (después de NCNN_MODEL_PATH):
model_hailo_cfg = parser.get("PATHS", "model_hailo", fallback="assets/models/april22_2.hef")
HAILO_MODEL_PATH = os.path.join(BASE_DIR, model_hailo_cfg)

# Dict de retorno:
'hailo_model_path': HAILO_MODEL_PATH,
```

### 3.4 `config.ini` — agregar entrada en `[PATHS]`

```ini
[PATHS]
model_hailo = assets/models/april22_2.hef
```

---

## Fase 4 — System setup y disponibilidad

### 4.1 `src/core/dependency_manager.py` — nueva función

```python
def check_hailo_availability():
    """Check if Hailo-8 PCIe device and HailoRT SDK are available."""
    try:
        import hailo
        devices = hailo.Device.scan()
        if not devices:
            return False, "HailoRT installed but no Hailo device found"
        return True, f"Hailo-8 available: {devices[0]}"
    except ImportError:
        return False, "HailoRT not installed — see README Hailo-8 section"
    except Exception as e:
        return False, f"Hailo check failed: {e}"
```

### 4.2 `src/core/system_setup.py` — `setup_inference_device()`

Agregar import en línea 8:
```python
from .dependency_manager import check_and_install_dependencies, check_rknn_availability, check_gpu_availability, check_hailo_availability, ensure_root_permissions, require_root_permissions
```

Agregar rama `"HAILO"` en `setup_inference_device()`, antes del bloque `# For CPU mode or fallback`:
```python
elif inference_device == "HAILO":
    hailo_ok, hailo_msg = check_hailo_availability()
    if hailo_ok:
        print(f"[INFO] {hailo_msg}")
        return "HAILO", True, {"hailo_available": True}
    print(f"[WARNING] Hailo not available: {hailo_msg}")
    print("[INFO] Falling back to CPU inference mode...")
    return "CPU", False, {}
```

---

## Fase 5 — Web UI

### 5.1 `src/web/templates/index.html`

Agregar opción en el `<select id="inference-device-select">` (línea ~228):
```html
<option value="HAILO">Hailo-8 (PCIe)</option>
```

Agregar selector de modelo HEF junto a los selectores de RKNN/ONNX existentes:
```html
<label for="model-hailo-select">Hailo Model (HEF):</label>
<select id="model-hailo-select" name="model_hailo">
    <option value="">Loading models...</option>
</select>
```

### 5.2 `src/web/web_server.py`

En el endpoint `GET /api/config` (línea ~172), agregar al dict de respuesta:
```python
'hailo_model_path': HAILO_MODEL_PATH,
```

En el endpoint `GET /api/models`, extender para incluir archivos `.hef`:
```python
hailo_models = [f for f in os.listdir(models_dir) if f.endswith('.hef')]
# Incluir en la respuesta JSON bajo clave 'hailo_models'
```

En el endpoint `POST /api/config` (línea ~208), agregar persistencia del campo:
```python
if 'model_hailo' in data:
    parser.set('PATHS', 'model_hailo', data['model_hailo'])
```

### 5.3 `src/web/static/script.js`

En la función que puebla los dropdowns de modelos (llamada en `loadConfig()` o similar), agregar lógica para poblar `model-hailo-select` con los HEF devueltos por `/api/models`.

---

## Archivos a modificar (resumen)

| Archivo | Cambio |
|---|---|
| `config.ini` | Agregar `model_hailo` en `[PATHS]` |
| `requirements.txt` | Sección comentada para HailoRT wheel |
| `setup.sh` | Bloque paso 7: instalar HailoRT `.deb` y `.whl` si presentes en `installation/` |
| `README.md` | Sección "Hailo-8 NPU Setup", actualizar tabla de devices y performance |
| `src/core/config.py` | Global `HAILO_MODEL_PATH`, parseo, retorno |
| `src/core/dependency_manager.py` | Nueva `check_hailo_availability()` |
| `src/core/system_setup.py` | Rama `"HAILO"` en `setup_inference_device()`, nuevo import |
| `src/rockchip/yolo11_infer.py` | Rama `.hef` en `setup_model()` + entrada en header |
| `src/processing/yolo11_inference.py` | Rama `HAILO` en factory, preprocess, postprocess |
| `src/web/web_server.py` | Exponer `HAILO_MODEL_PATH`, listar HEF en `/api/models`, persistir `model_hailo` |
| `src/web/templates/index.html` | Opción `HAILO` en `<select>`, selector HEF |
| `src/web/static/script.js` | Poblar dropdown de HEF |

## Archivo nuevo a crear

| Archivo | Contenido |
|---|---|
| `src/processing/hailo_executor.py` | `Hailo_model_container` — duck-typed, HailoRT VDevice API, con docstring de integración |

---

## Verificación end-to-end

1. **Verificar SDK en el dispositivo:**
   ```bash
   python3 -c "import hailo; print(hailo.Device.scan())"
   hailortcli fw-control identify
   ```
2. **Verificar HEF generado:**
   ```bash
   hailortcli run assets/models/april22_2.hef --input-files assets/images/bus.jpg
   ```
3. **Prueba console:**
   ```bash
   sudo ./venv/bin/python main.py   # con inference_device = HAILO en config.ini
   ```
   Verificar que aparezca `[DETECTIONS]` y FPS en stdout.
4. **Prueba web:**
   ```bash
   sudo ./venv/bin/python main.py --web
   ```
   Abrir `http://<ip>:8080`, seleccionar "Hailo-8 (PCIe)" en el dropdown, iniciar procesamiento, verificar video con detecciones y métricas de FPS.
5. **Benchmark comparativo:** Ejecutar con cada dispositivo (NPU/GPU/CPU/HAILO) y comparar los CSV descargables de `/api/download/latest_csv`.

---

## Notas y riesgos

- **HailoRT API puede variar entre versiones** (4.x vs 5.x). El executor usa la API de alto nivel `VDevice + InferVStreams` que es estable, pero los nombres de método pueden diferir. Verificar con `python3 -c "import hailo; help(hailo.VDevice)"` después de instalar.
- **Formato de salida del HEF depende de cómo se compile.** Si el HEF incluye decode integrado, la salida es 1 tensor y se usa `post_process_ncnn()`. Si exporta los FPN heads raw (recomendado), son múltiples tensores y se usa `post_process()`. El notebook de Colab debe documentar qué opción se usó.
- **Calibración para cuantización INT8.** Para mantener precisión, proveer al DFC un dataset de calibración representativo de las clases pistola/cuchillo (100–300 imágenes). Si la precisión baja significativamente, considerar INT16 o Float16 si el HEF lo soporta.
- **Root permissions** ya son requeridas por el proyecto; el acceso PCIe al Hailo también las requiere.
- **El wheel de HailoRT es específico de Python 3.12 y ARM64.** Confirmar la versión exacta de Python en el venv (`python3 --version`) al descargar el wheel.