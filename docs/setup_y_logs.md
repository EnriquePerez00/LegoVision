# Parámetros y Criterios del Pipeline de Entrenamiento de LegoVision

Este documento detalla las especificaciones técnicas, números de imágenes de entrenamiento, criterios de parada y simetrías geométricas empleadas en los pipelines de entrenamiento de LegoVision, optimizados para el chipset **Apple Silicon M4**.

---

## 1. Pipeline de Detección: YOLO11n-seg

El objetivo de los detectores YOLO es localizar y segmentar (obtener máscaras) de las piezas sobre la cinta transportadora.

### Parámetros de Entrenamiento por Defecto
* **Épocas Máximas**: 35
* **Batch Size**: 16 (optimizado para saturar ~5.5GB de memoria unificada sin provocar paginación en macOS)
* **Resolución de Imagen (`imgsz`)**: 640 px
* **Dispositivo de Cómputo**: `mps` (Metal Performance Shaders / Apple Silicon GPU)
* **AMP (Mixed Precision)**: Desactivado (`amp=False`) debido a inestabilidades del compilador MPS en macOS.
* **Workers**: 2 (`workers=2`) para evitar cuellos de botella en lectura/escritura de imágenes.

### Criterio de Parada Temprana (Early Stopping)
* **Paciencia (`patience`)**: 15 épocas.
* **Descripción**: Si la métrica de validación (pérdida agregada y mAP50) no muestra una mejora de al menos $0.001$ durante 15 épocas consecutivas, el proceso de entrenamiento de YOLO se detendrá automáticamente. Esto evita el sobreajuste y ahorra ciclos de cómputo en la GPU del M4.

### Cantidad de Imágenes del Dataset
* **Tamaño del Dataset**: 1,500 imágenes.
* **Criterio de generación**:
  - Imágenes con fondo sólido (Cinta azul petróleo y laterales de aluminio mate).
  - Una sola pieza aleatoria del set 75078-1 situada en el centro exacto de la cinta (directamente debajo de la cámara cenital) colocada en una de sus **posiciones estables precalculadas (stable_poses)**, reposando apoyada contra la superficie de la cinta (sin simular caídas por física, lo que acelera drásticamente el proceso).
  - **Cámara Cenital**: Rotación horizontal (Yaw / Eje Z) aleatoria uniforme entre $0^\circ$ y $360^\circ$ para aprender la silueta desde cualquier ángulo.
  - **Cámara Lateral**: Idem, pero la imagen es capturada aleatoriamente desde la posición de una de las dos cámaras laterales (izquierda o derecha).

### Estructura y Optimización del Dataset (Iteración 6 en adelante)
* **Método de División (Train/Val)**: En lugar de copiar físicamente las imágenes a carpetas separadas de `train` y `val` (duplicando espacio en disco), se utiliza el **Método B** nativo de YOLO.
* **Implementación**:
  - Se generan los archivos de texto `train.txt` y `val.txt` en el directorio procesado.
  - Cada archivo de texto contiene las rutas absolutas a las imágenes originales del dataset crudo (`yolo_cenital` o `yolo_lateral`).
  - El archivo `dataset.yaml` apunta directamente a estos archivos `.txt` en lugar de carpetas.
  - Esto mantiene el uso de disco al mínimo y evita la duplicación de miles de imágenes.

---

## 2. Pipeline de Clasificación: DINOv2 K-NN Indexer

DINOv2 (ViT-S/14) extrae características visuales robustas. Para clasificar mediante K-NN, necesitamos indexar un conjunto de embeddings de referencia.

### Criterio de Generación de Referencias
- La pieza se sitúa siempre en el centro exacto de la cinta (directamente debajo de la cámara cenital).
- **Sin simulación de física**: La pieza se coloca directamente apoyando una cara sobre la superficie de la cinta utilizando las orientaciones registradas de `stable_poses`.
- **Cámara Cenital**: Renderizado con la cámara cenital en perspectiva (PERSP) a 15 cm de altura con zoom de 50% (focal de 52.5mm).
- **Cámara Lateral**: Renderizado con una de las dos cámaras laterales en perspectiva (PERSP) a 15 cm de distancia con zoom de 50% (focal de 52.5mm), seleccionada aleatoriamente al inicio de cada pose.

### Cálculo de Embeddings Totales
Para un set como el **75078-1** que contiene unas 42 piezas únicas con un promedio de 2 poses estables por pieza, este criterio balanceado genera aproximadamente **640 embeddings** en lugar de los >2,000 que generaría un muestreo denso ciego. Esto reduce el espacio de búsqueda del KNN y acelera la inferencia a menos de **10ms por recorte** en el M4.


## Log de Ejecución: YOLO-CENITAL-RENDER (2026-06-02 23:21:32)
* **Directorio de imágenes**: `data/yolo_cenital_render`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
  [OK] Template 3298
Import: 0.019743919372558594
    [scale] Template_11477: max=39.8LDU -> 1.592BU (factor=0.04)
  [OK] Template 11477
Import: 0.029128074645996094
    [scale] Template_15068: max=39.8LDU -> 1.592BU (factor=0.04)
  [OK] Template 15068
Import: 0.02763199806213379
    [scale] Template_98138: max=19.8LDU -> 0.792BU (factor=0.04)
  [OK] Template 98138
Import: 0.007803916931152344
    [scale] Template_2431: max=79.8LDU -> 3.192BU (factor=0.04)
  [OK] Template 2431
Import: 0.013216018676757812
    [scale] Template_6636: max=119.8LDU -> 4.792BU (factor=0.04)
  [OK] Template 6636
Import: 0.04903697967529297
    [scale] Template_sw0614: max=111.4LDU -> 4.454BU (factor=0.04)
  [OK] Template sw0614
[Templates] 42/42 cargadas.
[OK] frame 1/5 - 30 piezas
[OK] frame 2/5 - 34 piezas
[OK] frame 3/5 - 20 piezas
[OK] frame 4/5 - 32 piezas
[Empty] frame 5/5
[DONE] 5/5 imagenes generadas en /Users/I764690/Code_personal/LegoVision/data/yolo_cenital
[TIMING] Total: 13.8s | Por imagen: 2.8s | Estimado 500 imgs: 23.0 min
Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:30:33)

Blender quit

```


## Log de Ejecución: YOLO-LATERAL-RENDER (2026-06-02 23:21:33)
* **Directorio de imágenes**: `data/yolo_lateral_render`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
  [OK] Template 3298
Import: 0.006902933120727539
    [scale] Template_11477: max=39.8LDU -> 1.592BU (factor=0.04)
  [OK] Template 11477
Import: 0.007673025131225586
    [scale] Template_15068: max=39.8LDU -> 1.592BU (factor=0.04)
  [OK] Template 15068
Import: 0.0037620067596435547
    [scale] Template_98138: max=19.8LDU -> 0.792BU (factor=0.04)
  [OK] Template 98138
Import: 0.002629995346069336
    [scale] Template_2431: max=79.8LDU -> 3.192BU (factor=0.04)
  [OK] Template 2431
Import: 0.0031442642211914062
    [scale] Template_6636: max=119.8LDU -> 4.792BU (factor=0.04)
  [OK] Template 6636
Import: 0.04350900650024414
    [scale] Template_sw0614: max=111.4LDU -> 4.454BU (factor=0.04)
  [OK] Template sw0614
[Templates] 42/42 cargadas.
[OK] frame 1/5 - 21 piezas
[OK] frame 2/5 - 20 piezas
[Empty] frame 3/5
[OK] frame 4/5 - 33 piezas
[OK] frame 5/5 - 29 piezas
[DONE] 5/5 imagenes generadas en /Users/I764690/Code_personal/LegoVision/data/yolo_lateral
[TIMING] Total: 11.2s | Por imagen: 2.2s | Estimado 500 imgs: 18.7 min
Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:30:33)

Blender quit

```


## Log de Ejecución: YOLO-CENITAL-RENDER (2026-06-02 23:33:25)
* **Directorio de imágenes**: `data/yolo_cenital_render`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
  [OK] Template 3298
Import: 0.018138885498046875
    [scale] Template_11477: max=39.8LDU -> 1.592BU (factor=0.04)
  [OK] Template 11477
Import: 0.02104020118713379
    [scale] Template_15068: max=39.8LDU -> 1.592BU (factor=0.04)
  [OK] Template 15068
Import: 0.021978139877319336
    [scale] Template_98138: max=19.8LDU -> 0.792BU (factor=0.04)
  [OK] Template 98138
Import: 0.007205963134765625
    [scale] Template_2431: max=79.8LDU -> 3.192BU (factor=0.04)
  [OK] Template 2431
Import: 0.009877920150756836
    [scale] Template_6636: max=119.8LDU -> 4.792BU (factor=0.04)
  [OK] Template 6636
Import: 0.10236620903015137
    [scale] Template_sw0614: max=111.4LDU -> 4.454BU (factor=0.04)
  [OK] Template sw0614
[Templates] 42/42 cargadas.
[OK] frame 1/5 - 33 piezas
[OK] frame 2/5 - 29 piezas
[Empty] frame 3/5
[OK] frame 4/5 - 25 piezas
[OK] frame 5/5 - 22 piezas
[DONE] 5/5 imagenes generadas en /Users/I764690/Code_personal/LegoVision/data/yolo_cenital
[TIMING] Total: 18.8s | Por imagen: 3.8s | Estimado 500 imgs: 31.4 min
Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:30:33)

Blender quit

```


## Log de Ejecución: DINOV2-CENITAL (2026-06-02 23:38:46)
* **Directorio de imágenes**: `data/dinov2_cenital`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
=== LEGOVISION PIPELINE DINOV2-CENITAL ===
Fecha de inicio: 2026-06-02 23:38:43
Paso 1: Generando referencias DINOv2 Cenital en /Users/I764690/Code_personal/LegoVision/data/dinov2_cenital...
Traceback (most recent call last):
  File "/Users/I764690/Code_personal/LegoVision/scripts/generate_physics_ref_multiangle.py", line 19, in <module>
    from generate_synthetic_set import (
    ...<3 lines>...
    )
ModuleNotFoundError: No module named 'generate_synthetic_set'
Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:30:33)

Blender quit
Paso 2: Iniciando indexación de embeddings DINOv2 Cenital...
Using cache found in /Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[LegoVision Index] Cargando DINOv2 ViT-S/14 en mps...
[LegoVision Index] DINOv2 cargado.
[WARN] No encontrado: /Users/I764690/Code_personal/LegoVision/data/dinov2_cenital
[INFO] No encontrado directorio multi-ángulo: /Users/I764690/Code_personal/LegoVision/data/dinov2_cenital

[LegoVision Index] Indexación completada:
  ✅ 0 embeddings guardados
  ❌ 0 errores

```


## Log de Ejecución: DINOV2-CENITAL (2026-06-02 23:54:41)
* **Directorio de imágenes**: `data/dinov2_cenital`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
=== LEGOVISION PIPELINE DINOV2-CENITAL ===
Fecha de inicio: 2026-06-02 23:54:38
Paso 1: Generando referencias DINOv2 Cenital en /Users/I764690/Code_personal/LegoVision/data/dinov2_cenital...
Traceback (most recent call last):
  File "/Users/I764690/Code_personal/LegoVision/scripts/generate_physics_ref_multiangle.py", line 19, in <module>
    from generate_synthetic_set import (
    ...<3 lines>...
    )
ModuleNotFoundError: No module named 'generate_synthetic_set'
Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:30:33)

Blender quit
Paso 2: Iniciando indexación de embeddings DINOv2 Cenital...
Using cache found in /Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[LegoVision Index] Cargando DINOv2 ViT-S/14 en mps...
[LegoVision Index] DINOv2 cargado.
[WARN] No encontrado: /Users/I764690/Code_personal/LegoVision/data/dinov2_cenital
[INFO] No encontrado directorio multi-ángulo: /Users/I764690/Code_personal/LegoVision/data/dinov2_cenital

[LegoVision Index] Indexación completada:
  ✅ 0 embeddings guardados
  ❌ 0 errores

```


## Log de Ejecución: YOLO-CENITAL-RENDER (2026-06-02 23:55:16)
* **Directorio de imágenes**: `data/yolo_cenital_render`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
[OK] frame 26/50 - 1 piezas
[OK] frame 27/50 - 1 piezas
[OK] frame 28/50 - 1 piezas
[OK] frame 29/50 - 1 piezas
[OK] frame 30/50 - 1 piezas
[OK] frame 31/50 - 1 piezas
[OK] frame 32/50 - 1 piezas
[OK] frame 33/50 - 1 piezas
[OK] frame 34/50 - 1 piezas
[OK] frame 35/50 - 1 piezas
[OK] frame 36/50 - 1 piezas
[OK] frame 37/50 - 1 piezas
[Empty] frame 38/50
[OK] frame 39/50 - 1 piezas
[OK] frame 40/50 - 1 piezas
[OK] frame 41/50 - 1 piezas
[OK] frame 42/50 - 1 piezas
[OK] frame 43/50 - 1 piezas
[OK] frame 44/50 - 1 piezas
[OK] frame 45/50 - 1 piezas
[OK] frame 46/50 - 1 piezas
[OK] frame 47/50 - 1 piezas
[OK] frame 48/50 - 1 piezas
[OK] frame 49/50 - 1 piezas
[OK] frame 50/50 - 1 piezas
[DONE] 50/50 imagenes generadas en /Users/I764690/Code_personal/LegoVision/data/yolo_cenital
[TIMING] Total: 44.0s | Por imagen: 0.9s | Estimado 500 imgs: 7.3 min
Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:30:33)

Blender quit

```


## Log de Ejecución: YOLO-LATERAL-RENDER (2026-06-02 23:55:39)
* **Directorio de imágenes**: `data/yolo_lateral_render`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
[OK] frame 26/50 - 1 piezas
[OK] frame 27/50 - 1 piezas
[OK] frame 28/50 - 1 piezas
[OK] frame 29/50 - 1 piezas
[OK] frame 30/50 - 1 piezas
[OK] frame 31/50 - 1 piezas
[OK] frame 32/50 - 1 piezas
[OK] frame 33/50 - 1 piezas
[OK] frame 34/50 - 1 piezas
[OK] frame 35/50 - 1 piezas
[OK] frame 36/50 - 1 piezas
[OK] frame 37/50 - 1 piezas
[OK] frame 38/50 - 1 piezas
[OK] frame 39/50 - 1 piezas
[Empty] frame 40/50
[OK] frame 41/50 - 1 piezas
[OK] frame 42/50 - 1 piezas
[OK] frame 43/50 - 1 piezas
[OK] frame 44/50 - 1 piezas
[OK] frame 45/50 - 1 piezas
[OK] frame 46/50 - 1 piezas
[OK] frame 47/50 - 1 piezas
[OK] frame 48/50 - 1 piezas
[OK] frame 49/50 - 1 piezas
[OK] frame 50/50 - 1 piezas
[DONE] 50/50 imagenes generadas en /Users/I764690/Code_personal/LegoVision/data/yolo_lateral
[TIMING] Total: 44.1s | Por imagen: 0.9s | Estimado 500 imgs: 7.3 min
Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:30:33)

Blender quit

```


## Log de Ejecución: DINOV2-CENITAL (2026-06-03 00:00:44)
* **Directorio de imágenes**: `data/dinov2_cenital`
* **Resultado**: Completado con éxito.
* **Detalle de logs**:
```
02:36.168  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot045.png'
02:37.138  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot060.png'
02:38.135  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot075.png'
02:39.071  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot090.png'
02:39.990  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot105.png'
02:40.915  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot120.png'
02:41.827  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot135.png'
02:42.740  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot150.png'
02:43.634  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot165.png'
02:44.558  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot180.png'
02:45.478  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot195.png'
02:46.390  render           | Saved: '/Users/I764690/Code_personal/LegoVision/data/dinov2_cenital/ref_59900_C91A09_pose00_rot210.png'
/opt/homebrew/bin/blender: line 2: 26824 Killed: 9               '/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender' "$@"
Paso 2: Iniciando indexación de embeddings DINOv2 Cenital...
Using cache found in /Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/Users/I764690/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[LegoVision Index] Cargando DINOv2 ViT-S/14 en mps...
[LegoVision Index] DINOv2 cargado.
[LegoVision Index] Indexando 0 renders isométricos...

[LegoVision Index] Indexando 135 renders multi-ángulo...

[LegoVision Index] Indexación completada:
  ✅ 0 embeddings guardados
  ❌ 0 errores

```
