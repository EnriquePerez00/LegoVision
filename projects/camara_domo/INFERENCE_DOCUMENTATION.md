# Documentación del Proceso de Inferencia - Cámara Domo

Este documento describe en detalle el proceso de inferencia del módulo **Cámara Domo** (`camara_domo`), detallando el setup de hardware/cámara, los modelos neuronales en uso, y las estrategias algorítmicas de clasificación geométrica y de color.

---

## 1. Setup y Arquitectura del Sistema

El setup de la **Cámara Domo** está diseñado para identificar piezas LEGO en una cinta transportadora utilizando un sistema multicámara:
*   **Cámara Cenital (Superior):** Ubicada directamente sobre la cinta. Captura vistas ortogonales (superiores) para medir el área cenital aparente de la pieza y realizar la clasificación de color primaria.
*   **Cámara Lateral / Frontal (Opcional):** Ubicada en un lateral para capturar el perfil de altura de la pieza. Su activación está controlada por software mediante la variable de configuración `camara_lateral` (dentro de `config.yaml`).

---

## 2. Modelos Neuronales en Uso

El pipeline de inferencia secuencial combina múltiples modelos neuronales especializados para cada fase del proceso:

| Modelo / Tarea | Archivo / Repositorio | Dispositivo (MPS/CPU) | Descripción |
| :--- | :--- | :--- | :--- |
| **Detección Cenital** | `models/yolo_cenital.pt` | GPU (MPS) / CPU | YOLO11 para localizar la caja delimitadora (bbox) de la pieza en la vista cenital. |
| **Detección Lateral** | `models/yolo_lateral.pt` | GPU (MPS) / CPU | YOLO11 para localizar la pieza en la cámara lateral (si está activa). |
| **Keypoints Cenital** | `models/yolo_cenital_pose.pt` | GPU (MPS) / CPU | YOLO-Pose para extraer puntos clave de la pieza desde arriba. |
| **Keypoints Lateral** | `models/yolo_lateral_pose.pt` (o `yolo_frontal_pose.pt`) | GPU (MPS) / CPU | YOLO-Pose para extraer puntos clave del perfil de altura. |
| **Segmentación** | `mobile_sam.pt` | GPU (MPS) / CPU | MobileSAM para segmentar con precisión pixel a pixel la silueta de la pieza. |
| **Clasificador de Pieza** | `timm: efficientnetv2_rw_s` / `EfficientNetV2-B0` | GPU (MPS) / CPU | Clasificador neuro-simbólico que extrae embeddings de la geometría (imagen en escala de grises con fondo enmascarado). |
| **Clasificador de Color** | `models/color_mlp_model.pt` | GPU (MPS) / CPU | MLP dedicado (Multi-Layer Perceptron) que clasifica el color usando estadísticas cromáticas Lab/HSV. |

---

## 3. Estrategias y Algoritmos de Inferencia

### A. Clasificación Cromática (MLP de Color)
Para evitar que la red neuronal asocie la forma física de la pieza al color (sesgo de forma), se implementa una estrategia puramente estadística:
1.  **Segmentación y Erosión:** Se obtiene la máscara binaria de MobileSAM y se le aplica una erosión morfológica de kernel $5 \times 5$ para eliminar contaminación de bordes (sangrado de fondo).
2.  **Filtro de Especularidades:** En el espacio **HSV**, se filtran los píxeles con saturación muy baja ($S < 25$) y brillo muy alto ($V \ge 230$) para descartar reflejos lumínicos.
3.  **Extracción de Características:** Sobre los píxeles válidos restantes, se calcula la **media y desviación estándar** de los canales **CIELAB** ($L, a, b$) y **HSV** ($H, S, V$), resultando en un vector plano de 12 dimensiones.
4.  **Inferencia MLP:** El vector se normaliza con el scaler guardado en `color_mlp_metadata.json` y se procesa en el MLP, devolviendo el color con una precisión del **97%**.
5.  **Fallback Seguro (Paso 1):** Si los archivos del MLP no existen, el sistema calcula la media RGB y busca el código de color más cercano en la paleta usando la distancia delta perceptual de CIELAB.

### B. Medición y Triangulación de Altura Lateral
Cuando la cámara lateral está activa:
*   **Segmentación Lateral:** Se proyectan las coordenadas horizontales de la cámara cenital para acotar la búsqueda y alimentar a MobileSAM.
*   **Triangulación por Keypoints:** Si ambos modelos de Pose (`yolo_cenital_pose.pt` y `yolo_lateral_pose.pt`) estiman keypoints con suficiente confianza ($\ge 0.20$), se corre un algoritmo de correspondencia epipolar (`_kpts_observer.py`) para calcular la altura física en milímetros.
*   **Fallback SAM:** Si la triangulación falla, se mide la altura del bbox de SAM escalado por la calibración nominal (`px_per_mm_lateral`).

### C. Clasificación de Pieza (EfficientNetV2 + Filtro de Base de Datos)
1.  **Gating Determinista (Fase 1):** Se calcula el área cenital aparente en milímetros cuadrados a partir de la máscara SAM cenital (compensando la distancia focal y altura local). Se interroga la base de datos (Supabase/PostgreSQL o caché local JSON) para obtener únicamente las piezas candidatas cuya superficie nominal coincida dentro de una tolerancia del $\pm 35\%$.
2.  **Extracción en Escala de Grises (Fase 2):** El recorte cenital (y lateral si aplica) de la pieza se enmascara con fondo negro y se convierte a escala de grises. Esto asegura que la red clasifique solo basándose en la silueta y textura tridimensional, no en el color.
3.  **Matching KNN Restringido (Fase 3):** Se extraen embeddings de la pieza y se comparan contra los embeddings de referencia pre-calculados **únicamente** de los candidatos filtrados en el Gating, devolviendo la pieza con mayor similitud combinada (0.7 peso Cenital + 0.3 Lateral).
