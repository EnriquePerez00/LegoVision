# Documentación del Pipeline de Inferencia — `camara_domo_monopieza_90`

Este documento describe el proceso de inferencia del módulo **Cámara Domo Monopieza 90°**, orientado a identificar piezas LEGO **sobre una cinta transportadora de color azul petróleo** en un setup monocámara cenital (cámara a 90° sobre la cinta), con cámara lateral/frontal opcional.

---

## 1. Setup y Arquitectura del Sistema

Setup optimizado para clasificación de piezas LEGO en escenario de cinta industrial:

- **Cinta transportadora:** color **azul petróleo (`#006064`)**, mate, roughness=1.0.
- **Cámara Cenital (Principal, 90° pitch):** Ubicada a 30 cm perpendicular a la cinta.
  Focal 55 mm, resolución 2048×2048. Captura vistas ortogonales para medir el área
  cenital aparente y realizar la clasificación primaria (geometría + color).
- **Cámara Frontal / Lateral (Opcional):** Ubicada a 15 cm del borde de caída,
  2 cm sobre la cinta. Focal 27 mm. Captura el perfil de altura para triangulación.
  Su activación está controlada por `cfg.inference.camara_lateral` en `config.yaml`.

---

## 2. Modelos Neuronales en Uso

| Modelo / Tarea | Archivo / Repositorio | Descripción |
| :--- | :--- | :--- |
| **Detección Cenital** | `models/yolo_cenital.pt` | YOLO11 — bbox de la pieza en vista cenital |
| **Detección Frontal** | `models/yolo_lateral.pt` | YOLO11 — bbox de la pieza en vista frontal/lateral |
| **Keypoints Cenital** | `models/yolo_cenital_pose.pt` | YOLO-Pose para keypoints cenitales |
| **Keypoints Frontal**  | `models/yolo_frontal_pose.pt` | YOLO-Pose para triangulación de altura |
| **Segmentación** | `mobile_sam.pt` | MobileSAM para máscara pixel-perfect |
| **Clasificador Geometría** | `EfficientNetV2-B0` (`models/efficientnet_cenital.pt`) | Extrae embeddings de la silueta en escala de grises |
| **Clasificador Color (MLP)** | `models/color_mlp_model_75078.pt`, `color_mlp_model.pt` | MLP sobre estadísticas Lab/HSV, 12 features |
| **Clasificador Color Jerárquico** | `models/color_router_all_colors.pt` | Router por saturación → cascadas por familia de color |

---

## 3. Color de la Cinta (variable de sistema)

⚠️ **Regla arquitectónica:** El color de la cinta transportadora está definido **en un único punto** del sistema:

```python
# scripts/scene_config.py   (fuente única de verdad)
BELT_COLOR_HEX = "#006064"   # Azul petróleo real (petrol blue / teal-petrol)
```

Todas las representaciones (RGB 0-255, linear sRGB para Blender, HSV para chromakey en OpenCV, canvas DINOv2) se **derivan automáticamente** desde esta constante mediante `scripts/scene_config.py`. Nunca se hardcodean valores numéricos del color fuera de este archivo.

Ver documento completo: [`BELT_COLOR_ARCHITECTURE.md`](./BELT_COLOR_ARCHITECTURE.md).

**Verificación:** `python scripts/test_belt_color.py`

---

## 4. Estrategias de Inferencia

### A. Chroma-Keying de la Cinta (crucial para clasificación de color)

Antes de calcular las estadísticas Lab/HSV para el MLP de color, se **filtran los píxeles de la cinta** que pudieron colarse dentro de la máscara SAM (por bordes, oclusiones parciales, sombras cerca del contacto pieza-cinta):

```python
from _belt_mask import filter_out_belt
pixels_rgb, pixels_hsv = filter_out_belt(pixels_rgb, pixels_hsv)
```

- **Rango HSV** derivado dinámicamente de `BELT_COLOR_HEX`:
  - H = 79-103 (± tolerancia sobre H≈91 del azul petróleo)
  - S = 60-255 (excluye grises casi neutros)
  - V = 30-220 (excluye sombras profundas y brillos especulares)
- Este rango **NO cubre** los azules LEGO habituales (Bright Blue H≈107, Dark Blue H≈106, Medium Blue H≈107), verificado en `test_belt_color.py`.

Todos los scripts de evaluación (`run_evaluation.py`, `run_evaluation_75078.py`, `run_evaluation_all.py`, `train_and_evaluate_color_mlp.py`, `test_color_*`) usan **exclusivamente** esta utilidad canónica: no hay rangos HSV literales en el código de inferencia.

### B. Clasificación Cromática (MLP de Color)

Pipeline determinista + estadístico para evitar sesgo forma→color:

1. **Segmentación y Erosión:** Máscara MobileSAM → erosión morfológica 5×5 (elimina sangrado de fondo).
2. **Aplicación de CCM (opcional):** Si `data/ccm_dome_light.json` existe, se aplica corrección de matriz de calibración cromática inversa (compensación por iluminación del dome).
3. **Chroma-Keying de cinta:** filtro HSV derivado dinámicamente (sección A).
4. **Filtro de Especularidades:** S ≥ 25 o V < 230 (descarta brillos).
5. **Percentiles L (Lab):** conservar píxeles con L ∈ [P25, P75] (descarta sombras y brillos residuales).
6. **Feature vector (12D):** media y std de L, a, b, H, S, V.
7. **MLP inference:** Normalización con scaler → predicción de color.
8. **Prior bayesiano opcional:** Si el set asociado es conocido (p.ej. 75078-1), se restringe la distribución posterior a las clases presentes en el set.
9. **Multi-frame aggregation:** Cuando hay tracking, las probabilidades por frame se ponderan por confianza máxima y se acumulan antes del argmax.
10. **Fallback:** Si el MLP no está disponible, se cae a búsqueda del color más cercano en la paleta por Delta E CIEDE2000.

### C. Clasificación de Pieza (EfficientNetV2 + Gating por Área)

1. **Gating Determinista (Fase 1):**
   - Área cenital aparente en mm² a partir de la máscara SAM (compensa distancia focal y altura z_eff).
   - Consulta a base de datos (Supabase o caché) para filtrar candidatas cuya superficie nominal esté dentro de ±35 %.

2. **Extracción de Embeddings (Fase 2):**
   - Recorte cenital + frontal (si aplica) con **fondo enmascarado en negro** y convertido a **escala de grises** → aísla la clasificación de la silueta, no del color.

3. **Matching KNN Restringido (Fase 3):**
   - Similitud coseno contra los embeddings pre-indexados **solo de las piezas candidatas** del Gating.
   - Fusión ponderada: `0.7 × Cenital + 0.3 × Lateral`.

### D. Medición Lateral y Triangulación de Altura

Cuando `camara_lateral: true`:
- **Segmentación Lateral:** Proyección de bbox cenital → alimenta MobileSAM lateral.
- **Triangulación por Keypoints:** Si ambos YOLO-Pose (`yolo_cenital_pose.pt` + `yolo_frontal_pose.pt`) devuelven keypoints con confianza ≥ 0.20 → correspondencia epipolar (`_kpts_observer.py`) → altura en mm.
- **Fallback SAM:** Si la triangulación falla, se mide altura del bbox lateral escalada por `px_per_mm_lateral`.

---

## 5. Referencias DINOv2 y Canvas Background

Las referencias visuales indexadas por DINOv2 se generan con un canvas cuyo color de fondo coincide con la cinta:

```python
# scripts/scene_config.py
DINO_BG_COLOR = BELT_COLOR_RGB_255   # (0, 96, 100)
```

Esto asegura que las imágenes de referencia y las imágenes de consulta comparten el mismo dominio de fondo, mejorando la similitud coseno en el espacio DINOv2.

---

## 6. Cambios de Configuración

Para modificar cualquier parámetro del pipeline:

| Qué quieres cambiar | Fichero |
| :--- | :--- |
| Color de la cinta | `scripts/scene_config.py` línea `BELT_COLOR_HEX` |
| Tolerancia del chromakey HSV | `_belt_mask.py` → parámetros de `filter_out_belt()` |
| Activar cámara lateral | `projects/camara_domo_monopieza_90/config.yaml` → `inference.camara_lateral: true` |
| CCM (calibración cromática) | `data/ccm_dome_light.json` |
| Paleta de colores | `data/color_calibration_palette.json` |
| Clasificador de color por set | `--color-classifier` argumento en `run_evaluation.py` |

---

## 7. Ejecución Rápida

### Simulación + evaluación (set 75078-1)
```bash
cd projects/camara_domo_monopieza_90
python scripts/run_simulation_100_75078.py
```

### Solo evaluación (metadata ya existe)
```bash
python scripts/run_evaluation_75078.py \
    --metadata data/simulation_100_75078/simulation_metadata.json \
    --report data/simulation_100_75078/inferencia_consolidada.json \
    --color-classifier 75078-1
```

### Test de integridad del color
```bash
python scripts/test_belt_color.py
```

---

## 8. Historial de Cambios Relevantes

| Fecha | Cambio |
| :--- | :--- |
| 2026-03-07 | Migrado color de cinta `#254154` → `#006064` (azul petróleo real). Fuente única de verdad en `scripts/scene_config.BELT_COLOR_HEX`. Refactor global del chromakey a `_belt_mask.py`. Test `test_belt_color.py`. Regenerados 100 renders del set 75078 con el nuevo color. |
---

## 9. ColorClassifierV2 — Arquitectura 4-Stage (2026-06-07)

### Mejoras sobre el clasificador anterior (`all_colors`, 4.2% accuracy)

| Stage | Descripción | Impacto |
| :--- | :--- | :--- |
| **Stage 0** | Pre-check determinista de material (TRANSPARENT/METALLIC/WHITE/BLACK) | Evita confusión Trans-Clear → White |
| **Stage 1** | CIELAB Match directo contra paleta calibrada con CIEDE2000 ponderado | Elimina MLP Router (fuente principal de errores) |
| **Stage 2** | MLP ligero 6D→5 clases de material (solo si ΔE top-1 > 8) | Resuelve ambigüedad alta |
| **Stage 3** | Resolución determinista de homónimos por mapa canónico | Elimina colisiones Dark Green BL7/80, etc. |

**Accuracy proyectada:** ~50% color (vs 4.2% baseline).

### Uso en evaluación

```bash
cd projects/camara_domo_monopieza_90
python scripts/run_evaluation_1D_all.py \
    --metadata data/simulation_x5_1D_all/simulation_metadata.json \
    --color-classifier v2 \
    --report reports/eval_colorv2.json
```

### Fix F0.1: Chromakey turquesas

`_belt_mask.py` actualizado con `s_bounds=(200, 255)` (era `(60, 255)`):
- **Cinta #006064**: S=255 → **FILTRADA** ✓
- **Dark Turquoise #00828E**: S≈243 en EEVEE, 4/7 pixels preservados ✓
- **Light Turquoise #54A4AE**: S=71 → **NO filtrada** ✓

**Test de integridad:** `python scripts/test_belt_color.py` (5/5 tests pasan)

