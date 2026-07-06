# Arquitectura del Color de la Cinta Transportadora

## 🎨 Regla del sistema

**El color de la cinta transportadora se define en UN ÚNICO PUNTO** del proyecto:

```python
# scripts/scene_config.py
BELT_COLOR_HEX = "#006064"   # Azul petróleo (petrol blue / teal-petrol)
```

Todos los demás valores (RGB 0-255, linear sRGB para Blender, HSV para
chromakey en OpenCV) **se derivan automáticamente** de esta constante.
Cambiar el color = editar SOLO esta línea.

## 📍 Fuente única: `scripts/scene_config.py`

```python
BELT_COLOR_HEX     = "#006064"                                 # ← editable
BELT_COLOR_RGB_255 = (0, 96, 100)                              # derivado
BELT_COLOR_LINEAR  = (0.0, 0.117, 0.127, 1.0)                  # derivado (Blender)
BELT_COLOR_HSV_OCV = (91, 255, 100)                            # derivado (OpenCV)
DINO_BG_COLOR      = BELT_COLOR_RGB_255                        # derivado

def belt_hsv_range(h_tol=12, s_bounds=(60, 255), v_bounds=(30, 220)):
    """Rango HSV derivado dinámicamente para chromakey."""
    ...
```

## 🧭 Flujo de propagación

```
scripts/scene_config.BELT_COLOR_HEX
│
├─► BELT_COLOR_LINEAR ─► scene_canonical.py ─► Blender/EEVEE
│                        └► Material "Belt_Blue_Petroleum"
│
├─► BELT_COLOR_RGB_255 ─► generate_piece_report.py (placeholders)
│                         DINO_BG_COLOR ──────► DINOv2 canvas
│
└─► belt_hsv_range()   ─► _belt_mask.py (utilidad canónica)
                          │
                          └─► filter_out_belt() usado por:
                              ├─ run_evaluation.py
                              ├─ run_evaluation_75078.py
                              ├─ run_evaluation_all.py
                              ├─ train_and_evaluate_color_mlp.py
                              ├─ test_pure_cielab_cascades.py
                              ├─ test_color_all_cascades.py
                              ├─ test_color_optimizations.py
                              └─ diag_dist_key.py
```

## ⚠️ Restricciones (verificadas por `test_belt_color.py`)

- **NO** se permite hardcodear el valor numérico (hex, RGB, linear, HSV) del
  color de la cinta fuera de `scripts/scene_config.py`.
- **NO** se permite duplicar el color en los YAML de configuración
  (`config.yaml`, `config_75078.yaml`, `config_all.yaml`).
- **NO** se permite escribir rangos HSV literales para el chromakey en scripts
  de inferencia — usar `_belt_mask.filter_out_belt()`.

## 🧪 Test de integridad

Ejecutar en cualquier momento:

```bash
python projects/camara_domo_monopieza_90/scripts/test_belt_color.py
```

Comprueba:
1. Coherencia de las derivaciones (hex ⇄ rgb ⇄ linear ⇄ hsv).
2. El color de la cinta cae en su propio rango HSV de chromakey.
3. Los colores LEGO azules (Bright/Dark/Medium Blue) NO caen en el rango,
   así que no serán descartados durante la clasificación de color.
4. Ningún archivo `.py` del proyecto contiene el color hardcoded.

## 🔄 Procedimiento para cambiar el color

1. Editar `scripts/scene_config.py` línea `BELT_COLOR_HEX = "..."`.
2. Ejecutar `python projects/camara_domo_monopieza_90/scripts/test_belt_color.py`.
3. Si el test pasa, regenerar los renders (los datasets sintéticos y renders
   de referencia usan el color como fondo):
   ```bash
   cd projects/camara_domo_monopieza_90
   python scripts/run_simulation_100_75078.py       # 100 renders del set 75078
   ```
4. Considerar re-entrenar los modelos YOLO/EfficientNet/MLP-color cuyo
   dataset de entrenamiento tenga el color de cinta como fondo, ya que
   podrían haber aprendido features ligadas al color anterior.

## 📜 Historial

| Fecha | Cambio |
|-------|--------|
| 2026-03-07 | Migrado de `#254154` (azul marino oscuro) → `#006064` (azul petróleo real). Refactor completo para eliminar duplicaciones y hardcoding. Fuente única = `scripts/scene_config.BELT_COLOR_HEX`. |