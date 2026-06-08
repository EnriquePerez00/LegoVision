# TARPS · Tipping-Aware Random Pose Selection

> Regla de selección de pose estable para los renders de entrenamiento YOLO
> (cenital y lateral) en el pipeline `2camaras_pieza_unica`.

## 1. Definición formal

Para cada pieza LEGO con poses estables simuladas y guardadas en la tabla
`stable_poses` de la BD (filtradas por `is_stable = TRUE`), elegimos una
única pose para cada frame de render aplicando el siguiente algoritmo:

```text
function SELECT_POSE_TARPS(part_ref):
    poses ← todas las poses con is_stable = TRUE de la BD para part_ref
    if poses está vacío:
        return None  (la pieza se omite del frame)

    candidates ← { p en poses : p.tipping_energy_ratio ≥ 0.04 }

    if candidates no vacío:
        return random.choice(candidates)            # rama principal
    else:
        return argmax_{p en poses}( p.tipping_energy_ratio )   # fallback determinista
```

- **Umbral** `TARPS_MIN_TIPPING = 0.04`. Configurable en
  `2camaras_pieza_unica/config.yaml` bajo `stable_poses.tarps_min_tipping`.
- **Métrica** `tipping_energy_ratio` (campo `tipping_energy_ratio` de
  `stable_poses`): adimensional, determinista, depende sólo de la geometría
  del mesh LDraw y de `orientation_quat`. Definida como
  `(√(margin² + h_com²) − h_com) / h_com`, donde `margin` es la distancia
  mínima del CdM proyectado al borde del polígono de soporte y `h_com` es
  la altura del centro de masa.

## 2. Justificación

### 2.1 Por qué `tipping_energy_ratio` y no `stability_ratio`

`stability_ratio` proviene de la simulación física Bullet (impulsos sobre
la cinta) y es **estocástico**: el mismo escenario puede dar valores
distintos en re-simulaciones, y no es comparable entre piezas porque
depende del número de caras candidatas que el detector haya producido.

`tipping_energy_ratio` es **determinista** y **comparable**: depende sólo
de la geometría LDraw y del cuaternión de la pose. Tras analizar las 35
piezas del set 75078-1 contra el criterio humano del usuario (Excel
`poses_estables_75078-1 (filtrada).xlsx`), la métrica geométrica resulta
ser **mucho mejor predictora** que la estocástica.

### 2.2 Cobertura sobre el set 75078-1

Las 35 piezas regulares se reparten en tres grupos según cuán bien el
criterio único `tipping ≥ 0.04` reproduce el conteo del usuario:

| Grupo | Piezas | Aciertos exactos del criterio | Comentario |
|-------|-------:|------------------------------:|------------|
| **A — separación clara** | 24 | **24 / 24 (100 %)** | Caras planas bien definidas; salto de un orden de magnitud entre `tipping` de poses estables vs. inestables. Ejemplos: `2877`, `3004`, `3022`, `2412b`, `60481`. |
| **B — frontera** | 7 | **3 / 7 (43 %)** ± 1 pose | Casos límite: poses con `tipping` cerca del umbral 0.04 (ej. `2654 → tipping = 0.0`, `61184 → 0.029`, `87620 → 0.04`) o discrepancias subjetivas (`6541 → +1`). |
| **C — patológico** | 4 | **4 / 4 (100 %)** ⚠️ casualmente | Piezas con simetría rotacional (`32054`, `61184`-like) o muchas caras candidatas equivalentes (`15391`, `15392`). El criterio acierta el conteo pero no por una razón geométricamente sólida. |

Total: **31 / 35 piezas con conteo exacto** (Σ\|diff\| = 4 sobre el total
de poses esperadas). Es el techo encontrado tras evaluar 13 criterios
distintos (`stability_ratio` solo, `margin` solo, combinaciones, `srn`,
etc.).

### 2.3 Por qué el fallback `argmax(tipping)`

Al aplicar TARPS sobre el set 75078-1 ocurre lo siguiente:

| Caso | Piezas | Comportamiento |
|------|-------:|----------------|
| `len(candidates) ≥ 1` | **34 / 35** | Rama principal: pose elegida al azar entre las que cumplen `tipping ≥ 0.04`. |
| `len(candidates) = 0` | **1 / 35** (`61184`) | Fallback determinista: la pieza tiene 9 poses estables todas con `tipping ∈ [0.016, 0.029]`. Elegimos la pose con `tipping = 0.029` (la lateral del rod, que es además la única pose físicamente sensata para un rod 7L con esferas en los extremos). |

Sin el fallback, `61184` no se podría renderizar nunca. Con el fallback,
TARPS **garantiza cobertura del 100 % de las piezas** y, dentro de cada
pieza, **garantiza tipping ≥ 0.04 siempre que sea posible**.

## 3. Comparativa con la regla anterior

La función `get_stable_poses()` de `generate_yolo_training_dataset.py`
hasta esta migración aplicaba:

```python
stable = [p for p in all_poses if p["stability_ratio"] ≥ MIN_STABILITY]   # 0.01
filtered = [p for p in stable if min(contact_w, contact_l) ≥ 4 mm]
if not filtered:
    filtered = [p for p in all_poses if face_class in ("Top", "Bottom")]
pose = random.choice(filtered or all_poses)
```

Problemas resueltos por TARPS:

1. **`stability_ratio` no comparable entre piezas**: en piezas re-simuladas
   antes de la migración 008/009, todas las poses tenían
   `stability_ratio = 1.0` (Grupo B histórico), o reparticiones que dependían
   del número de caras candidatas. TARPS usa una métrica geométrica
   determinista.

2. **El filtro `min(contact_w, contact_l) ≥ 4 mm` falla en piezas de
   pequeña base** (`14769`, `61184`, `54200` quedaban con 0 poses tras el
   filtro y caían a un fallback poco semántico).

3. **El fallback "Top/Bottom"** elegía caras invariantes a la geometría
   real de la pieza (un rod horizontal no tiene "Top" físico estable).

TARPS sustituye todo esto por una sola regla con un único umbral, basada
en la métrica determinista que mejor correlaciona con el criterio humano
de “estable en la cinta”.

## 4. Implementación

### 4.1 Cache JSON
El cache `data/stable_poses_cache.json` debe contener, para cada pose:

```json
{
  "pose_index": 0,
  "contact_normal": [0,1,0],
  "face_class": "Top",
  "contact_area": 760.0,
  "orientation_quat": [...],
  "orientation_euler": [...],
  "stability_ratio": 0.45,
  "stability_ratio_normalized": 1.0,
  "tipping_energy_ratio": 0.32,
  "support_polygon_margin_mm": 4.43,
  "contact_stable_width": 4.0,
  "contact_stable_length": 16.0,
  "lateral_height": 8.0,
  "zenith_observable_area": 200.0
}
```

El script `2camaras_pieza_unica/scripts/sync_stable_poses_cache.py` ahora
expone los tres campos nuevos (`tipping_energy_ratio`,
`support_polygon_margin_mm`, `stability_ratio_normalized`) y guarda
todas las poses con `is_stable = TRUE` sin pre-filtrar.

### 4.2 Función `select_pose_tarps`
Implementada en
`2camaras_pieza_unica/scripts/generate_yolo_training_dataset.py`. Es la
única consumidora de la regla. Para los renders de DINOv2 (refs
multi-pose) **se conserva el comportamiento actual** de iterar todas las
poses estables, porque el catálogo de embeddings necesita ser exhaustivo
(no se elige una sola pose por pieza).

### 4.3 Configuración
En `2camaras_pieza_unica/config.yaml`:

```yaml
stable_poses:
  cache_path: data/stable_poses_cache.json
  # Tipping-Aware Random Pose Selection (TARPS) - ver docs/stable_pose_selection_rule.md
  tarps_min_tipping: 0.04
  # Campos legacy (mantenidos para compatibilidad con scripts antiguos):
  render_min_stability: 0.01
  min_contact_dimension_mm: 4.0
```

## 5. Limitaciones conocidas

1. **Piezas con simetría rotacional continua** (cilindros, conos como
   `4073`): el simulador genera N poses discretas alrededor del eje de
   revolución. TARPS elige una al azar, lo cual produce renders
   visualmente distintos pero todos válidos. No requiere tratamiento
   especial.

2. **Piezas sin LDraw mesh** (ej. `4589b` para set 75078-1): no aparecen
   en `stable_poses` porque la simulación falla. TARPS devuelve `None` y
   el frame se descarta. Estas piezas **no son renderizables** hasta que
   se incorpore su mesh.

3. **Umbral global vs. piezas pequeñas**: piezas muy pequeñas (volumen <
   2 LDU³) pueden tener `tipping` < 0.04 en todas sus poses por simple
   escala. El fallback `argmax(tipping)` cubre estos casos sin
   recalibrar el umbral.

## 6. Validación

Tras la implementación, ejecutar el dry-run:

```bash
SUPABASE_DB_PASSWORD=legvision_pass_2024 \
  /Users/.../.venv/bin/python \
  2camaras_pieza_unica/scripts/sync_stable_poses_cache.py
```

Y luego validar manualmente que para `61184` la pose elegida es siempre
la del fallback (la única con `tipping = 0.029`), mientras que para
`3004`, `2877`, `61780` etc. se selecciona aleatoriamente entre las 5–6
caras estables.