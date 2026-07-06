# CLAUDE.md — Guía maestra para agentes IA en LegoVision

> Este archivo es la fuente única de verdad para cualquier agente
> (Claude Code, Cline, Copilot, Cursor, etc.) que opere sobre este
> repositorio. **Se sube a git** y es de lectura obligatoria antes
> de cualquier modificación.
>
> Complementa (no sustituye) a `.clinerules` (local, específico del
> flujo de Cline) y a los `SETUP_DOC.md` de cada subproyecto.

---

## 1. Propósito del repositorio

LegoVision es una plataforma de **inferencia de piezas LEGO** sobre
imágenes sintéticas (Blender EEVEE) y —en el futuro— reales.

Se desarrollan varios pipelines paralelos, cada uno con distinta
disposición de piezas y cámaras. El objetivo cuantitativo es común
para todos:

| KPI | Target |
|---|---|
| **Accuracy top-1 por pieza** | ≥ **98 %** |
| **Latencia inferencia end-to-end** | mínimo posible en M4 (< 500 ms/pieza objetivo) |
| **Cobertura** | piezas del set 75078-1 (`monopieza`) → BD completa (~700 piezas) |

Cualquier PR / iteración debe justificarse contra estos dos KPIs.

---

## 2. Arquitectura de proyectos

El repo hospeda varios pipelines vivos en `projects/`. Compárten
`core/`, `database/`, `scripts/` y `docs/`, y cada uno se aísla en
su subdirectorio con `config.yaml`, `scripts/`, `models/`, `data/`,
`logs/`, `reports/`.

| Proyecto | Descripción | Distribución piezas | Estado |
|---|---|---|---|
| `camara_domo_monopieza_90/` | Cámara cenital 90° + frontal opcional sobre cinta. MLP+cascadas de color, EfficientNet-B0 geometría, YOLO11-pose. | **1D continua** (piezas en fila sobre cinta) y **2D** | **ACTIVO PRINCIPAL** |
| `2camaras_random_pieza_unica/` | Cenital + lateral, 1 pieza por frame, poses aleatorias. Pipeline DINOv2 canónico. | 1D (aleatoria) | Iteración v4 canónica |
| `camara_domo/` | Iteración previa del monopieza (dome light). Legacy pero referencia. | 1D/2D | Congelado |
| `2camaras_multi_pieza/` | Multi-pieza sobre cinta con YOLO detector + reidentificación. | 2D | Early stage |

**Regla de foco:** salvo indicación explícita del usuario, todo
cambio se aplica primero a `camara_domo_monopieza_90/`.

---

## 3. Pipeline de inferencia canónico (cascada)

```
imagen cenital ──► YOLO detección bbox
                 └─► MobileSAM segmentación pixel-perfect
                     └─► crop + mask
                         ├─► Color pipeline: Lab/HSV stats → MLP → familia → cascada CIELAB → color final
                         ├─► Geometría: EfficientNetV2-B0 embedding → k-NN sobre refs → shape shortlist
                         ├─► Dims aparentes (área silueta, bbox mm) → filtro por altura efectiva DB
                         └─► DINOv2 fine-tuned (metric head) → similaridad cosenos → ranking final
imagen frontal (opcional) ──► YOLO-pose → keypoints → triangulación altura real
                                                    → refinamiento del ranking

Fusión bayesiana / votación ponderada → predicción final (part_id, color_id)
```

**Regla:** el orden y los pesos de la cascada son parámetros de
`config.yaml`, nunca hardcoded. Cualquier reorden requiere
actualizar `INFERENCE_DOCUMENTATION.md` del proyecto.

---

## 4. Hardware target y reglas de optimización

**Máquina de desarrollo:** MacBook Pro M4 (Apple Silicon)
- 12 CPUs (10 performance + 2 efficiency)
- 48 GB RAM unificada
- GPU integrada Metal (10-core, ~2.5 TFLOPS FP16)
- **Sin CUDA** — jamás asumir CUDA en local.

### 4.1 Reglas obligatorias en cualquier script Python nuevo

1. **Device detection helper** — todo script que use PyTorch:
   ```python
   def get_device() -> torch.device:
       if torch.backends.mps.is_available():
           return torch.device("mps")
       if torch.cuda.is_available():
           return torch.device("cuda")
       return torch.device("cpu")
   ```

2. **Threads CPU** — al principio de `main()`:
   ```python
   import os, torch
   n = int(os.environ.get("TORCH_NUM_THREADS", 10))
   torch.set_num_threads(n)
   torch.set_num_interop_threads(2)
   ```
   Se reservan 2 CPUs para OS/UI (10 workers efectivos).

3. **Env vars recomendadas** (ver `scripts/setup_env.sh`):
   ```bash
   export OMP_NUM_THREADS=10
   export MKL_NUM_THREADS=10
   export VECLIB_MAXIMUM_THREADS=10
   export NUMEXPR_NUM_THREADS=10
   export PYTORCH_ENABLE_MPS_FALLBACK=1   # ops no soportadas → CPU
   export TORCH_NUM_THREADS=10
   ```

4. **Fallback MPS explícito**: siempre `PYTORCH_ENABLE_MPS_FALLBACK=1`
   porque ops como `aten::_upsample_bicubic2d_aa` aún no están en MPS.

5. **Precision**: en inferencia MPS usar **`float16`** salvo
   operaciones numéricamente sensibles (Lab conversion, distancias).
   ```python
   model.half().to("mps")
   x = x.half().to("mps")
   ```

6. **Batching**: nunca inferir imagen a imagen si hay >1 disponible.
   Batch sizes recomendados M4:
   | Modelo | Batch inferencia |
   |---|---|
   | YOLO11n/s | 32 |
   | MobileSAM | 8 (memory-heavy) |
   | EfficientNetV2-B0 (224×224) | 64 |
   | DINOv2 ViT-B/14 (518×518) | 16 |
   | DINOv2 ViT-S/14 (224×224) | 32 |

7. **DataLoader**:
   ```python
   DataLoader(ds,
       batch_size=..., num_workers=8,
       persistent_workers=True,
       pin_memory=False,        # MPS no lo aprovecha
       prefetch_factor=2)
   ```

8. **`torch.compile`** en modelos fijos de inferencia:
   ```python
   model = torch.compile(model, mode="reduce-overhead", dynamic=False)
   ```
   (con guard: `if hasattr(torch, "compile") and device.type != "mps"` — MPS
   soporta parcialmente compile; benchmarkear antes de habilitar en producción).

9. **CoreML para producción**: los modelos ganadores YOLO exportan a
   `.mlpackage` (usar `yolo export format=coreml half=True`).
   Speedup típico M4: **2-3×** vs PyTorch/MPS.
   Los `.mlpackage` NO se suben a git (regenerables).

10. **Cache de embeddings**: DINOv2 y EfficientNet refs de galería se
    cachean **una vez** en `models/<x>_ref_embeddings.npz`. Prohibido
    recomputar en cada evaluación.

### 4.2 Reglas para generación de datasets (Blender)

1. **Workers Blender** = `max(1, os.cpu_count() - 2)` = **10 workers**.
   Cada worker consume ~2-3 GB RAM → total ~25-30 GB → cabe en 48 GB.

2. Blender EEVEE, no Cycles (10× más rápido en M4).

3. `bpy.context.scene.render.threads_mode = 'FIXED'` y
   `threads = 1` (un thread por worker; el paralelismo va por procesos).

4. Nunca renderizar en el proceso principal si hay ≥2 imágenes:
   siempre orquestar con `run_*_parallel.py`.

### 4.3 Reglas de entrenamiento

- **YOLO**: `ultralytics` con `device=0` en MPS o `device='mps'`
  explícito. Usar `batch=-1` (auto) y `workers=8`.
- **EfficientNet / MLP / DINOv2 head**: `torch.optim.AdamW`,
  `torch.cuda.amp.autocast` **NO** (MPS no soporta amp completo);
  usar `torch.autocast("mps", dtype=torch.float16)` **solo tras
  benchmark**. Por defecto `float32` en training para estabilidad.
- **Fine-tuning masivo (>1h)**: considerar Kaggle/Colab GPU
  (ver `scripts/remote_manager.py` y `training/remote_manager.py`).

---

## 5. Convenciones de código y layout

### 5.1 Estructura por proyecto

```
projects/<nombre>/
├── config.yaml                # única fuente de verdad de parámetros
├── SETUP_DOC.md              # doc humana + changelog
├── INFERENCE_DOCUMENTATION.md # detalle del pipeline
├── requirements.txt          # dependencias específicas (opcional)
├── scripts/
│   ├── scene_config.py       # o scene_canonical.py — geometría escena
│   ├── generate_*.py         # datasets sintéticos Blender
│   ├── train_*.py            # entrenamientos
│   ├── run_*.py              # orquestadores paralelos
│   ├── inferencia_neuronal.py  # pipeline completo de inferencia
│   ├── evaluate_*.py         # evaluación sobre test sets
│   └── analyze_*.py          # análisis post-hoc
├── models/                   # SOLO modelos ganadores + MANIFEST.md
├── data/                     # regenerable, IGNORADO por git
├── logs/                     # IGNORADO
└── reports/                  # IGNORADO
```

### 5.2 Nombres de scripts

| Prefijo | Uso |
|---|---|
| `generate_*` | Genera datasets (normalmente Blender) |
| `train_*` | Entrenamiento de un modelo |
| `run_*` | Orquestador multi-proceso |
| `evaluate_*` | Evaluación batch sobre test set |
| `analyze_*` | Análisis exploratorio de resultados |
| `test_*` | Tests unitarios/regresión |
| `_*` (underscore) | Helper interno, no ejecutable directamente |

### 5.3 Config.yaml

- YAML con comentarios inline cuando el parámetro no sea obvio.
- Secciones por dominio: `scene`, `cameras`, `render`, `pieces`,
  `yolo`, `dinov2`, `color`, `inference`, `evaluation`.
- Tipos consistentes; no mezclar `str` con `int` para lo mismo.
- Un parámetro **debe** estar usado por al menos un script (nunca huérfanos).
- Cambios en `config.yaml` disparan actualización en `SETUP_DOC.md` (regla 1 de `.clinerules`).

### 5.4 Prohibiciones estrictas

- **Nunca** subir a git: imágenes, `.npy`, `.npz` (salvo whitelist), `.pt` no whitelisted, `.log`, `.html` generados, `.xlsx`, `.csv` (salvo `database/colors.csv`).
- **Nunca** `git add -A` sin haber revisado `git status` primero.
- **Nunca** commitear scripts scratch en la raíz — usar `scratch/` (ignorado) o mover a `projects/*/scripts/`.
- **Nunca** introducir código CUDA-specific sin fallback MPS/CPU explícito.
- **Nunca** deducir esquema de BD leyendo `/database/` — usar servidor MCP PostgreSQL (`docs/mcp_postgres_setup.md`).

---

## 6. Comandos canónicos

### 6.1 Bootstrap del entorno

```bash
source .venv/bin/activate
source scripts/setup_env.sh          # exporta TORCH_NUM_THREADS, MPS flags, etc.
```

### 6.2 Generar datasets (proyecto activo)

```bash
# YOLO training set (10 workers Blender paralelos)
python projects/camara_domo_monopieza_90/scripts/run_yolo_dataset_parallel.py

# Simulación 1000 piezas
python projects/camara_domo_monopieza_90/scripts/run_data1000_parallel.py

# Sim 100 piezas del set 75078
python projects/camara_domo_monopieza_90/scripts/run_simulation_100_75078.py
```

### 6.3 Entrenar

```bash
python projects/camara_domo_monopieza_90/scripts/train_yolo_pose.py \
    --epochs 200 --batch -1 --device mps
python projects/camara_domo_monopieza_90/scripts/train_hierarchical_color.py
python projects/camara_domo_monopieza_90/scripts/train_efficientnet_head.py
```

### 6.4 Evaluar

```bash
python projects/camara_domo_monopieza_90/scripts/run_evaluation_1D_all.py
python projects/camara_domo_monopieza_90/scripts/evaluate_sim100.py
python projects/camara_domo_monopieza_90/scripts/evaluate_sim300.py
```

### 6.5 Exportar a CoreML (producción M4)

```bash
yolo export model=projects/camara_domo_monopieza_90/models/yolo_cenital.pt \
    format=coreml half=True imgsz=1024
```

### 6.6 Benchmark de latencia

```bash
python projects/camara_domo_monopieza_90/scripts/inferencia_neuronal.py \
    --benchmark --n-warmup 20 --n-iter 100
```

---

## 7. Base de datos

- Motor: **PostgreSQL 15** en Supabase (schema `public`).
- Migraciones versionadas en `core/db/migrations/` (`00X_*.sql`).
- **Acceso desde agentes**: usar SIEMPRE servidor **MCP PostgreSQL** configurado en Cline (ver `docs/mcp_postgres_setup.md`).
- **Prohibido**: deducir esquema del código; usar `psql` desde scripts one-off; hardcodear conexiones.
- Tablas clave: `parts`, `stable_poses`, `pose_areas`, `contact_stable_dims`, `minifig_assemblies`, `colors`, `topological_features`.

---

## 8. Playbook de optimización

### Cuando el usuario pide **"mejorar accuracy"**

1. **Baseline**: correr `evaluate_*.py` reciente; accuracy por pieza y por color en test canónico.
2. **Confusión**: matriz de confusión → top-5 pares confundidos.
3. **Dominio del error**:
   - Error de **color** → cascada CIELAB / MLP router / palette.
   - Error de **shape** → EfficientNet head / DINOv2 metric head / refs de galería.
   - Error de **dims** → `contact_stable_dims` en BD y silueta cenital.
4. **Refs de galería** actualizadas con la escena canónica actual.
5. **Data augmentation**: ¿training set cubre poses estables reales? Ver `simulate_stable_poses.py`.
6. **Threshold tuning**: `tune.py` (Optuna) sobre pesos de la cascada.

### Cuando el usuario pide **"reducir latencia"**

1. **Profile primero** (`torch.profiler` o `cProfile`). Nunca optimizar a ciegas.
2. **Batching** (§ 4.1.6).
3. **Export a CoreML** (§ 4.1.9).
4. **Reducir resolución** de entrada si accuracy no cae (probar 640 → 512 → 384).
5. **Reducir refs DINOv2** por pieza (de 12 a top-3 más informativas).
6. **`torch.compile`** (§ 4.1.8).
7. **Fusión de operaciones** en el post-procesado.
8. **Eliminar recomputación** — cachear todo lo que no cambie entre frames.

### Presupuesto de latencia por etapa (target M4)

| Etapa | Target |
|---|---|
| YOLO det + pose | 30 ms |
| MobileSAM | 60 ms |
| Color pipeline | 20 ms |
| EfficientNet embed | 40 ms |
| DINOv2 embed + kNN | 100 ms |
| Fusión + DB lookup | 10 ms |
| **Total end-to-end** | **≤ 260 ms** (holgura para 500 ms) |

Ver `docs/latency_budget.md` para el detalle vivo.

---

## 9. Modelos ganadores (whitelist git)

Cada `projects/<name>/models/` DEBE tener un `MANIFEST.md` con los pesos versionados en git y sus métricas. Sólo los listados en el manifest se suben; el resto se ignora vía `.gitignore`.

Formato mínimo:

```markdown
# Models — <project> / <iteración>

| Archivo | Rol | Arquitectura | Entrenado en | Métrica clave |
|---|---|---|---|---|
| yolo_cenital.pt | detección cenital | YOLO11n | dataset_v9 (12k) | mAP50 0.982 |
| efficientnet_cenital.pt | embed geometría | EffNetV2-B0 | data1000 | top-1 96.4 % |
| color_mlp_model.pt | color primary | MLP 12-32-32-N | palette_v4 | acc 99.1 % |
```

Los modelos base descargables (`mobile_sam.pt`, `yolo11s-pose.pt`, DINOv2 ViT-B raw) **NUNCA** van en el manifest — se descargan on-demand desde su hub.

---

## 10. Referencias

- `README.md` — visión general del repo
- `docs/SYSTEM_DOCUMENTATION.md` — arquitectura macro
- `docs/hardware_setup.md` — setup físico cinta/cámara
- `docs/architecture_and_workflow.md` — flujo end-to-end
- `docs/latency_budget.md` — presupuesto de latencia vivo
- `docs/training_guide.md` — guía de entrenamiento
- `projects/<X>/SETUP_DOC.md` — setup por proyecto
- `projects/<X>/INFERENCE_DOCUMENTATION.md` — pipeline por proyecto
- `core/utils/hw.py` — helpers HW-aware (`get_device`, `set_threads`)
- `.clinerules` — reglas específicas de Cline (local, no en git)

---

## 11. Changelog

| Fecha | Cambio |
|---|---|
| 2026-07-06 | Creación inicial (v1.0) tras auditoría de repo y análisis de HW target M4. |
