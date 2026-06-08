# LegoVision - Documentacion Tecnica del Sistema

> **Version:** 2.0 (post-refactoring)
> **Fecha:** Febrero 2026
> **Audiencia:** Desarrolladores de software

---

## Tabla de Contenidos

1. [Vision General](#1-vision-general)
2. [Arquitectura de Componentes](#2-arquitectura-de-componentes)
3. [Flujos de Trabajo Principales](#3-flujos-de-trabajo-principales)
4. [Modelo de Datos y Base de Datos](#4-modelo-de-datos-y-base-de-datos)
5. [Algoritmos Clave](#5-algoritmos-clave)
6. [API de Inferencia - Endpoints](#6-api-de-inferencia---endpoints)
7. [Interfaz Grafica - GUI Bridge](#7-interfaz-grafica---gui-bridge)
8. [Scripts y Utilidades](#8-scripts-y-utilidades)
9. [Variables de Entorno](#9-variables-de-entorno)
10. [Puesta en Marcha](#10-puesta-en-marcha)

---

## 1. Vision General

LegoVision es un sistema de **vision artificial en tiempo real** para detectar y clasificar piezas LEGO sobre una cinta transportadora negra. Opera a **5 FPS** con latencia total inferior a **200 ms**.

### Pipeline de 2 Fases

**Fase 1 - Deteccion (YOLO11n)**
- Entrada: frame de camara industrial (2448x2048 px, Sony IMX264)
- Salida: lista de bounding boxes con clase generica (lego_piece / minifigure)

**Fase 2 - Clasificacion (DINOv2 ViT-S/14 + MLP Projection Head + K-NN)**
- Entrada: crop recortado de cada bbox detectada
- Salida: top-3 candidatos con part_ref, score, color detectado, pose

### Hardware de Referencia

| Parametro        | Valor                        |
|------------------|------------------------------|
| Camara           | Sony IMX264 (Global Shutter) |
| Resolucion       | 5 MP (2448 x 2048 px)        |
| Lente            | 12 mm C-mount                |
| Working Distance | 355 mm                       |
| FOV              | 250 mm (cinta 200 mm)        |
| Velocidad cinta  | max. 83.3 mm/s (5 m/min)     |
| Latencia target  | menor de 200 ms (5 FPS)      |
| Plataforma dev   | Apple M4 (MPS acceleration)  |

---

## 2. Arquitectura de Componentes

### Estructura del Proyecto


```
LegoVision/
  inference/
    api.py                  FastAPI - servidor REST puerto 8005
    detector.py             Wrapper YOLO11n - produce bboxes normalizadas
    knn_classifier.py       DINOv2 + MLP + K-NN - identifica part_ref

  gui/
    app.py                  PyWebView + ApiBridge (puente Python a JS)
    static/
      index.html            SPA principal en dark mode
      css/                  Estilos (style.css, detection_cards.css)
      js/                   Logica frontend JS
      models/               Archivos GLB para visor 3D de minifiguras

  database/
    supabase_client.py      Cliente PostgreSQL via psycopg2
    schema.sql              DDL completo: tablas, indices, vistas
    set_catalog.py          Inventarios estaticos de sets LEGO
    color_catalog.json      Catalogo colores LDraw (hex, alpha, material_type)
    migrations/             Migraciones SQL incrementales 001 a 016

  training/
    train_yolo.py               Entrena YOLO11n sobre dataset sintetico
    index_synthetic_renders.py  Indexa embeddings DINOv2 en la BD
    eval_on_set_simulation.py   Evaluacion post-entrenamiento
    dataset.yaml                Config dataset generado dinamicamente

  scripts/
    scene_config.py                     CENTRAL: todos los parametros de escena
    generate_yolo_training_dataset.py   Genera dataset YOLO con Blender
    generate_physics_ref_multiangle.py  Renders multi-angulo para DINOv2
    generate_dino_fov_renders.py        Scatter de piezas en FOV real
    index_dino_fov_crops.py             Indexa crops del scatter FOV
    physics_belt_generator.py           Simulacion fisica por pieza individual
    physics_set_belt_generator.py       Simulacion fisica del set completo
    generate_inference_test_belt.py     Render de test de inferencia
    generate_inference_renders.py       Renders isometricos unitarios
    validate_stable_poses.py            Valida poses estables con fisica
    analyze_stable_poses_ldraw.py       Analisis geometrico LDraw
    generate_validation_excel.py        Reporte Excel comparativo
    render_stable_poses_validation.py   Renders de poses para Excel
    render_minifig_parts.py             Renders de partes de minifiguras
    assemble_minifig.py                 Ensambla minifiguras en GLB
    simulate_stable_poses.py            Simulacion de poses individuales
    generate_synthetic_set.py           Helper de escena Blender compartido
    batch_physics_runner.py             Runner batch de simulacion
    migrate_sets.py                     Migra catalogo de sets a la BD
    ldraw_mesh_parser.py                Parser de archivos .dat LDraw
    project_all_embeddings.py           Proyecta embeddings 384d a 128d

  data/
    raw_dataset/          Dataset YOLO: images/*.png y labels/*.txt
    processed_dataset/    Dataset dividido en train/ y val/
    synthetic_renders/    Renders isometricos: render_PART_HEX.png
    ref_multiangle/       Renders 12 angulos: ref_PART_HEX_rotANG.png
    dino_scatter/         Crops scatter FOV por pieza
    pipeline_ref/         Imagenes de referencia para validacion pipeline
    pipeline_val/         Imagenes de validacion del pipeline
    validation_renders/   Renders de poses para el Excel
    tmp/                  Archivos temporales de simulacion

  models/
    best.pt               Pesos YOLO11n entrenados (produccion)
    dino_metric_head.pt   Pesos MLP projection head (opcional)

  docker-compose.yml      Supabase local: PostgreSQL:5434, PostgREST:5437
  run.sh                  Script de arranque unificado
  requirements.txt        Dependencias Python
  .env                    Variables de entorno (NO versionar en git)
```

### Flujo de Datos en Tiempo Real


```
Camara / Simulador GUI
    |
    | POST /detect  (multipart image)
    v
FastAPI :8005  (inference/api.py)
    |
    |-- LegoDetector.detect()
    |       YOLO11n: imagen --> lista bboxes [xc,yc,w,h] normalizadas 0-1
    |       Retorna JSON inmediato a la GUI para pintar bboxes
    |
    +-- BackgroundTask --> save_detections_batch()
            Inserta detecciones en PostgreSQL de forma asincrona

    (para cada bbox detectada)
    |
    | POST /classify_crop  (bbox + frame completo en base64)
    v
LegoKNNClassifier.classify()  (inference/knn_classifier.py)
    |
    |-- Extrae crop del frame segun coordenadas bbox
    |-- Sustraccion de fondo si existe data/empty_belt.png
    |-- Fit-to-canvas 224x224 con fondo negro #000000
    |-- DINOv2 ViT-S/14 forward pass --> embedding 384d
    |-- MLP Projection Head --> embedding 128d  (si dino_metric_head.pt existe)
    |-- Filtro por color HSV --> candidatos del color detectado
    |-- Filtro por tamano fisico (px / 3.2 = mm)
    |-- Similitud coseno vs piece_embeddings en RAM
    +-- K-NN consensus voting (k=5) --> top-3 resultados con score
```

---

## 3. Flujos de Trabajo Principales

### Flujo A - Arranque (run.sh)

1. Activa .venv
2. Instala dependencias
3. docker compose up -d
4. Libera puerto 8005
5. uvicorn inference.api:app en background
6. Espera API responda (max 90s)
7. python gui/app.py (bloqueante)
8. Al salir: kill API automatico

### Flujo B - Entrenamiento YOLO11

Set 75078-1. PASO1 Blender generate_yolo_training_dataset.py: cache check, fisica 100 frames, Cycles 4s, bboxes YOLO. PASO2 train_yolo.py: split 70/30, yolo11n 35ep rot360 patience12, exporta best.pt.

### Flujo C - Indexacion DINOv2

PASO1 Blender generate_physics_ref_multiangle.py: 12 angulos por pose, fondo #254154. PASO2 index_synthetic_renders.py: embed 384d L2, guarda piece_embeddings.
CRITICO: preprocessing identico en indexacion e inferencia.

### Flujo D - Simulacion FOV

Prereq: stable_poses en BD. Blender scatter -> renders. Python crop+embed+BD.

### Flujo E - Validacion Poses

validate_stable_poses.py: 8 perturbaciones sinusoidales por pieza. stability_ratio=passes/total. Umbral 0.875.

### Flujo F - Excel Validacion

render_stable_poses_validation.py -> analyze_stable_poses_ldraw.py -> generate_validation_excel.py.
Output: data/validation_report_75078-1.xlsx

---

## 4. Modelo de Datos y Base de Datos

PostgreSQL 16 Docker puerto 5434. Cliente psycopg2 RealDictCursor. DDL: schema.sql + migrations/001-016.

### models
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| id | UUID PK | Identificador unico |
| version | TEXT UNIQUE | Etiqueta version |
| model_path | TEXT | Ruta al .pt |
| map50 | FLOAT | mAP@0.5 |
| is_active | BOOLEAN | En produccion |

### inference_sessions
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| id | UUID PK | Identificador unico |
| started_at | TIMESTAMPTZ | Inicio sesion |
| ended_at | TIMESTAMPTZ | Fin (null si activa) |
| total_detections | INTEGER | Conteo al cerrar |
| avg_confidence | FLOAT | Confianza media |
| avg_fps | FLOAT | FPS promedio |
| belt_speed_mm_s | FLOAT | Velocidad cinta mm/s |

### detections
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| id | UUID PK | Identificador unico |
| session_id | UUID FK | CASCADE DELETE |
| detected_at | TIMESTAMPTZ | Timestamp exacto |
| piece_class | TEXT | Clase YOLO como string |
| piece_name | TEXT | lego_piece / minifigure |
| confidence | FLOAT | 0-1 |
| bbox_x_center | FLOAT | Centro X norm 0-1 |
| bbox_y_center | FLOAT | Centro Y norm 0-1 |
| bbox_width | FLOAT | Ancho norm 0-1 |
| bbox_height | FLOAT | Alto norm 0-1 |
| inference_ms | FLOAT | Latencia ms |
Indices: session_id, piece_class, detected_at DESC, confidence DESC.

### piece_embeddings
Base de similitud del K-NN. Cargada en RAM al iniciar.
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| part_ref | TEXT | ID LDraw: 3001,3004,2877 |
| stable_face | INTEGER | 0=isometrico 1=multi-angulo |
| rotation_angle | INTEGER | 0,30,60...330 |
| embedding | DOUBLE PRECISION[] | Vector 384d DINOv2 |
| embedding_projected | DOUBLE PRECISION[] | Vector 128d MLP head |
| color_code | TEXT | 85=LBG, 0=Negro, 4=Rojo |
| color_hex | TEXT | Hex sin simbolo: A0A5A9 |
| pose_index | INTEGER | Indice de pose fisica |
PK: (part_ref, stable_face, rotation_angle). ON CONFLICT DO UPDATE.

### stable_poses
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| id | UUID PK | Identificador |
| part_ref | TEXT | ID LDraw |
| pose_index | INTEGER | 0,1,2... |
| contact_normal | DOUBLE PRECISION[] | Normal del plano de contacto (apunta del centro de la pieza HACIA AFUERA por la cara que toca la cinta) |
| face_class | TEXT | Top / Side / Bottom (semántica LDraw: cara que reposa en el suelo) |
| contact_area | DOUBLE PRECISION | Área de la cara LDraw seleccionada por `analyze_stable_poses_ldraw` (mm²; ≥ MIN_FACE_AREA_LDU2) |
| orientation_quat | DOUBLE PRECISION[] | Cuaternión [w,x,y,z] que orienta la pieza con esa cara hacia abajo |
| orientation_euler | DOUBLE PRECISION[] | Mismo giro como Euler XYZ |
| simulation_passes / simulation_total | INTEGER | Numerador y denominador de `stability_ratio` (perturbaciones aleatorias en la cinta simulada) |
| stability_ratio | DOUBLE PRECISION | passes/total. **El cache JSON exige > 0.05** (≤ 5 % de éxitos = pose metaestable, descartada) |
| energy_barrier_min / com_distance_to_boundary / is_rolling | DOUBLE PRECISION / BOOL | Métricas físicas adicionales del solver |
| is_stable | BOOLEAN | True si la pose pasó la simulación |
| set_id | TEXT | Set en cuyo inventario se considera la pieza |
| zenith_observable_area | DOUBLE PRECISION | mm². Área de la silueta observable desde la cámara cenital ideal: ConvexHull 2D de la proyección de los vértices LDraw sobre el plano ⟂ `contact_normal` |
| zenith_bbox_area | DOUBLE PRECISION | mm². Área del minAreaRect 2D de la silueta cenital (≥ zenith_observable_area) |
| lateral_height | DOUBLE PRECISION | mm. Extensión real de la pieza a lo largo de `contact_normal` (`max(v·n) − min(v·n)`); altura física que el carril vería desde el lateral |
| contact_stable_length | DOUBLE PRECISION | mm. Lado **mayor** del minAreaRect 2D de la cara que toca la cinta (vértices con `proj·n ≤ min(proj·n) + 0.5 LDU`) |
| contact_stable_width | DOUBLE PRECISION | mm. Lado **menor** del mismo minAreaRect. **El cache JSON exige ≥ 4.0 mm** (4 mm = ancho de un stud 2×2 → mínimo físicamente sostenible en una cinta real) |
UNIQUE: (part_ref, pose_index).

**Origen de los datos derivados:**
- `simulate_stable_poses.py` (Blender + Bullet, `--save_db`) → `contact_normal`, `face_class`, `orientation_quat/euler`, `stability_ratio`, `simulation_passes/total`, `is_stable`.
- `scripts/populate_stable_pose_dims.py` (Python + LDraw mesh + scipy/cv2) → `zenith_observable_area`, `lateral_height`, `contact_stable_length`, `contact_stable_width`. Idempotente y reejecutable: lee `contact_normal` de la BD y recalcula los 4 campos a partir del mesh LDraw.
- `scripts/populate_pose_areas.py` → versión legacy (sólo `zenith_observable_area` + `lateral_height`); reemplazado por el script anterior.

**Filtros del cache JSON `2camaras_pieza_unica/data/stable_poses_cache.json`:**
El script `2camaras_pieza_unica/scripts/sync_stable_poses_cache.py` proyecta las filas de `stable_poses` al cache que consume el pipeline de Blender (`generate_test_set.py`, `generate_eevee_dinov2_refs.py`, `generate_yolo_training_dataset.py`, `generate_piece_report.py`). Sólo escribe poses que cumplen **AMBOS** criterios:

1. `stability_ratio > 0.05` — descarta poses prácticamente metaestables.
2. `contact_stable_width ≥ 4.0 mm` — descarta poses "de canto" sobre superficies imposibles de sostener.

Las poses con `contact_stable_width IS NULL` (cara de contacto degenerada por geometría redondeada) se conservan si pasan el primer criterio, asumiendo que la simulación física ya las validó.

### training_runs
| Campo | Tipo | Descripcion |
|-------|------|-------------|
| id | UUID PK | Identificador |
| status | TEXT | running/completed/failed |
| epochs | INTEGER | Total epocas |
| current_epoch | INTEGER | Ultima completada |
| loss / val_loss | FLOAT | Losses ultima epoca |
| map50 | FLOAT | mAP@0.5 |
| logs | TEXT | Logs acumulados (concat) |
| config_used | JSONB | Params: batch, epochs, etc. |

### Otras tablas
- lego_sets: code PK, name
- lego_set_parts: PK(set_code,part_ref,color_code), color_hex, color_name, qty
- lego_set_minifigures: PK(set_code,minifig_ref), name, qty
- minifig_assemblies: minifig_ref PK, glb_path, glb_data BYTEA, components JSONB

Vistas: session_summary, top_detected_classes, stable_pose_summary

Sets: 75078-1, 911943-1, 75280-1, 75218-1, 75337-1, 10692-1. Sets no registrados: inventario sintetico via random.seed(hash(set_id)).

---

## 5. Algoritmos Clave

### K-NN Consensus Voting (knn_classifier.py)

1. Fit-to-canvas 224x224 fondo negro; sustraccion fondo si existe empty_belt.png
2. DINOv2 ViT-S/14: CLS token 384d -> L2 normalize
3. Si MLP head: Linear(384->512)->LayerNorm->ReLU->Linear(512->128)->L2 norm
4. Filtro color HSV (sat>15=cromatico): H<15=Rojo, 15-45=Amarillo, 45-85=Verde, resto=Azul; acromatico=dist euclidiana
5. Filtro tamano: w_mm=px/3.2; max_crop<=diag+8mm y max>=major*0.55-2mm
6. scores = ref_matrix @ query_vec (dot product L2-norm = coseno)
7. Top-5 vecinos -> agrupar por part_ref
8. weighted_score = max_sim * (0.5 + 0.5*votes/k)
9. Escalado GUI: projected: >=0.5 -> 0.95+(s-0.5)*0.098; <0.5 -> 0.1+s*1.7

### Sustraccion de Fondo (api.py: apply_subtraction_mask)

1. diff = absdiff(frame, empty_belt_ref)
2. gray -> threshold(15) -> mask binaria
3. CLOSE(3x3, iter=2) + OPEN(3x3, iter=1)
4. result[mask>0]=frame, resto negro; tight-crop +5px

### Preprocessing DINOv2 (identico indexacion e inferencia)

- Referencias: alpha->fondo #254154, fit-to-canvas 224px margen 8px
- Inferencia: fit-to-canvas 224px fondo negro
- Transform: Resize(256)->CenterCrop(224)->ToTensor->Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

### Poses Estables

8 perturbaciones sinusoidales v(t)=v_max*sin(pi*t/T). Agrupar orientaciones si dist<15 grados. Estable si freq/total >= 0.875.

---

## 6. API de Inferencia - Endpoints

FastAPI en http://localhost:8005

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| / | GET | Health check |
| /session/start | POST | Inicia sesion: {model_version, belt_speed_mm_s, set_id} |
| /session/stop | POST | Cierra sesion: calcula avg_fps, avg_confidence |
| /detect | POST | Deteccion YOLO. Multipart: file, conf?, session_id? |
| /classify_crop | POST | Clasifica crop DINOv2. Body: {bbox, frame_b64, color_code, filename, set_id} |
| /inference-run/images | GET | Lista PNGs en temp_inference_run/ |
| /inference-run/image/{f} | GET | Devuelve imagen temporal |
| /inference-run/detect/{f} | POST | Deteccion sobre imagen temporal |
| /generate_inference_render | POST | Genera render de test via Blender |

Archivos estaticos: /renders/* -> synthetic_renders/ (8005), /models/* -> gui/static/models/ (8005), gui/static/ -> 8006.

---

## 7. Interfaz Grafica - GUI Bridge

SPA HTML/JS renderizada por PyWebView (WebKit nativo). Comunicacion: window.pywebview.api.metodo(). Puerto 8006: servidor estatico para GLBs.

### Metodos ApiBridge principales (gui/app.py)

| Metodo | Descripcion |
|--------|-------------|
| get_historical_stats() | Top 5 clases + 10 detecciones recientes |
| check_connection() | Verifica conexion PostgreSQL |
| get_set_inventory(set_id) | Inventario del set |
| start_training(...) | Pipeline dataset+YOLO en hilo daemon |
| stop_training() | Mata procesos con SIGKILL |
| get_training_status() | Estado training_run reciente |
| start_indexing(set_id) | Indexacion DINOv2 en hilo daemon |
| stop_indexing() | Mata procesos indexacion |
| get_indexing_progress() | {current, total, pct, active, done} |
| start_validation(runs) | Validacion poses estables |
| get_validation_progress() | {current, total, results} |
| start_dinov2_fov_simulation(...) | Simulacion FOV en hilo daemon |
| simulate_physics_scatter(...) | Fisica por pieza (bloqueante, con cache) |
| simulate_set_physics_scatter(set_id) | Fisica set completo |
| generate_inference_test_render(...) | Render de test |
| generate_validation_excel(set_id) | Excel comparativo poses |
| assemble_minifig(ref) | GLB de minifigura |

Gestion de subprocesos: start_new_session=True + os.killpg(SIGKILL) + flags _stop_*_flag.

---

## 8. Scripts y Utilidades

| Script | Descripcion |
|--------|-------------|
| scene_config.py | Parametros centralizados (BELT_COLOR #254154, CAMERA_Z 25, LDRAW_TO_BU 0.04, etc.) |
| generate_yolo_training_dataset.py | Dataset YOLO con Blender headless |
| generate_physics_ref_multiangle.py | 12 vistas por pieza para DINOv2 |
| validate_stable_poses.py | Simula estabilidad con perturbaciones |
| analyze_stable_poses_ldraw.py | Analisis geometrico LDraw |
| generate_validation_excel.py | Excel comparativo |
| physics_belt_generator.py | Fisica pieza individual (con cache) |
| physics_set_belt_generator.py | Fisica set completo |
| generate_inference_test_belt.py | Render de test de inferencia |
| migrate_sets.py | Migra REAL_SETS a tablas BD (one-time) |
| project_all_embeddings.py | Proyecta 384d a 128d con MLP head |
| ldraw_mesh_parser.py | Parser .dat LDraw |
| train_dino_metric_head.py | Entrena MLP Triplet Loss |

---

## 9. Variables de Entorno

Archivo .env (copiar desde .env.example). load_dotenv(override=True).

| Variable | Default | Descripcion |
|----------|---------|-------------|
| SUPABASE_DB_HOST | localhost | Host PostgreSQL |
| SUPABASE_DB_PORT | 5434 | Puerto no-estandar |
| SUPABASE_DB_NAME | legvision | Nombre BD |
| SUPABASE_DB_USER | postgres | Usuario |
| SUPABASE_DB_PASSWORD | legvision_pass_2024 | Password |
| MODEL_PATH | ./runs/train/best.pt | Modelo YOLO |
| MODEL_DEVICE | (auto) | mps/cuda/cpu |
| CONFIDENCE_THRESHOLD | 0.5 | Umbral YOLO |
| BELT_SPEED_MM_S | 83.3 | Velocidad cinta |
| API_PORT | 8005 | Puerto FastAPI |
| BLENDER_PATH | .../Blender | Ejecutable Blender |

---

## 10. Puesta en Marcha

### Requisitos

- Python 3.11+, Docker Desktop, Blender 4.x, Libreria LDraw en ~/ldraw/

### Primera vez

    cp .env.example .env  # configurar BLENDER_PATH
    bash scripts/setup_env.sh
    docker compose up -d
    psql -h localhost -p 5434 -U postgres -d legvision -f database/schema.sql
    python scripts/migrate_sets.py

### Arranque normal

    ./run.sh

### Ciclo tipico

1. ./run.sh
2. GUI: Entrenamiento -> generar dataset + entrenar (Flujo B)
3. GUI: Indexacion DINOv2 -> generar refs + indexar (Flujo C)
4. GUI: Activar sesion de inferencia
5. Enviar frames a /detect -> /classify_crop
6. Ver resultados en tiempo real

### Dependencias principales

ultralytics, torch, torchvision, fastapi, uvicorn, psycopg2, pywebview, Pillow, opencv-python, numpy, python-dotenv

---
*Documentacion LegoVision v2.0*
