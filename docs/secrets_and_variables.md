# LegoVision — Variables de Entorno y Secretos

Este documento detalla la configuración del sistema, las variables de entorno y el manejo de secretos para el proyecto **LegoVision**.

> [!WARNING]
> Nunca cometas el archivo `.env` o cualquier credencial real al repositorio Git. El archivo `.gitignore` está configurado para evitar fugas accidentales.

## Gestión de Secretos

Para evitar problemas de seguridad, todas las credenciales se almacenan localmente en un archivo `.env` en la raíz del proyecto. Este archivo está excluido del control de versiones.

Se proporciona un archivo plantilla [`.env.example`](file:///Users/I764690/Code_personal/LegoVision/.env.example) para guiar la configuración.

---

## Tabla de Variables de Entorno

A continuación se describen todas las variables utilizadas por el sistema:

### 1. Configuración de Base de Datos (Supabase Local Docker)

Estas variables configuran la conexión a la base de datos PostgreSQL local corriendo en Docker.

| Variable | Tipo / Valor por defecto | Descripción | ¿Es Secreto? |
|----------|--------------------------|-------------|--------------|
| `SUPABASE_URL` | `http://localhost:5437` | URL de la API PostgREST. | No |
| `SUPABASE_DB_HOST` | `localhost` | Host del servidor de base de datos. | No |
| `SUPABASE_DB_PORT` | `5434` | Puerto expuesto de PostgreSQL (evita conflictos en 5432). | No |
| `SUPABASE_DB_NAME` | `legvision` | Nombre de la base de datos PostgreSQL. | No |
| `SUPABASE_DB_USER` | `postgres` | Nombre del usuario administrador de PostgreSQL. | No |
| `SUPABASE_DB_PASSWORD` | `legvision_pass_2024` | Contraseña para el usuario `postgres`. | **Sí** |
| `SUPABASE_ANON_KEY` | `legvision-anon-key` | Clave anónima para PostgREST (se utiliza si usas Kong API Gateway). | No |

---

### 2. Inferencia y API (FastAPI)

Configuración de la API local de FastAPI encargada de recibir las imágenes de la cámara y correr el detector YOLOv8.

| Variable | Tipo / Valor por defecto | Descripción | ¿Es Secreto? |
|----------|--------------------------|-------------|--------------|
| `API_HOST` | `0.0.0.0` | IP de escucha del servidor FastAPI. | No |
| `API_PORT` | `8001` | Puerto del servidor de la API de inferencia. | No |
| `API_RELOAD` | `true` | Habilita auto-reload en desarrollo (`true` o `false`). | No |
| `MODEL_PATH` | `./runs/train/best.pt` | Path local a los pesos entrenados del modelo YOLOv8 (`.pt`). | No |
| `MODEL_DEVICE` | `mps` | Hardware para inferencia: `mps` (Metal M4), `cpu` o `cuda`. | No |
| `INFERENCE_FPS_TARGET` | `5` | FPS objetivo de la inferencia (latencia máx ~200ms). | No |
| `CONFIDENCE_THRESHOLD` | `0.5` | Umbral de confianza mínimo para reportar una detección. | No |
| `IOU_THRESHOLD` | `0.45` | Umbral NMS (Non-Maximum Suppression) IOU. | No |

---

### 3. Cámara y Cinta Transportadora

Configuración física de la captura de imágenes sobre la cinta.

| Variable | Tipo / Valor por defecto | Descripción | ¿Es Secreto? |
|----------|--------------------------|-------------|--------------|
| `CAMERA_INDEX` | `0` | Índice del dispositivo de cámara física en OpenCV. | No |
| `CAMERA_TRIGGER_MODE` | `polling` | Modo de disparo: `polling` (bucle continuo) o `trigger` (vía encoder). | No |
| `BELT_SPEED_MM_S` | `83.3` | Velocidad de la cinta transportadora en mm/s (5 metros/min = 83.3 mm/s). | No |

---

### 4. Pipeline de Blender (Generación Sintética)

Configuración paramétrica para el renderizador de Blender Cycles y el importador LDraw.

| Variable | Tipo / Valor por defecto | Descripción | ¿Es Secreto? |
|----------|--------------------------|-------------|--------------|
| `BLENDER_PATH` | *Auto-detectado por setup* | Ruta absoluta al ejecutable de Blender en macOS. | No |
| `LDRAW_PATH` | `./data/ldraw` | Ruta al directorio donde se extrae la librería de LDraw. | No |
| `DATASET_OUTPUT` | `./data/raw_dataset` | Ruta de salida para las imágenes y etiquetas YOLO generadas. | No |
| `NUM_IMAGES` | `1000` | Número de imágenes sintéticas a generar para el entrenamiento. | No |
| `PIECES_PER_IMAGE` | `15` | Cantidad máxima de piezas LEGO que caen en cada escena de simulación física. | No |

---

### 5. Entrenamiento en la Nube (Lightning AI)

Variables para lanzar entrenamientos remotos en la plataforma Lightning AI utilizando GPUs NVIDIA T4.

| Variable | Tipo / Valor por defecto | Descripción | ¿Es Secreto? |
|----------|--------------------------|-------------|--------------|
| `LIGHTNING_API_KEY` | *(Vacío / Obligatorio)* | Token de autenticación de tu cuenta de Lightning AI. | **Sí** |
| `LIGHTNING_PROJECT` | `legvision` | Nombre del proyecto/espacio de trabajo en Lightning AI. | No |

---

## Seguridad en Git (.gitignore)

El archivo [`.gitignore`](file:///Users/I764690/Code_personal/LegoVision/.gitignore) en la raíz excluye explícitamente:
- Archivos de entorno y secretos (`.env`, `.env.local`, `.env.production`, `*.key`, `*.pem`).
- El catálogo de LDraw (`data/ldraw/`), que ocupa más de 500MB y se autodescarga.
- Los datasets generados localmente (`data/raw_dataset/`, `data/processed_dataset/`).
- Los pesos del modelo de deep learning (`*.pt`, `*.pth`, `*.onnx`, `*.engine`) y carpetas de runs (`runs/`).
- Los temporales de Blender (`*.blend1`, `*.blend2`) y de Python (`__pycache__/`, `.venv`).
