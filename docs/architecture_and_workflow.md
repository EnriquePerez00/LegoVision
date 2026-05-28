# LegoVision — Arquitectura, Variables y Flujo de Trabajo

Este documento proporciona una visión detallada de la arquitectura de LegoVision, el flujo de datos a través del sistema, la configuración de variables de entorno y los flujos de trabajo de desarrollo.

---

## 1. Arquitectura del Sistema

LegoVision está diseñado bajo una arquitectura modular y desacoplada que separa la generación de datos sintéticos, el entrenamiento, la API de inferencia en tiempo real y la interfaz de usuario.

```mermaid
graph TD
    subgraph Generación de Dataset (Headless Blender)
        A[Catálogo LDraw .dat] -->|Importación Rust| B[ldr_tools_blender]
        B -->|Físicas y Render| C[Blender bpy Pipeline]
        C -->|Render fotorrealista| D[Dataset Sintético: PNG + TXT]
    end

    subgraph Entrenamiento (Lightning AI / Local)
        D -->|Entrenamiento con GPU| E[YOLOv8 Training]
        E -->|Pesos Entrenados| F[best.pt / best.onnx]
    end

    subgraph Servidor de Inferencia (FastAPI)
        F -->|Carga de modelo en MPS| G[Detector YOLOv8]
        H[Cámara / Canvas Sim] -->|POST /detect| I[Endpoints FastAPI]
        I -->|Inferencia| G
        I -->|Async Background Task| J[psycopg2 Client]
    end

    subgraph Persistencia y Control
        J -->|Inserción en Lote| K[(Supabase Postgres Docker)]
        L[PyWebView GUI] -->|API REST / WebSocket| I
        L -->|Python Bridge| K
    end
```

### Componentes Clave:
1.  **Motor de Generación Sintética**: Corre sobre **Blender (Headless)**. Importa geometría LDraw mediante un plugin escrito en **Rust (`ldr_tools_blender`)**, aplica físicas tridimensionales con el motor de cuerpos rígidos de Blender y renderiza imágenes usando **Cycles** (con soporte para la GPU Metal del chip M4).
2.  **Inferencia Decoplada**: Servida a través de **FastAPI**. El API levanta el modelo YOLOv8 utilizando aceleración **MPS (Metal Performance Shaders)** en macOS o **CUDA** en Linux.
3.  **Persistencia Robusta**: Utiliza una instancia dockerizada de **Supabase (PostgreSQL 16)**. Toda la comunicación de base de datos se realiza en puertos no estándar (`5434` para PostgreSQL y `5437` para PostgREST) para evitar conflictos con bases de datos preexistentes.
4.  **UI Ligera**: Construida con **PyWebView**. Utiliza el motor WebKit nativo del sistema operativo (evitando Electron) y ofrece un simulador dinámico en Canvas 2D que interactúa directamente con el API de inferencia.

---

## 2. Flujo de Datos (Data Flow)

El flujo de información en tiempo de ejecución sigue los siguientes pasos:

1.  **Adquisición**: La cámara industrial (o el simulador de cinta en el Canvas de la interfaz gráfica) captura un frame a una frecuencia constante (tasa de muestreo de inferencia de 5 FPS).
2.  **Transmisión**: La imagen se envía como un archivo binario mediante una petición `POST /detect` al servidor FastAPI.
3.  **Detección**:
    *   FastAPI recibe el archivo.
    *   El wrapper `LegoDetector` convierte la imagen a formato RGB y la envía a la red neuronal YOLOv8 corriendo en el procesador gráfico (MPS).
    *   El modelo genera las predicciones de bounding boxes 2D (normalizadas de 0 a 1) y los índices de las clases detectadas.
4.  **Respuesta rápida**: La API retorna inmediatamente el JSON con las detecciones a la interfaz gráfica para pintar las bounding boxes sobre el video sin retrasos.
5.  **Persistencia Asíncrona**: En paralelo, el API crea una tarea en segundo plano (`BackgroundTasks`) que inserta en lote las detecciones y las métricas de latencia de inferencia en la base de datos de Supabase local, manteniendo la UI y el API libres de bloqueos de I/O.

---

## 3. Variables de Entorno y Configuración

El comportamiento del sistema es totalmente paramétrico y se define a través de variables de entorno cargadas desde el archivo [`.env`](file:///Users/I764690/Code_personal/LegoVision/.env) con la opción `override=True` para evitar la polución por variables globales del sistema operativo.

### Módulos de Configuración:

```ini
# === CONFIGURACIÓN DE BASE DE DATOS ===
SUPABASE_URL=http://localhost:5437
SUPABASE_DB_HOST=localhost
SUPABASE_DB_PORT=5434
SUPABASE_DB_NAME=legvision
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=legvision_pass_2024

# === INFERENCIA Y DETECTOR ===
MODEL_PATH=./runs/train/best.pt
MODEL_DEVICE=mps             # mps (Mac M4) | cuda (GPU) | cpu
CONFIDENCE_THRESHOLD=0.5     # Umbral de confianza mínimo

# === CINTA TRANSPORTADORA ===
BELT_SPEED_MM_S=83.3         # Velocidad física (5 m/min)

# === PIPELINE DE GENERACIÓN DE IMÁGENES ===
BLENDER_PATH=/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender
LDRAW_PATH=./data/ldraw
DATASET_OUTPUT=./data/raw_dataset
NUM_IMAGES=1000
PIECES_PER_IMAGE=15
```

---

## 4. Flujo de Trabajo de Desarrollo (Workflow)

El ciclo de desarrollo y puesta en marcha del proyecto sigue este flujo paso a paso:

```
[1. Configuración de Entorno] 
       ↓ (bash scripts/setup_env.sh)
[2. Descarga e Indexación de Catálogo]
       ↓ (bash scripts/download_ldraw.sh)
[3. Generación Sintética Headless]
       ↓ (blender --background --python blender_pipeline/generate_dataset.py)
[4. Entrenamiento en la Nube / Local]
       ↓ (python training/train_lightning.py)
[5. Despliegue del API y de la Interfaz GUI]
```

### Paso 1: Inicialización
Ejecutar `bash scripts/setup_env.sh` para crear el entorno virtual, instalar dependencias y encender el motor de base de datos PostgreSQL local en Docker.

### Paso 2: Descarga de Catálogo
Ejecutar `bash scripts/download_ldraw.sh` para obtener las piezas del catálogo LDraw e instalar el addon de importación nativa en Blender.

### Paso 3: Generación Sintética
Correr el pipeline de Blender indicando el número de imágenes. El renderizador utiliza Cycles con Metal (MPS) en macOS para acelerar el renderizado fotorrealista de las piezas.

### Paso 4: Entrenamiento
Subir el dataset generado a Lightning AI y correr `train_lightning.py` para entrenar YOLOv8 con GPU NVIDIA T4. El modelo exporta pesos optimizados en formato `.pt` y `.onnx`.

### Paso 5: Ejecución
Iniciar el API de inferencia con `uvicorn` y lanzar la interfaz gráfica con `python gui/app.py` para visualizar y persistir en vivo la clasificación de piezas sobre la cinta transportadora.
