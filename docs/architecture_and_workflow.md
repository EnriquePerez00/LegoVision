# LegoVision — Arquitectura, Variables y Flujo de Trabajo

Este documento proporciona una visión detallada de la arquitectura de LegoVision, el flujo de datos a través del sistema, la configuración de variables de entorno y los flujos de trabajo de desarrollo.

---

## 1. Arquitectura del Sistema

LegoVision está diseñado bajo una arquitectura modular y desacoplada que separa el entrenamiento, la API de inferencia en tiempo real y la interfaz de usuario.

```mermaid
graph TD
    subgraph Dataset (Imágenes Reales)
        A[Captura de imágenes] --> B[Anotaciones YOLO]
        B --> C[Dataset Local: PNG + TXT]
    end

    subgraph Entrenamiento (Lightning AI / Local)
        C -->|Entrenamiento con GPU| E[YOLO11 Training]
        E -->|Pesos Entrenados| F[best.pt]
    end

    subgraph Servidor de Inferencia (FastAPI)
        F -->|Carga de modelo en MPS| G[Detector YOLO11]
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
1.  **Inferencia Decoplada**: Servida a través de **FastAPI**. El API levanta el modelo YOLO11 utilizando aceleración **MPS (Metal Performance Shaders)** en macOS o **CUDA** en Linux.
2.  **Persistencia Robusta**: Utiliza una instancia dockerizada de **Supabase (PostgreSQL 16)**. Toda la comunicación de base de datos se realiza en puertos no estándar (`5434` para PostgreSQL y `5437` para PostgREST) para evitar conflictos con bases de datos preexistentes.
3.  **UI Ligera**: Construida con **PyWebView**. Utiliza el motor WebKit nativo del sistema operativo (evitando Electron) y ofrece un simulador dinámico en Canvas 2D que interactúa directamente con el API de inferencia.

---

## 2. Flujo de Datos (Data Flow)

El flujo de información en tiempo de ejecución sigue los siguientes pasos:

1.  **Adquisición**: La cámara industrial (o el simulador de cinta en el Canvas de la interfaz gráfica) captura un frame a una frecuencia constante (tasa de muestreo de inferencia de 5 FPS).
2.  **Transmisión**: La imagen se envía como un archivo binario mediante una petición `POST /detect` al servidor FastAPI.
3.  **Detección**:
    *   FastAPI recibe el archivo.
    *   El wrapper `LegoDetector` convierte la imagen a formato RGB y la envía a la red neuronal YOLO11 corriendo en el procesador gráfico (MPS).
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

# === PIPELINE DE ENTRENAMIENTO ===
DATASET_OUTPUT=./data/raw_dataset
NUM_IMAGES=1000
```

---

## 4. Flujo de Trabajo de Desarrollo (Workflow)

El ciclo de desarrollo y puesta en marcha del proyecto sigue este flujo paso a paso:

```
[1. Configuración de Entorno] 
       ↓ (bash scripts/setup_env.sh)
[2. Recopilación de Dataset]
       ↓ (captura de fotos reales en /data/raw_dataset)
[3. Entrenamiento en la Nube / Local]
       ↓ (python training/train_yolo.py)
[4. Despliegue del API y de la Interfaz GUI]
```

### Paso 1: Inicialización
Ejecutar `bash scripts/setup_env.sh` para crear el entorno virtual, instalar dependencias y encender el motor de base de datos PostgreSQL local en Docker.

### Paso 2: Recopilación del Dataset
Colocar las imágenes reales y sus correspondientes archivos de anotación en la carpeta de entrada del dataset.

### Paso 3: Entrenamiento
Correr `train_yolo.py` para procesar el dataset y entrenar el modelo YOLO11. El modelo exporta los pesos entrenados.

### Paso 4: Ejecución
Iniciar el API de inferencia y la GUI de visualización para ver y persistir la clasificación de piezas sobre la cinta transportadora.
 y persistir en vivo la clasificación de piezas sobre la cinta transportadora.
