# =============================================================
# LegoVision — Configuración de Generación Sintética
# =============================================================
import os
from dotenv import load_dotenv

# Cargar .env si está disponible (no sobrescribir si ya está definido en el entorno)
load_dotenv(override=False)

# --- 1. SETUP FÍSICO Y ÓPTICO ---
RESOLUTION_X = 2448         # px (Sony IMX264)
RESOLUTION_Y = 2048         # px
SENSOR_WIDTH_MM = 8.44      # mm (Sensor de 2/3")
LENS_FOCAL_LENGTH_MM = 12.0  # mm (Lente de 12mm C-mount)
CAMERA_HEIGHT_MM = 355.0     # mm (Working Distance / Altura cámara)
BELT_WIDTH_MM = 200.0       # mm (Ancho de la cinta transportadora)
FOV_LONGITUDINAL_MM = 250.0  # mm (Campo de visión longitudinal)

# --- 2. CONFIGURACIÓN DE CONVERSIÓN ---
BLENDER_SCALE = 0.001       # mm -> metros en Blender (1mm = 0.001m)

# --- 3. PARÁMETROS DEL DATASET ---
NUM_IMAGES = int(os.getenv("NUM_IMAGES", "1000"))
PIECES_PER_IMAGE = int(os.getenv("PIECES_PER_IMAGE", "15"))

# --- 4. RUTAS DE ARCHIVOS ---
LDRAW_PATH = os.getenv("LDRAW_PATH", "./data/ldraw")
CATALOG_INDEX = os.path.join(LDRAW_PATH, "catalog_index.json")
DATASET_OUTPUT = os.getenv("DATASET_OUTPUT", "./data/raw_dataset")

# --- 5. RENDER ENGINE (Cycles) ---
RENDER_SAMPLES = 128
USE_DENOISING = True
DEVICE = "GPU"              # GPU | CPU
