# -*- coding: utf-8 -*-
BU_PER_MM = 0.1
LDRAW_TO_BU = 0.04
LDRAW_THRESHOLD = 5.0
BELT_SURFACE_Z = 0.0
BELT_WIDTH_BU = 20.0
BELT_LENGTH_BU = 60.0
BELT_THICKNESS_BU = 8.0
BELT_COLOR_LINEAR = (0.145, 0.255, 0.33, 1.0)
BELT_COLOR_HEX = "#254154"
BELT_COLOR_RGB_255 = (37, 65, 84)
BELT_FRICTION = 0.95
BELT_RESTITUTION = 0.02
CAMERA_TYPE = "ORTHO"
CAMERA_ORTHO_SCALE = 20.0
CAMERA_Z = 25.0
CAMERA_ROTATION = (0.0, 0.0, 0.0)
RENDER_RES_X = 640
RENDER_RES_Y = 1920
RENDER_RES_SQUARE = 640
PX_PER_BU = RENDER_RES_X / CAMERA_ORTHO_SCALE
PX_PER_MM = PX_PER_BU * BU_PER_MM
GRAVITY_Z = -9.81
PHYSICS_FRAMES = 100
PIECE_MASS_KG = 0.008
PIECE_FRICTION = 0.95
PIECE_RESTITUTION = 0.02
# Iluminacion cenital principal (reducida para evitar hotspot central)
TOP_LIGHT_SIZE = 22.0
TOP_LIGHT_ENERGY = 250.0
TOP_LIGHT_Z = 22.0
# Luces de esquina: 4 luces de area en (+-8, +-8, 18) BU
# Tamano 14 BU cubre el cuadrante 10x10 BU con 2 BU de solape -> iluminacion uniforme
CORNER_LIGHT_OFFSET_XY = 8.0
CORNER_LIGHT_Z = 18.0
CORNER_LIGHT_SIZE = 14.0
CORNER_LIGHT_ENERGY = 180.0
WORLD_BG_STRENGTH = 0.1
WORLD_BG_COLOR = (0.9, 0.9, 0.9, 1.0)
CYCLES_SAMPLES_DATASET = 4
CYCLES_SAMPLES_REF = 48
CYCLES_SAMPLES_FINAL = 64
DINO_CANVAS_SIZE = 224
DINO_CANVAS_MARGIN_PX = 8
DINO_BG_COLOR = BELT_COLOR_RGB_255
# DEFAULT_SPAWN_Z=8.0: margen para piezas grandes (3.2 BU) rotando en vuelo
DEFAULT_SPAWN_Z = 8.0
SPAWN_JITTER_XY = 1.5
GRID_COLS_X = [-6.0, 0.0, 6.0]
NUM_PHYSICS_DROPS = 20
YOLO_FRAMES = 500
# Proporciones reales: piezas grandes ocupan mas espacio -> menos piezas por frame
YOLO_PIECES_PER_FRAME_MIN = 20
YOLO_PIECES_PER_FRAME_MAX = 35
ALL_SET_IDS = ["75078-1", "911943-1", "75280-1", "75218-1", "75337-1", "10692-1", "75038-1", "31062-1", "75018-1", "6008-1"]
YOLO_EMPTY_FRAME_RATIO = 0.05
YOLO_GRID_COLS = [-8.5, -5.0, -1.5, 1.5, 5.0, 8.5]

# ── Límites del FOV de la cámara ORTHO para spawn de piezas ──────────────────
# La cámara ORTHO cubre exactamente CAMERA_ORTHO_SCALE BU en cada eje.
# El centro es (0,0), por lo tanto el FOV va de -HALF a +HALF en X e Y.
# Se aplica un margen interior de SPAWN_FOV_MARGIN_BU para evitar que
# piezas grandes generen bounding boxes parciales fuera del frame.
CAMERA_FOV_HALF_BU = CAMERA_ORTHO_SCALE / 2.0          # 10.0 BU
SPAWN_FOV_MARGIN_BU = 2.0                                # margen de seguridad (BU)
SPAWN_FOV_MIN = -CAMERA_FOV_HALF_BU + SPAWN_FOV_MARGIN_BU   # -8.0 BU
SPAWN_FOV_MAX =  CAMERA_FOV_HALF_BU - SPAWN_FOV_MARGIN_BU   #  8.0 BU

# Directorios de salida separados por pipeline
# YOLO training dataset -> data/raw_dataset/
YOLO_OUTPUT_SUBDIR = "raw_dataset"
# DINOv2 physics scatter crops -> data/dino_scatter/
DINO_SCATTER_SUBDIR = "dino_scatter"

# -- Parametros de simulacion de posiciones estables -------------------------
# Perturbacion tipo arranque suave de cinta de goma (perfil sinusoidal)
# v(t) = v_max * sin(pi*t/T), aceleracion pico = pi*v_max/T aprox 0.26g
STABLE_POSE_BELT_SPEED_MS = 0.0833       # Velocidad max cinta: 83.3 mm/s
STABLE_POSE_IMPULSE_DURATION_S = 0.1     # Duracion del arranque: 0.1 s
STABLE_POSE_SIM_FPS = 60                 # FPS de la simulacion
STABLE_POSE_SETTLE_FRAMES = 20           # Frames de asentamiento previo
STABLE_POSE_IMPULSE_FRAMES = 6           # Frames del impulso (0.1s * 60fps)
STABLE_POSE_EVAL_FRAMES = 40             # Frames de evaluacion tras impulso
STABLE_POSE_TOTAL_FRAMES = 66            # Total frames por perturbacion
STABLE_POSE_N_DIRECTIONS = 8             # Numero de direcciones aleatorias
STABLE_POSE_STABILITY_THRESHOLD = 0.875  # 7/8 perturbaciones deben superarse
STABLE_POSE_ANGLE_THRESHOLD_DEG = 15.0  # Angulo max para misma pose (grados)
STABLE_POSE_MIN_FACE_AREA_LDU2 = 28.3  # Area minima = 1 stud circle (pi*3^2)
STABLE_POSE_SETTLE_Z_OFFSET = 0.3       # Offset Z sobre superficie al colocar
