import os
import sys
import time
import io
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import List, Optional
from dotenv import load_dotenv

# Añadir directorio raíz al path para importar desde database y detector
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference import config
from database import supabase_client
from inference.detector import LegoDetector

# Cargar variables de entorno
load_dotenv(override=True)

app = FastAPI(
    title="LegoVision Inference API",
    description="API de inferencia en tiempo real para detección de piezas LEGO sobre cinta transportadora.",
    version="1.0.0"
)

# Habilitar CORS para integración con la GUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
renders_dir = os.path.join(project_root, "data", "synthetic_renders")
os.makedirs(renders_dir, exist_ok=True)
app.mount("/renders", StaticFiles(directory=renders_dir), name="renders")
models_dir = os.path.join(project_root, "gui", "static", "models")
os.makedirs(models_dir, exist_ok=True)
app.mount("/models", StaticFiles(directory=models_dir), name="models")


# Inicializar detector global con autocalibración M4
MODEL_PATH = os.getenv("MODEL_PATH", "./runs/train/best.pt")
DEVICE = os.getenv("MODEL_DEVICE") or config.DEFAULT_DEVICE
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

detector = LegoDetector(model_path=MODEL_PATH, device=DEVICE, conf_threshold=CONF_THRESHOLD)

# Clasificador DINOv2 K-NN y MLP Head (carga lazy en primera petición)
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        from inference.knn_classifier import LegoKNNClassifier
        _classifier = LegoKNNClassifier(k=5, top_k_classes=3)
        _classifier.load_projection_head()
        _classifier.load_reference_embeddings()
    return _classifier

# Estado de la sesión actual de la cinta transportadora
class SessionState:
    def __init__(self):
        self.active_session_id = None
        self.active_set_id = None
        self.start_time = None
        self.frame_count = 0
        self.total_latency_ms = 0.0
        self.confidences = []

session_state = SessionState()

class SessionStartRequest(BaseModel):
    model_version: str = "yolov8n_synthetic"
    belt_speed_mm_s: float = 83.3
    set_id: Optional[str] = None

class ClassifyCropRequest(BaseModel):
    """
    Solicitud de clasificación de un crop de pieza LEGO.
    bbox: [x1, y1, x2, y2] en píxeles absolutos del frame.
    frame_b64: imagen completa del frame en base64 (PNG o JPEG).
    """
    bbox: List[float]         # [x1, y1, x2, y2] píxeles absolutos
    frame_b64: str            # frame completo en base64
    color_code: Optional[str] = "0"  # para buscar render de referencia en BD
    filename: Optional[str] = None   # nombre del archivo original para cargar de disco si existe
    set_id: Optional[str] = None     # ID del set activo para filtrar candidatos

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model": detector.model_path,
        "device": detector.device,
        "active_session": session_state.active_session_id
    }

@app.post("/session/start")
def start_session(req: SessionStartRequest):
    """Inicia una nueva sesión de inferencia en la cinta transportadora."""
    if session_state.active_session_id is not None:
        return {
            "status": "error",
            "message": f"Ya hay una sesión activa: {session_state.active_session_id}",
            "session_id": session_state.active_session_id
        }
        
    try:
        # Limpiar directorio temporal
        temp_dir = os.path.join(project_root, "data", "temp_inference_run")
        if os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[API Warning] No se pudo borrar la carpeta temporal: {e}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            session_id = supabase_client.create_session(
                model_version=req.model_version,
                belt_speed_mm_s=req.belt_speed_mm_s
            )
        except Exception as db_err:
            import uuid
            print(f"[API Warning] No se pudo crear sesión en DB, usando local/demo ID: {db_err}")
            session_id = f"local-{uuid.uuid4()}"

        session_state.active_session_id = session_id
        session_state.active_set_id = req.set_id
        session_state.start_time = time.time()
        session_state.frame_count = 0
        session_state.total_latency_ms = 0.0
        session_state.confidences = []
        
        print(f"[LegoVision API] Sesión de inferencia iniciada: {session_id} para set: {req.set_id}")
        return {
            "status": "success",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando sesión: {str(e)}")

@app.post("/session/stop")
def stop_session():
    """Cierra la sesión de inferencia activa y calcula las métricas medias."""
    if session_state.active_session_id is None:
        raise HTTPException(status_code=400, detail="No hay ninguna sesión activa para detener.")
        
    try:
        duration = time.time() - session_state.start_time
        avg_fps = session_state.frame_count / duration if duration > 0 else 0
        
        avg_confidence = None
        if session_state.confidences:
            avg_confidence = sum(session_state.confidences) / len(session_state.confidences)
            
        try:
            supabase_client.close_session(
                session_id=session_state.active_session_id,
                avg_fps=avg_fps,
                avg_confidence=avg_confidence
            )
        except Exception as db_err:
            print(f"[API Warning] No se pudo cerrar la sesión en DB: {db_err}")
            
        stopped_id = session_state.active_session_id
        session_state.active_session_id = None
        
        print(f"[LegoVision API] Sesión de inferencia cerrada: {stopped_id}")
        return {
            "status": "success",
            "session_id": stopped_id,
            "avg_fps": avg_fps,
            "avg_confidence": avg_confidence,
            "total_frames": session_state.frame_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cerrando sesión: {str(e)}")

def save_detections_background(session_id: str, detections: list[dict], inference_ms: float):
    """Guarda detecciones de forma asíncrona para no bloquear el response del API."""
    if not session_id or not detections:
        return
    try:
        formatted_detections = []
        for d in detections:
            formatted_detections.append({
                "class": d["class"],
                "name": d["name"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
                "inference_ms": inference_ms
            })
        supabase_client.save_detections_batch(session_id, formatted_detections)
    except Exception as e:
        print(f"[LegoVision API DB error] Falló el guardado en segundo plano: {e}")

@app.post("/detect")
async def detect_pieces(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = None,
    session_id: str = Form(None)
):
    """
    Recibe una imagen multipart de cámara, ejecuta inferencia
    y guarda las detecciones si hay una sesión activa.
    """
    start_time = time.time()
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagen inválida: {str(e)}")
        
    detections = detector.detect(image, conf=conf)
    
    latency_ms = (time.time() - start_time) * 1000.0
    
    frame_filename = None
    
    # Restoring active session if provided by frontend
    if session_id:
        session_state.active_session_id = session_id

    if session_state.active_session_id is not None:
        session_state.frame_count += 1
        session_state.total_latency_ms += latency_ms
        for d in detections:
            session_state.confidences.append(d["confidence"])
            
        background_tasks.add_task(
            save_detections_background,
            session_state.active_session_id,
            detections,
            latency_ms
        )

        # Guardar frame y JSON en temp_inference_run
        temp_dir = os.path.join(project_root, "data", "temp_inference_run")
        os.makedirs(temp_dir, exist_ok=True)
        frame_filename = f"frame_{session_state.frame_count:05d}.png"
        try:
            # 1. Guardar la imagen original limpia
            out_img_clean_path = os.path.join(temp_dir, frame_filename.replace(".png", "_clean.png"))
            image.save(out_img_clean_path, format="PNG")

            # 2. Dibujar las bboxes y guardar la anotada
            from PIL import ImageDraw
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            w_img, h_img = img_with_boxes.size
            for d in detections:
                xc, yc, wn, hn = d["bbox"]
                x1 = int((xc - wn/2) * w_img)
                y1 = int((yc - hn/2) * h_img)
                x2 = int((xc + wn/2) * w_img)
                y2 = int((yc + hn/2) * h_img)
                draw.rectangle([x1, y1, x2, y2], outline="#00ff88", width=3)
                label = f"{d['name']} {int(d['confidence'] * 100)}%"
                draw.text((x1 + 4, max(0, y1 - 12)), label, fill="#00ff88")
                
            out_img_path = os.path.join(temp_dir, frame_filename)
            img_with_boxes.save(out_img_path, format="PNG")
                
            out_json_path = os.path.join(temp_dir, frame_filename.replace(".png", ".json"))
            import json
            with open(out_json_path, "w") as f:
                json.dump({
                    "filename": frame_filename,
                    "latency_ms": latency_ms,
                    "detections_count": len(detections),
                    "detections": detections
                }, f, indent=2)
        except Exception as e:
            print(f"[API Error] No se pudo guardar el frame temporal de sesión: {e}")
        
    return {
        "latency_ms": latency_ms,
        "detections_count": len(detections),
        "detections": detections,
        "session_id": session_state.active_session_id,
        "filename": frame_filename
    }


def get_part_info(part_ref: str, color_code: str = None) -> tuple[str, str, str]:
    """
    Busca metadatos de la pieza y color en el catálogo estático.
    Retorna: (part_name, color_name, color_hex)
    """
    from database.set_catalog import REAL_SETS, LDRAW_PARTS, LDRAW_COLORS
    
    # 1. Buscar nombre en REAL_SETS o LDRAW_PARTS
    part_name = None
    
    # Buscar en sets reales
    for set_id, set_data in REAL_SETS.items():
        for p in set_data.get("parts", []):
            if p["ref"] == part_ref:
                part_name = p.get("name")
                if part_name:
                    break
        if part_name:
            break
            
    # Buscar en catálogo de piezas comunes
    if not part_name:
        for p in LDRAW_PARTS:
            if p["ref"] == part_ref:
                part_name = p.get("name")
                break
                
    if not part_name:
        part_name = f"Pieza LDraw {part_ref}"

    # 2. Buscar información del color
    color_name = "Unknown Color"
    color_hex = "#FFFFFF"
    
    if color_code:
        found_color = False
        # Buscar en sets reales
        for set_id, set_data in REAL_SETS.items():
            for p in set_data.get("parts", []):
                if str(p.get("color_code")) == str(color_code):
                    color_name = p.get("color_name") or "Unknown Color"
                    color_hex = p.get("color_hex") or "#FFFFFF"
                    found_color = True
                    break
            if found_color:
                break
                
        # Buscar en colores comunes
        if not found_color:
            for c in LDRAW_COLORS:
                if str(c["code"]) == str(color_code):
                    color_name = c["name"]
                    color_hex = c["hex"]
                    found_color = True
                    break
                    
    return part_name, color_name, color_hex


def load_empty_belt_ref() -> Optional[Image.Image]:
    """Carga la imagen de referencia de la cinta vacía si existe en el disco."""
    paths = [
        os.path.join(project_root, "scratch", "empty_belt.png"),
        os.path.join(project_root, "data", "empty_belt.png"),
        os.path.join(project_root, "empty_belt.png"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[API Warning] No se pudo cargar {p}: {e}")
    return None


def apply_subtraction_mask(crop_img: Image.Image, empty_crop_img: Image.Image) -> Image.Image:
    """Aplica la máscara de sustracción de fondo sobre el recorte y pinta el fondo de negro (0,0,0)."""
    try:
        import cv2
        import numpy as np
        img_bgr = cv2.cvtColor(np.array(crop_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        empty_bgr = cv2.cvtColor(np.array(empty_crop_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Resta absoluta
        diff = cv2.absdiff(img_bgr, empty_bgr)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Umbral (15)
        _, mask = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
        
        # Morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Pintar fondo de negro plano (0,0,0) para DINOv2
        result = np.zeros_like(img_bgr)
        result[mask > 0] = img_bgr[mask > 0]
        
        # Convertir a PIL
        piece = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        
        # Tight crop al bounding box de la máscara
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            h_c, w_c = result.shape[:2]
            y_min = max(0, y_min - 5)
            x_min = max(0, x_min - 5)
            y_max = min(h_c, y_max + 5)
            x_max = min(w_c, x_max + 5)
            piece = piece.crop((x_min, y_min, x_max, y_max))
            
        return piece
    except Exception as e:
        print(f"[API Warning] Error en apply_subtraction_mask: {e}")
        return crop_img


@app.post("/classify_crop")
async def classify_crop(req: ClassifyCropRequest):
    """
    Clasifica un crop de pieza LEGO usando DINOv2 por similitud coseno.

    Recibe:
      - bbox [x1, y1, x2, y2] en píxeles absolutos del frame.
      - frame_b64: frame completo en base64.

    Devuelve el top-3 de piezas más similares con:
      - part_ref, part_name, score (0-1), ref_image_b64 (render de referencia desde BD).
    """
    start_time = time.time()

    # 1. Decodificar frame y extraer crop
    loaded_from_disk = False
    if req.filename:
        clean_filename = req.filename.replace(".png", "_clean.png")
        clean_path = os.path.join(TEMP_RUN_DIR, clean_filename)
        if os.path.exists(clean_path):
            try:
                frame_img = Image.open(clean_path).convert("RGB")
                loaded_from_disk = True
            except Exception as e:
                print(f"[API Warning] No se pudo cargar imagen limpia desde disco: {e}")
                
    if not loaded_from_disk:
        try:
            frame_bytes = base64.b64decode(req.frame_b64)
            frame_img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Frame inválido: {str(e)}")

    x1, y1, x2, y2 = [int(v) for v in req.bbox]
    # Clamping al tamaño del frame
    w, h = frame_img.size
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    # Cargar fondo de referencia vacío
    empty_belt = load_empty_belt_ref()
    
    # Extraer crop principal
    crop = frame_img.crop((x1, y1, x2, y2))
    
    # Si hay fondo de referencia vacío, aplicar sustracción
    if empty_belt is not None:
        try:
            # Asegurar que el fondo de referencia tiene el mismo tamaño que el frame
            if empty_belt.size == frame_img.size:
                empty_crop = empty_belt.crop((x1, y1, x2, y2))
                crop = apply_subtraction_mask(crop, empty_crop)
                print("[API] Aplicada máscara de sustracción de fondo al crop de inferencia.")
            else:
                print(f"[API Warning] Dimensiones incompatibles: empty_belt {empty_belt.size} vs frame {frame_img.size}")
        except Exception as e:
            print(f"[API Error] Error aplicando resta de fondo: {e}")

    # 2. Clasificar con DINOv2
    try:
        clf = get_classifier()
        if not clf.is_ready():
            return {
                "status": "not_ready",
                "message": "Embeddings de referencia no cargados. Ejecuta primero la indexación DINOv2.",
                "top3": []
            }
        set_id = req.set_id or session_state.active_set_id
        top3 = clf.classify(crop, set_id=set_id)
    except Exception as e:
        print(f"[LegoVision API] Error en clasificación DINOv2: {e}")
        return {"status": "error", "message": str(e), "top3": []}

    # 3. Enriquecer resultados con metadatos del catálogo
    enriched = []
    for match in top3:
        part_ref = match["part_ref"]
        # Determinar color: preferir el detectado, o el del request, o fallback a blanco (15)
        detected_color = match.get("detected_color") or req.color_code or "15"
        
        # Resolver metadatos de pieza y color
        part_name, color_name, color_hex = get_part_info(part_ref, detected_color)
        
        enriched.append({
            "rank": match["rank"],
            "part_ref": part_ref,
            "part_name": part_name,
            "score": round(match["score"], 4),
            "face": match["face"],
            "angle": match["angle"],
            "detected_color": detected_color,
            "color_name": color_name,
            "color_hex": color_hex,
            "ref_image_b64": "",
        })

    latency_ms = (time.time() - start_time) * 1000.0

    return {
        "status": "ok",
        "latency_ms": latency_ms,
        "best_match": enriched[0] if enriched else None,
        "top3": enriched,
    }


# ══════════════════════════════════════════════════════════════════
# MODO DE INFERENCIA ESTÁTICA (REVIEW MODE)
# ══════════════════════════════════════════════════════════════════

import shutil
from fastapi.responses import FileResponse

TEMP_RUN_DIR = os.path.join(project_root, "data", "temp_inference_run")

@app.post("/inference-run/start")
def start_inference_run():
    """Limpia el directorio temporal y copia las primeras 10 imágenes de prueba."""
    if os.path.exists(TEMP_RUN_DIR):
        try:
            shutil.rmtree(TEMP_RUN_DIR)
        except Exception as e:
            print(f"[API Warning] No se pudo borrar la carpeta temporal: {e}")
    os.makedirs(TEMP_RUN_DIR, exist_ok=True)

    test_images_src = os.path.join(project_root, "data", "test_dataset", "images")
    if not os.path.exists(test_images_src):
        raise HTTPException(status_code=404, detail="Directorio de dataset de pruebas no encontrado.")

    png_files = sorted([f for f in os.listdir(test_images_src) if f.endswith(".png")])[:10]
    if not png_files:
        raise HTTPException(status_code=404, detail="No se encontraron imágenes en el dataset de pruebas.")

    for f in png_files:
        shutil.copy2(os.path.join(test_images_src, f), os.path.join(TEMP_RUN_DIR, f))

    return {"status": "started", "images": png_files}

@app.get("/inference-run/images")
def get_inference_run_images():
    """Retorna la lista de imágenes en la corrida temporal (excluyendo imágenes de depuración _clean.png)."""
    if not os.path.exists(TEMP_RUN_DIR):
        return {"images": []}
    files = sorted([f for f in os.listdir(TEMP_RUN_DIR) if f.endswith(".png") and not f.endswith("_clean.png")])
    return {"images": files}

@app.get("/inference-run/image/{filename}")
def get_inference_run_image(filename: str):
    """Devuelve el archivo de imagen especificado de la carpeta temporal."""
    file_path = os.path.join(TEMP_RUN_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    return FileResponse(file_path)

@app.post("/inference-run/detect/{filename}")
def detect_inference_run_image(filename: str):
    """Corre detección YOLOv8 sobre la imagen especificada del lote temporal, u obtiene el JSON guardado."""
    file_path = os.path.join(TEMP_RUN_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    
    # Intentar leer desde JSON pre-guardado
    json_path = os.path.join(TEMP_RUN_DIR, filename.replace(".png", ".json"))
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[API Warning] No se pudo leer JSON pre-guardado: {e}")

    start_time = time.time()
    try:
        img = Image.open(file_path).convert("RGB")
        detections = detector.detect(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en detección: {str(e)}")
        
    latency_ms = (time.time() - start_time) * 1000.0
    return {
        "latency_ms": latency_ms,
        "detections_count": len(detections),
        "detections": detections
    }

@app.post("/generate_inference_render")
def api_generate_inference_render(pieces_in_field: int = 30, set_id: str = "75078-1", is_rolling: bool = True):
    try:
        blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
        if not os.path.exists(blender_path):
            raise HTTPException(status_code=500, detail=f"Blender no encontrado en: {blender_path}")

        script_path = os.path.join(project_root, "scripts", "generate_inference_test_belt.py")
        if not os.path.exists(script_path):
            raise HTTPException(status_code=500, detail="Script no encontrado.")

        renders_dir = os.path.join(project_root, "data", "synthetic_renders")
        os.makedirs(renders_dir, exist_ok=True)
        set_tag = set_id.replace("-", "_")
        
        import time
        timestamp = int(time.time())
        output_filename = f"inference_test_{set_tag}_pf{pieces_in_field}_{timestamp}.png"
        output_path = os.path.join(renders_dir, output_filename)
        metadata_path = output_path.replace(".png", ".json")

        import subprocess, json
        cmd = [
            blender_path, "-b", "-P", script_path, "--",
            "--set_id", set_id,
            "--pieces_in_field", str(pieces_in_field),
            "--output_path", output_path,
            "--is_rolling", "true" if is_rolling else "false"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Blender falló: {result.stderr[-300:] if result.stderr else result.stdout[-300:]}")

        meta_data = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

        api_port = os.getenv("API_PORT", "8005")
        return {
            "status": "success",
            "image_url": f"http://localhost:{api_port}/renders/{output_filename}",
            "metadata": meta_data,
            "message": "Render generado con éxito."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/inference-render/{filename}")
def api_delete_inference_render(filename: str):
    try:
        renders_dir = os.path.join(project_root, "data", "synthetic_renders")
        png_path = os.path.join(renders_dir, filename)
        json_path = png_path.replace(".png", ".json")
        
        deleted = False
        if os.path.exists(png_path):
            os.remove(png_path)
            deleted = True
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted = True
            
        if deleted:
            return {"status": "success", "message": "Render eliminado."}
        else:
            raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_single_piece_renders")
def api_generate_single_piece_renders(set_id: str = "75078-1"):
    try:
        blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
        if not os.path.exists(blender_path):
            raise HTTPException(status_code=500, detail=f"Blender no encontrado en: {blender_path}")

        script_path = os.path.join(project_root, "scripts", "generate_single_piece_three_cameras.py")
        if not os.path.exists(script_path):
            raise HTTPException(status_code=500, detail="Script generate_single_piece_three_cameras.py no encontrado.")

        output_dir = os.path.join(project_root, "data", "synthetic_renders", "multicam")
        os.makedirs(output_dir, exist_ok=True)

        import subprocess, json
        cmd = [
            blender_path, "-b", "-P", script_path, "--",
            "--set_id", set_id,
            "--output_dir", output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Blender falló: {result.stderr[-300:] if result.stderr else result.stdout[-300:]}")

        metadata_path = os.path.join(output_dir, "multicam_metadata.json")
        meta_data = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

        return {
            "status": "success",
            "metadata": meta_data,
            "message": "Renders multicámara generados con éxito."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


STATIC_PART_HEIGHTS_MM = {
    # Bricks (height 9.6mm)
    "3001": 9.6, "3002": 9.6, "3003": 9.6, "3004": 9.6, "3005": 9.6, "3010": 9.6, "3622": 9.6, "3700": 9.6, "3701": 9.6,
    "2877": 9.6, "32000": 9.6, "3062": 9.6, "4070": 9.6,
    # Plates (height 3.2mm)
    "3020": 3.2, "3021": 3.2, "3022": 3.2, "3023": 3.2, "3024": 3.2, "3710": 3.2, "2420": 3.2, "2412": 3.2, "15573": 3.2,
    "60478": 3.2, "48336": 3.2, "6141": 3.2, "98138": 3.2, "59900": 3.2, "4032": 3.2, "11477": 3.2, "15068": 3.2,
    "85984": 3.2, "54200": 3.2, "99206": 3.2,
    # Tiles (height 3.2mm)
    "3068": 3.2, "3069": 3.2, "2431": 3.2, "6636": 3.2,
    # Slopes (height 9.6mm)
    "3039": 9.6, "3298": 9.6, "3037": 9.6, "3665": 9.6,
}

def load_part_heights_from_db():
    heights = dict(STATIC_PART_HEIGHTS_MM)
    try:
        with supabase_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT part_ref, MAX(lateral_height) as max_h FROM stable_poses GROUP BY part_ref")
                for row in cur.fetchall():
                    if row.get("max_h") is not None:
                        heights[row["part_ref"]] = float(row["max_h"])
        print(f"[LegoVision API] Loaded {len(heights)} part heights dynamically from database.")
    except Exception as e:
        print(f"[LegoVision API Warning] Failed to load part heights from DB: {e}. Using static catalog.")
    return heights

PART_HEIGHTS_MM = load_part_heights_from_db()

@app.post("/inference_multicam_set")
def api_inference_multicam_set(set_id: str = "75078-1"):
    try:
        clf = get_classifier()
        if not clf.is_ready():
            return {
                "status": "not_ready",
                "message": "Embeddings de referencia no cargados. Indexa DINOv2 primero.",
                "results": []
            }
            
        output_dir = os.path.join(project_root, "data", "synthetic_renders", "multicam")
        metadata_path = os.path.join(output_dir, "multicam_metadata.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Metadata de renders multicámara no encontrado. Genera renders primero.")
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            
        import torch
        # Liberación de memoria MPS al iniciar inferencia masiva
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        results = []
        correct_count = 0
        total_predictions = 0
        
        # Conteo para la tabla final
        inventory_stats = {}
        
        # Cargar catálogo de sets para obtener qty original
        from database.set_catalog import REAL_SETS
        set_data = REAL_SETS.get(set_id, {})
        
        # Inicializar stats del inventario original
        for p in set_data.get("parts", []):
            ref = p["ref"]
            if ref not in inventory_stats:
                inventory_stats[ref] = {
                    "ref": ref,
                    "name": p.get("name", "Pieza Lego"),
                    "qty_original": p.get("qty", 0),
                    "qty_detected": 0,
                    "correct_predictions": 0,
                    "total_predictions": 0
                }
        for fig in set_data.get("minifigures", []):
            ref = fig["ref"]
            if ref not in inventory_stats:
                inventory_stats[ref] = {
                    "ref": ref,
                    "name": fig.get("name", "Minifigura"),
                    "qty_original": fig.get("qty", 0),
                    "qty_detected": 0,
                    "correct_predictions": 0,
                    "total_predictions": 0
                }
                
        for render_entry in meta_data.get("renders", []):
            ref_gt = render_entry["ref"]
            color_hex = render_entry["color_hex"]
            color_code = render_entry["color_code"]
            name_gt = render_entry["name"]
            
            cameras_data = render_entry["cameras"]
            
            cam_results = {}
            for cam_name in ["cenital", "lateral_l", "lateral_r"]:
                cam_meta = cameras_data[cam_name]
                img_filename = cam_meta["file_name"]
                img_path = os.path.join(output_dir, img_filename)
                
                if not os.path.exists(img_path):
                    continue
                    
                # Cargar imagen y recortar
                img_full = Image.open(img_path).convert("RGB")
                w, h = img_full.size
                x1, y1, x2, y2 = cam_meta["bbox_norm"]
                
                # Clampar y recortar
                x1_px = max(0, min(int(x1 * w), w - 1))
                y1_px = max(0, min(int(y1 * h), h - 1))
                x2_px = max(x1_px + 1, min(int(x2 * w), w))
                y2_px = max(y1_px + 1, min(int(y2 * h), h))
                
                crop_img = img_full.crop((x1_px, y1_px, x2_px, y2_px))
                
                # Inferencia DINOv2
                top3 = clf.classify(crop_img, set_id=set_id, filter_by_color=False)
                
                # Modificar score del lateral usando restricción de altura
                if cam_name in ["lateral_l", "lateral_r"] and top3:
                    cen_meta = cameras_data.get("cenital")
                    if cen_meta:
                        cx1, cy1, cx2, cy2 = cen_meta["bbox_norm"]
                        cen_width_norm = cx2 - cx1
                        lat_height_norm = y2 - y1
                        if cen_width_norm > 0:
                            obs_ratio = lat_height_norm / cen_width_norm
                            
                            for cand in top3:
                                cand_ref = cand["part_ref"]
                                cand_height = PART_HEIGHTS_MM.get(cand_ref, 3.2)
                                
                                from inference.knn_classifier import FALLBACK_FOOTPRINT_MM
                                cand_dims = FALLBACK_FOOTPRINT_MM.get(cand_ref, (8.0, 8.0))
                                cand_width = max(cand_dims)
                                
                                theoretical_ratio = cand_height / cand_width
                                ratio_diff = abs(obs_ratio - theoretical_ratio)
                                
                                if ratio_diff > 0.25:
                                    cand["score"] = max(0.01, cand["score"] * 0.4)
                                elif ratio_diff < 0.1:
                                    cand["score"] = min(0.99, cand["score"] * 1.15)
                                    
                    top3.sort(key=lambda x: x["score"], reverse=True)
                
                if top3:
                    best = top3[0]
                    is_correct = (best["part_ref"] == ref_gt)
                    
                    cam_results[cam_name] = {
                        "predicted_ref": best["part_ref"],
                        "predicted_name": best["part_name"],
                        "score": best["score"],
                        "is_correct": is_correct,
                        "image_url": f"/renders/multicam/{img_filename}",
                        "bricklink_url": f"https://img.bricklink.com/ItemImage/PN/{best['detected_color'] or '15'}/{best['part_ref']}.png"
                    }
                    
                    total_predictions += 1
                    if is_correct:
                        correct_count += 1
                        
                    if ref_gt in inventory_stats:
                        inventory_stats[ref_gt]["total_predictions"] += 1
                        if is_correct:
                            inventory_stats[ref_gt]["correct_predictions"] += 1
                else:
                    cam_results[cam_name] = {
                        "predicted_ref": "Desconocido",
                        "predicted_name": "No clasificado",
                        "score": 0.0,
                        "is_correct": False,
                        "image_url": f"/renders/multicam/{img_filename}",
                        "bricklink_url": ""
                    }
            
            candidate_votes = {}
            weights = {"cenital": 0.4, "lateral_l": 0.3, "lateral_r": 0.3}
            
            for cam_name, r_cam in cam_results.items():
                ref_pred = r_cam["predicted_ref"]
                score = r_cam["score"]
                if ref_pred != "Desconocido":
                    candidate_votes[ref_pred] = candidate_votes.get(ref_pred, 0.0) + score * weights[cam_name]
                    
            if candidate_votes:
                consensus_ref = max(candidate_votes, key=candidate_votes.get)
                consensus_score = min(0.9999, candidate_votes[consensus_ref])
                
                consensus_name, _, _ = get_part_info(consensus_ref, color_code)
            else:
                consensus_ref = "Desconocido"
                consensus_name = "No identificado"
                consensus_score = 0.0
                
            if consensus_ref in inventory_stats:
                inventory_stats[consensus_ref]["qty_detected"] += 1
            elif consensus_ref != "Desconocido":
                inventory_stats[consensus_ref] = {
                    "ref": consensus_ref,
                    "name": consensus_name,
                    "qty_original": 0,
                    "qty_detected": 1,
                    "correct_predictions": 0,
                    "total_predictions": 0
                }
                
            results.append({
                "ref_gt": ref_gt,
                "name_gt": name_gt,
                "color_hex": color_hex,
                "color_code": color_code,
                "consensus_ref": consensus_ref,
                "consensus_name": consensus_name,
                "consensus_score": round(consensus_score, 4),
                "is_consensus_correct": (consensus_ref == ref_gt),
                "cameras": cam_results
            })
            
        mean_accuracy = (correct_count / total_predictions * 100) if total_predictions > 0 else 0.0
        
        worst_candidates = []
        for ref_key, stat in inventory_stats.items():
            tot = stat["total_predictions"]
            corr = stat["correct_predictions"]
            acc = (corr / tot * 100) if tot > 0 else 0.0
            stat["accuracy_pct"] = round(acc, 1)
            if stat["qty_original"] > 0:
                worst_candidates.append((ref_key, acc, stat["name"]))
                
        worst_candidates.sort(key=lambda x: x[1])
        worst_3_pieces = ", ".join([f"{item[0]} ({item[2]} · {item[1]:.0f}%)" for item in worst_candidates[:3]]) if worst_candidates else "Ninguna"
        
        inventory_list = list(inventory_stats.values())
        
        config.auto_release_memory()

        return {
            "status": "success",
            "mean_accuracy": round(mean_accuracy, 1),
            "worst_3_pieces": worst_3_pieces,
            "inventory": inventory_list,
            "results": results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



