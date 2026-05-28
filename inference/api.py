import os
import sys
import time
import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import List, Optional
from dotenv import load_dotenv

# Añadir directorio raíz al path para importar desde database y detector
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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

# Inicializar detector global
MODEL_PATH = os.getenv("MODEL_PATH", "./runs/train/best.pt")
DEVICE = os.getenv("MODEL_DEVICE", None)
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

detector = LegoDetector(model_path=MODEL_PATH, device=DEVICE, conf_threshold=CONF_THRESHOLD)

# Clasificador DINOv2 (carga lazy en primera petición)
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        from inference.classifier import LegoClassifier
        _classifier = LegoClassifier(top_k=3)
        _classifier.load_model()
        _classifier.load_reference_embeddings()
    return _classifier

# Estado de la sesión actual de la cinta transportadora
class SessionState:
    def __init__(self):
        self.active_session_id = None
        self.start_time = None
        self.frame_count = 0
        self.total_latency_ms = 0.0
        self.confidences = []

session_state = SessionState()

class SessionStartRequest(BaseModel):
    model_version: str = "yolov8n_synthetic"
    belt_speed_mm_s: float = 83.3

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

        session_id = supabase_client.create_session(
            model_version=req.model_version,
            belt_speed_mm_s=req.belt_speed_mm_s
        )
        session_state.active_session_id = session_id
        session_state.start_time = time.time()
        session_state.frame_count = 0
        session_state.total_latency_ms = 0.0
        session_state.confidences = []
        
        print(f"[LegoVision API] Sesión de inferencia iniciada: {session_id}")
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
            
        supabase_client.close_session(
            session_id=session_state.active_session_id,
            avg_fps=avg_fps,
            avg_confidence=avg_confidence
        )
        
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
    conf: float = None
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

    crop = frame_img.crop((x1, y1, x2, y2))

    # 2. Clasificar con DINOv2
    try:
        clf = get_classifier()
        if not clf.is_ready():
            return {
                "status": "not_ready",
                "message": "Embeddings de referencia no cargados. Ejecuta primero la indexación DINOv2.",
                "top3": []
            }
        top3 = clf.classify(crop)
    except Exception as e:
        print(f"[LegoVision API] Error en clasificación DINOv2: {e}")
        return {"status": "error", "message": str(e), "top3": []}

    # 3. Enriquecer resultados con metadatos del catálogo e imagen de referencia de la BD
    enriched = []
    for match in top3:
        part_ref = match["part_ref"]
        # Determinar color: preferir el detectado, o el del request, o fallback a blanco (15)
        detected_color = match.get("detected_color") or req.color_code or "15"
        
        # Resolver metadatos de pieza y color
        part_name, color_name, color_hex = get_part_info(part_ref, detected_color)
        
        # Recuperar render de la BD
        ref_img_b64 = supabase_client.get_part_render(part_ref, detected_color) or ""
        
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
            "ref_image_b64": ref_img_b64,
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
    """Retorna la lista de imágenes en la corrida temporal."""
    if not os.path.exists(TEMP_RUN_DIR):
        return {"images": []}
    files = sorted([f for f in os.listdir(TEMP_RUN_DIR) if f.endswith(".png")])
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


