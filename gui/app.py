import os
import sys
import json
import threading
import subprocess
import webview
import http.server
import socketserver

# Añadir directorio raíz al path para importar de database e inference
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv(override=True)

from database import supabase_client

# Hilo de entrenamiento activo (solo uno a la vez)
_training_thread: threading.Thread | None = None
_indexing_thread: threading.Thread | None = None



def _start_static_server(port=8006):
    """Inicia un servidor HTTP estático en gui/static/ para servir GLBs y otros assets."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    class SilentHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=static_dir, **kwargs)
        def log_message(self, format, *args):
            pass  # suppress logs
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()
    try:
        httpd = socketserver.TCPServer(("", port), SilentHandler)
        httpd.allow_reuse_address = True
        print(f"[LegoVision Static] Servidor estático en http://localhost:{port}/")
        httpd.serve_forever()
    except Exception as e:
        print(f"[LegoVision Static] Error iniciando servidor: {e}")


class ApiBridge:
    """Clase puente para comunicar JS en la interfaz con la lógica de Python."""

    # ------------------------------------------------------------------
    # Base de datos / estado
    # ------------------------------------------------------------------

    def get_historical_stats(self) -> dict:
        """Obtiene estadísticas generales históricas para mostrarlas en la GUI."""
        try:
            top_classes = supabase_client.get_top_classes(limit=5)
            recent = supabase_client.get_recent_detections(limit=10)

            # Formatear fechas para JSON
            for r in recent:
                if "detected_at" in r and r["detected_at"]:
                    r["detected_at"] = r["detected_at"].isoformat()

            return {
                "status": "success",
                "top_classes": top_classes,
                "recent_detections": recent,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_connection(self) -> bool:
        """Comprueba que la base de datos esté online."""
        return supabase_client.test_connection()

    # ------------------------------------------------------------------
    # Búsqueda de sets e inventario
    # ------------------------------------------------------------------

    def get_set_inventory(self, set_id: str) -> dict:
        """Obtiene el inventario de un set rápidamente desde la BD o catálogo estático."""
        try:
            # 1. Intentar obtener desde la base de datos
            set_data = supabase_client.get_set_from_db(set_id)
            
            if not set_data:
                # 2. Si no existe en la BD, buscar en catálogo y persistir
                print(f"[LegoVision GUI] Set {set_id} no encontrado en BD. Obteniendo de set_catalog y guardando...")
                from database import set_catalog
                set_data = set_catalog.get_set_data(set_id)
                supabase_client.save_set_to_db(set_id, set_data)
            
            for part in set_data.get("parts", []):
                ref = part["ref"]
                color_hex = part.get("color_hex", "#A0A5A9").replace("#", "")
                filename = f"render_{ref}_{color_hex}.png"
                render_path = os.path.join(project_root, "data", "synthetic_renders", filename)
                if os.path.exists(render_path):
                    part["image"] = f"http://localhost:8005/renders/{filename}"
                else:
                    part["image"] = ""
                    
            for fig in set_data.get("minifigures", []):
                ref = fig["ref"]
                color_hex = "F2F3F2"
                filename = f"render_{ref}_{color_hex}.png"
                render_path = os.path.join(project_root, "data", "synthetic_renders", filename)
                if os.path.exists(render_path):
                    fig["image"] = f"http://localhost:8005/renders/{filename}"
                else:
                    fig["image"] = ""
                
            return {
                "status": "success",
                "set_name": set_data["name"],
                "minifigures": set_data.get("minifigures", []),
                "parts": set_data.get("parts", []),
            }
        except Exception as e:
            print(f"[LegoVision GUI ERROR] get_set_inventory: {e}")
            return {"status": "error", "message": str(e)}

    def get_set_inventory_light(self, set_id: str) -> dict:
        """Obtiene el inventario de un set rápidamente sin generar renders 3D."""
        return self.get_set_inventory(set_id)

    # ------------------------------------------------------------------
    # Pipeline de Entrenamiento (Fase 1: YOLO)
    # ------------------------------------------------------------------

    def start_training(self, epochs: int = 15, dataset_size: int = 200, batch: int = 16) -> dict:
        """
        Inicia el pipeline de entrenamiento YOLO en un hilo de fondo.
        Retorna inmediatamente con el estado del hilo.
        """
        global _training_thread
        if _training_thread and _training_thread.is_alive():
            return {"status": "error", "message": "Ya hay un entrenamiento en curso."}

        def _run():
            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            script = os.path.join(project_root, "training", "train_yolo.py")
            cmd = [
                python_exec, script,
                "--epochs", str(int(epochs)),
                "--batch", str(int(batch)),
                "--dataset_size", str(int(dataset_size)),
            ]
            print(f"[LegoVision GUI] Lanzando entrenamiento: {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                proc.wait()
                print(f"[LegoVision GUI] Entrenamiento finalizado. Código: {proc.returncode}")
            except Exception as ex:
                print(f"[LegoVision GUI ERROR] Entrenamiento fallido: {ex}")

        _training_thread = threading.Thread(target=_run, daemon=True)
        _training_thread.start()
        return {"status": "started", "message": "Entrenamiento iniciado en segundo plano."}

    def start_indexing(self, set_id: str = "50SIMPLE-1") -> dict:
        """
        Inicia el proceso de indexación DINOv2 (embedding extraction de renders ya existentes).
        Ejecuta en un hilo de fondo.
        """
        global _indexing_thread
        if _indexing_thread and _indexing_thread.is_alive():
            return {"status": "error", "message": "Ya hay un proceso de indexación en curso."}

        def _run():
            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable

            # Indexar embeddings DINOv2 usando renders existentes o imágenes de referencia
            idx_script = os.path.join(project_root, "training", "index_synthetic_renders.py")
            print("[LegoVision GUI] Indexando embeddings DINOv2...")
            proc2 = subprocess.Popen(
                [python_exec, idx_script],
                cwd=project_root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc2.stdout:
                sys.stdout.write(line); sys.stdout.flush()
            proc2.wait()
            print(f"[LegoVision GUI] Indexación completada. Código: {proc2.returncode}")

        _indexing_thread = threading.Thread(target=_run, daemon=True)
        _indexing_thread.start()
        return {"status": "started", "message": "Indexación DINOv2 iniciada en segundo plano."}

    def get_training_status(self) -> dict:
        """
        Consulta la corrida de entrenamiento más reciente en la BD y devuelve su estado.
        Utilizado por la GUI para actualizar gráficos y logs en tiempo real.
        """
        try:
            run = supabase_client.get_latest_training_run()
            if not run:
                return {"status": "idle", "message": "No hay entrenamientos registrados."}

            # Serializar fechas
            for key in ("started_at", "ended_at"):
                if run.get(key):
                    run[key] = run[key].isoformat() if hasattr(run[key], "isoformat") else str(run[key])

            # Añadir estado de hilo
            run["thread_alive"] = bool(_training_thread and _training_thread.is_alive())
            run["index_alive"] = bool(_indexing_thread and _indexing_thread.is_alive())
            return {"status": "ok", "run": run}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def is_training_active(self) -> bool:
        """Devuelve True si hay un hilo de entrenamiento corriendo activamente."""
        return bool(_training_thread and _training_thread.is_alive())

    def is_indexing_active(self) -> bool:
        """Devuelve True si hay un hilo de indexación corriendo activamente."""
        return bool(_indexing_thread and _indexing_thread.is_alive())

    def get_embedding_count(self) -> dict:
        """Cuenta cuántos embeddings de referencia hay en la BD."""
        try:
            count = supabase_client.count_embeddings()
            return {"status": "ok", "count": count}
        except Exception as e:
            return {"status": "error", "count": 0, "message": str(e)}



    # ------------------------------------------------------------------
    # Minifiguras — Ensamblaje 3D
    # ------------------------------------------------------------------

    def get_minifigures_for_sets(self) -> dict:
        try:
            sys.path.insert(0, project_root)
            from scripts.assemble_minifig import get_minifigs_from_test_sets, MINIFIG_DATABASE
            minifigs = get_minifigs_from_test_sets()
            return {"status": "success", "minifigures": minifigs}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_minifig_components(self, minifig_ref: str) -> dict:
        try:
            from scripts.assemble_minifig import MINIFIG_DATABASE
            if minifig_ref not in MINIFIG_DATABASE:
                return {"status": "error", "message": "Minifigura no encontrada"}
            config = MINIFIG_DATABASE[minifig_ref]
            # Deep copy to avoid mutating the original database
            import copy
            components = copy.deepcopy(config["components"])
            for comp in components:
                part_ref = comp["part_file"].replace(".dat", "")
                ldraw_color = comp["ldraw_color"]
                color_hex = comp["color_hex"].replace("#", "")
                # 1. Try local synthetic render first
                filename = "render_" + part_ref + "_" + color_hex + ".png"
                render_path = os.path.join(project_root, "data", "synthetic_renders", filename)
                if os.path.exists(render_path):
                    comp["image"] = "http://localhost:8005/renders/" + filename
                    comp["image_source"] = "local_render"
                else:
                    # 2. Fallback: BrickLink public image API
                    # URL format: https://img.bricklink.com/ItemImage/PN/{color_id}/{part_ref}.png
                    # BrickLink uses their own color IDs — use ldraw_color as approximation
                    bl_url = f"https://img.bricklink.com/ItemImage/PN/{ldraw_color}/{part_ref}.png"
                    comp["image"] = bl_url
                    comp["image_source"] = "bricklink"
            return {"status": "success", "name": config["name"], "components": components}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_minifig_assembly_status(self, minifig_ref: str) -> dict:
        try:
            glb_path = os.path.join(project_root, "gui", "static", "models", minifig_ref + ".glb")
            glb_exists = os.path.exists(glb_path)
            db_record = None
            try:
                db_record = supabase_client.get_minifig_assembly(minifig_ref)
            except Exception:
                pass
            return {
                "status": "success",
                "assembled": glb_exists,
                "in_db": db_record is not None,
                "glb_url": "http://localhost:8006/models/" + minifig_ref + ".glb" if glb_exists else None,
                "db_record": db_record
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def assemble_minifig(self, minifig_ref: str) -> dict:
        import threading
        def _run():
            try:
                from scripts.assemble_minifig import build_minifig
                result = build_minifig(minifig_ref, save_to_db=True)
                print("[LegoVision GUI] Ensamblaje completado:", result)
            except Exception as e:
                print("[LegoVision GUI ERROR] assemble_minifig:", e)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"status": "started", "message": "Ensamblaje iniciado en segundo plano para " + minifig_ref}

    def simulate_physics_scatter(self, part_ref: str, color_hex: str) -> dict:
        """
        Ejecuta la simulación de física de caída y estabilización en Blender
        para 15 piezas de la referencia y color especificados.
        """
        try:
            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            if not os.path.exists(blender_path):
                return {
                    "status": "error",
                    "message": f"Ejecutable de Blender no encontrado en la ruta: {blender_path}. Por favor, configúralo en el archivo .env."
                }

            script_path = os.path.join(project_root, "scripts", "physics_belt_generator.py")
            clean_color = color_hex.replace("#", "")
            output_filename = f"physics_scatter_{part_ref}_{clean_color}.png"
            output_path = os.path.join(project_root, "data", "synthetic_renders", output_filename)
            
            # Asegurar que el directorio de destino exista
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # FIX-D: cache hit — si los 15 crops ya existen, devolver inmediatamente
            crops_dir = os.path.dirname(output_path)
            cached_crops = []
            for ci in range(15):
                cp = os.path.join(crops_dir, f"physics_scatter_{part_ref}_{clean_color}_crop_{ci}.png")
                if os.path.exists(cp):
                    cached_crops.append(f"http://localhost:8005/renders/physics_scatter_{part_ref}_{clean_color}_crop_{ci}.png")
            if len(cached_crops) == 15:
                print(f"[LegoVision Physics] Cache hit: 15 crops ya existen para {part_ref}_{clean_color}.")
                return {
                    "status": "success",
                    "image_url": cached_crops[0],
                    "crops": cached_crops,
                    "output_path": os.path.join(crops_dir, f"physics_scatter_{part_ref}_{clean_color}_crop_0.png"),
                    "message": f"Crops cargados desde cache (15 vistas de {part_ref})."
                }

            cmd = [
                blender_path,
                "-b",
                "-P", script_path,
                "--",
                "--part_ref", part_ref,
                "--color_hex", color_hex,
                "--output_path", output_path
            ]
            
            print(f"[LegoVision Physics] Lanzando simulación física: {' '.join(cmd)}")
            
            # Ejecutar el subproceso bloqueante para esta petición del UX
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[LegoVision Physics ERROR] Blender falló con código {result.returncode}")
                print(f"Stdout:\n{result.stdout[-1000:]}")
                print(f"Stderr:\n{result.stderr[-1000:]}")
                return {
                    "status": "error",
                    "message": f"Blender falló con código de salida {result.returncode}",
                    "details": result.stderr[-500:] if result.stderr else "Sin detalles de error."
                }
                
            crop0_filename = f"physics_scatter_{part_ref}_{clean_color}_crop_0.png"
            crop0_path = os.path.join(os.path.dirname(output_path), crop0_filename)
            
            if not os.path.exists(crop0_path):
                return {
                    "status": "error",
                    "message": "La simulación finalizó pero no se generaron los recortes individuales de piezas esperados."
                }
                
            # Cargar la lista de recortes pre-renderizados a Z=10
            crops = []
            for i in range(15):
                crop_filename = f"physics_scatter_{part_ref}_{clean_color}_crop_{i}.png"
                crop_path = os.path.join(os.path.dirname(output_path), crop_filename)
                if os.path.exists(crop_path):
                    crops.append(f"http://localhost:8005/renders/{crop_filename}")
            
            # Devolver URL de la imagen (fallback al crop 0) y los renders del carrusel
            image_url = f"http://localhost:8005/renders/{crop0_filename}"
            return {
                "status": "success",
                "image_url": image_url,
                "crops": crops,
                "output_path": crop0_path,
                "message": f"Simulación física (caída a 5cm) y renderizado a Z=10 completados para 15 vistas de plano."
            }
        except Exception as e:
            print(f"[LegoVision Physics ERROR] Excepción: {e}")
            return {"status": "error", "message": str(e)}

def main():
    # Obtener la ruta del archivo HTML estático
    gui_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(gui_dir, "static", "index.html")

    if not os.path.exists(html_path):
        print(f"[LegoVision GUI ERROR] No se encontró el archivo HTML en: {html_path}")
        sys.exit(1)

    print("[LegoVision GUI] Lanzando ventana de PyWebView...")

    bridge = ApiBridge()

    # Crear ventana (diseño de 1280x800, responsivo, modo oscuro nativo)
    window = webview.create_window(
        title="LegoVision — Panel de Clasificación y Visión Artificial",
        url=html_path,
        js_api=bridge,
        width=1280,
        height=820,
        resizable=True,
        min_size=(1000, 600),
    )

    # Arrancar servidor estático para GLBs y assets en puerto 8006
    static_thread = threading.Thread(target=_start_static_server, args=(8006,), daemon=True)
    static_thread.start()

    # Arrancar bucle de interfaz
    webview.start(debug=True)


if __name__ == "__main__":
    main()
