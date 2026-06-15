import os
import sys
import json
import threading
import subprocess
import webview
import http.server
import socketserver
import re
import glob

# Progreso de generación de imágenes sintéticas YOLO
_generation_progress = {"current": 0, "total": 0, "active": False}


# Añadir directorio raíz al path para importar de database e inference
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv(override=True)

from database import supabase_client

# Hilo de entrenamiento activo (solo uno a la vez)
_training_thread: threading.Thread | None = None
_indexing_thread: threading.Thread | None = None
_validation_thread: threading.Thread | None = None

# Referencias a subprocesos activos para stop
_training_proc1 = None
_training_proc2 = None
_indexing_proc1 = None
_indexing_proc2 = None
_validation_proc = None
_stop_training_flag: bool = False
_stop_indexing_flag: bool = False
_stop_validation_flag: bool = False

# Estado de progreso de validación de estabilidad
_validation_progress = {
    "current": 0,
    "total": 41,
    "current_piece": "",
    "active": False,
    "done": False,
    "error": "",
    "results": None
}

# Estado de progreso de simulación FOV DINOv2
_dino_fov_progress = {
    "current": 0,
    "total": 0,
    "current_file": "",
    "active": False,
    "done": False,
    "error": "",
    "phase": "idle",
}
_dino_fov_thread = None
_dino_fov_proc1 = None
_dino_fov_proc2 = None
_stop_dino_fov_flag = False


def _kill_proc(proc, name: str):
    """Mata un subproceso de forma agresiva con SIGKILL al grupo de procesos completo.
    Usar killpg garantiza que los procesos hijo de Blender (render workers) tambien mueren.
    """
    if proc is None:
        return
    try:
        if proc.poll() is None:
            try:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                print(f'[LegoVision GUI] KILL grupo procesos {name} (pid={proc.pid})')
            except Exception:
                proc.kill()
                print(f'[LegoVision GUI] KILL {name} (pid={proc.pid})')
    except Exception as e:
        print(f'[LegoVision GUI] Error matando {name}: {e}')

# Estado de progreso de indexacion DINOv2 (compartido entre hilo y UI)
_indexing_progress = {"current": 0, "total": 0, "current_file": "", "active": False, "done": False, "error": ""}



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
    def __init__(self):
        self.api_port = os.getenv("API_PORT", "8005")

    def get_api_base(self) -> str:
        return f"http://localhost:{self.api_port}"

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
                    part["image"] = f"http://localhost:{self.api_port}/renders/{filename}"
                else:
                    part["image"] = ""
                    
            for fig in set_data.get("minifigures", []):
                ref = fig["ref"]
                color_hex = "F2F3F2"
                filename = f"render_{ref}_{color_hex}.png"
                render_path = os.path.join(project_root, "data", "synthetic_renders", filename)
                if os.path.exists(render_path):
                    fig["image"] = f"http://localhost:{self.api_port}/renders/{filename}"
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

    def start_training(self, epochs: int = 50, dataset_size: int = 500, batch: int = 16, pieces_per_image: int = 25, empty_ratio: float = 5.0) -> dict:
        """Pipeline completo: 1) Genera dataset con Blender (si no existe con los mismos parámetros), 2) Entrena YOLO11. Set fijo: 75078-1."""
        global _training_thread
        if _training_thread and _training_thread.is_alive():
            return {"status": "error", "message": "Ya hay un entrenamiento en curso."}

        def _run():
            global _training_proc1, _training_proc2, _stop_training_flag
            _stop_training_flag = False
            import time
            
            # Registrar inicio de entrenamiento en la Base de Datos
            config_dict = {
                "batch": batch,
                "dataset_size": dataset_size,
                "pieces_per_image": pieces_per_image,
                "empty_ratio": empty_ratio,
                "set_id": "75078-1"
            }
            try:
                run_id = supabase_client.create_training_run(epochs, config_dict)
                print(f"[LegoVision GUI] Corrida registrada en BD con ID: {run_id}")
            except Exception as db_err:
                print(f"[LegoVision GUI ERROR] No se pudo registrar la corrida en la BD: {db_err}")
                run_id = None

            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            blender_exec = "/opt/homebrew/bin/blender"
            if not os.path.exists(blender_exec):
                blender_exec = "blender"

            raw_dataset_dir = os.path.join(project_root, "data", "raw_dataset")
            images_dir = os.path.join(raw_dataset_dir, "images")
            labels_dir = os.path.join(raw_dataset_dir, "labels")
            metadata_path = os.path.join(raw_dataset_dir, "dataset_metadata.json")

            # Helper para escribir logs de forma eficiente (bufeado)
            log_buffer = []
            last_write_time = [time.time()]

            def append_buffered_log(line):
                log_buffer.append(line)
                now = time.time()
                if now - last_write_time[0] >= 2.0 or len(log_buffer) >= 10:
                    text = "".join(log_buffer)
                    log_buffer.clear()
                    last_write_time[0] = now
                    if run_id:
                        try:
                            supabase_client.append_training_log(run_id, text)
                        except Exception as e:
                            print(f"[LegoVision GUI DB WARN] Error escribiendo log: {e}")

            def flush_buffered_log():
                if log_buffer:
                    text = "".join(log_buffer)
                    log_buffer.clear()
                    if run_id:
                        try:
                            supabase_client.append_training_log(run_id, text)
                        except Exception as e:
                            print(f"[LegoVision GUI DB WARN] Error en flush log: {e}")

            # Verificar si disponemos de los renders generados para el set 75078-1
            skip_generation = False
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    # Comprobar que los parámetros coincidan
                    param_match = (
                        meta.get("set_id") == "75078-1" and
                        meta.get("dataset_size") == dataset_size and
                        meta.get("pieces_per_image") == pieces_per_image and
                        abs(meta.get("empty_ratio", 0.0) - empty_ratio) < 1e-4
                    )
                    
                    # Comprobar que los archivos existan
                    if param_match and os.path.exists(images_dir):
                        png_files = glob.glob(os.path.join(images_dir, "*.png"))
                        if len(png_files) >= dataset_size * 0.9:
                            skip_generation = True
                except Exception as meta_err:
                    print(f"[LegoVision GUI WARN] Error leyendo metadatos del dataset: {meta_err}")

            if skip_generation:
                log_msg = "> Dataset local coincide con los parámetros de la interfaz. Omitiendo generación de imágenes.\n"
                print(f"[LegoVision GUI] {log_msg.strip()}")
                if run_id:
                    supabase_client.append_training_log(run_id, log_msg)
            else:
                # Limpiar directorios
                print("[LegoVision GUI] Parámetros no coinciden o dataset incompleto. Limpiando directorios y regenerando...")
                log_clean = "> Parámetros cambiados o dataset inexistente. Limpiando y regenerando dataset...\n"
                if run_id:
                    supabase_client.append_training_log(run_id, log_clean)
                
                for d in (images_dir, labels_dir):
                    if os.path.exists(d):
                        for f in glob.glob(os.path.join(d, "*")):
                            try:
                                if os.path.isfile(f):
                                    os.remove(f)
                            except Exception as e:
                                print(f"Error borrando {f}: {e}")
                    os.makedirs(d, exist_ok=True)
                
                if os.path.exists(metadata_path):
                    try:
                        os.remove(metadata_path)
                    except Exception:
                        pass

                pieces_exact = max(1, int(pieces_per_image))
                empty_ratio_clamped = max(0.0, min(50.0, float(empty_ratio))) / 100.0
                gen_script = os.path.join(project_root, "scripts", "generate_yolo_training_dataset.py")
                
                gen_cmd = [blender_exec, "-b", "-P", gen_script, "--",
                    "--set_id", "75078-1", "--num_frames", str(int(dataset_size)),
                    "--output_dir", raw_dataset_dir,
                    "--pieces", str(pieces_exact),
                    "--empty_ratio", str(empty_ratio_clamped)]
                
                print(f"[LegoVision GUI] PASO 1: Generando {dataset_size} imgs ({pieces_exact} piezas/img exactas, {empty_ratio:.0f}% vacias)...")
                _generation_progress.update({"current": 0, "total": dataset_size, "active": True})
                
                try:
                    _training_proc1 = subprocess.Popen(
                        gen_cmd, cwd=project_root,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, start_new_session=True,
                    )
                    for line in _training_proc1.stdout:
                        if _stop_training_flag:
                            _kill_proc(_training_proc1, 'Blender/Dataset')
                            print('[LegoVision] STOP: generacion de dataset cancelada.')
                            _generation_progress["active"] = False
                            if run_id:
                                supabase_client.complete_training_run(run_id, "failed", "⏹ Cancelado por el usuario durante generación.\n")
                            return
                        
                        sys.stdout.write(line); sys.stdout.flush()
                        append_buffered_log(line)
                        
                        # Parsear progreso de frames
                        frame_match = re.search(r'frame\s+(\d+)', line)
                        if frame_match:
                            current_frame = int(frame_match.group(1))
                            _generation_progress["current"] = min(current_frame, dataset_size)

                    _training_proc1.wait()
                    _generation_progress["active"] = False
                    flush_buffered_log()
                    
                    if _training_proc1.returncode != 0:
                        print(f"[LegoVision GUI WARN] Generacion codigo {_training_proc1.returncode}. Continuando con dataset existente...")
                    else:
                        # Guardar metadatos exitosos
                        try:
                            meta_data = {
                                "set_id": "75078-1",
                                "dataset_size": dataset_size,
                                "pieces_per_image": pieces_per_image,
                                "empty_ratio": empty_ratio
                            }
                            with open(metadata_path, "w", encoding="utf-8") as f:
                                json.dump(meta_data, f, indent=4)
                        except Exception as meta_save_err:
                            print(f"[LegoVision GUI ERROR] No se pudo guardar dataset_metadata.json: {meta_save_err}")

                except Exception as ex:
                    _generation_progress["active"] = False
                    print(f"[LegoVision GUI ERROR] Generacion: {ex}. Usando dataset existente...")
                    if run_id:
                        supabase_client.append_training_log(run_id, f"ERROR Generacion: {ex}\n")

            # Chequear stop antes de iniciar el entrenamiento YOLO
            if _stop_training_flag:
                print('[LegoVision] STOP: entrenamiento cancelado antes de iniciar YOLO.')
                if run_id:
                    supabase_client.complete_training_run(run_id, "failed", "⏹ Cancelado antes de iniciar YOLO.\n")
                return

            train_script = os.path.join(project_root, "training", "train_yolo.py")
            train_cmd = [python_exec, train_script,
                "--epochs", str(int(epochs)), "--batch", str(int(batch)),
                "--dataset_size", str(int(dataset_size)), "--set_id", "75078-1"]
            if run_id:
                train_cmd.extend(["--run_id", run_id])

            print(f"[LegoVision GUI] PASO 2: Entrenando YOLO11 ({epochs} epocas, batch={batch})...")
            try:
                _training_proc2 = subprocess.Popen(
                    train_cmd, cwd=project_root,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True,
                )
                for line in _training_proc2.stdout:
                    if _stop_training_flag:
                        _kill_proc(_training_proc2, 'YOLO Train')
                        print('[LegoVision] STOP: entrenamiento YOLO cancelado.')
                        if run_id:
                            supabase_client.complete_training_run(run_id, "failed", "⏹ Cancelado durante entrenamiento.\n")
                        return
                    sys.stdout.write(line); sys.stdout.flush()
                    append_buffered_log(line)
                _training_proc2.wait()
                flush_buffered_log()
                print(f"[LegoVision GUI] Entrenamiento finalizado. Codigo: {_training_proc2.returncode}")
            except Exception as ex:
                print(f"[LegoVision GUI ERROR] Entrenamiento: {ex}")
                if run_id:
                    supabase_client.complete_training_run(run_id, "failed", f"ERROR Entrenamiento: {ex}\n")

        _training_thread = threading.Thread(target=_run, daemon=True)
        _training_thread.start()
        return {"status": "started", "message": f"Pipeline: generando {dataset_size} imgs ({pieces_per_image}+/-5 piezas) + entrenando {epochs} epocas."}



    def start_indexing(self, set_id: str = "75078-1") -> dict:
        """DINOv2: 1) Genera refs multi-angulo con fisica Blender, 2) Indexa embeddings ViT-S/14."""
        global _indexing_thread
        if _indexing_thread and _indexing_thread.is_alive():
            return {"status": "error", "message": "Ya hay un proceso de indexacion en curso."}

        def _run():
            global _indexing_proc1, _indexing_proc2, _indexing_progress
            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            blender_exec = "/opt/homebrew/bin/blender"
            if not os.path.exists(blender_exec):
                blender_exec = "blender"
            ref_script = os.path.join(project_root, "scripts", "generate_physics_ref_multiangle.py")
            os.makedirs(os.path.join(project_root, "data", "ref_multiangle"), exist_ok=True)
            _indexing_progress = {"current": 0, "total": 100, "current_file": "Generando refs multi-angulo...", "active": True, "done": False, "error": ""}
            print(f"[LegoVision GUI] PASO 1: Generando referencias DINOv2 con fisica real (Blender)...")
            try:
                # start_new_session=True: grupo de procesos propio para matar Blender + hijos con killpg
                _indexing_proc1 = subprocess.Popen(
                    [blender_exec, "-b", "-P", ref_script],
                    cwd=project_root,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True,
                )
                for line in _indexing_proc1.stdout:
                    sys.stdout.write(line); sys.stdout.flush()
                    ls = line.strip()
                    if "[OK]" in ls or "Guardado" in ls or "guardada" in ls:
                        _indexing_progress["current"] = min(_indexing_progress["current"] + 1, 80)
                        _indexing_progress["current_file"] = ls[:80]
                    # Chequear flag durante la generacion de refs
                    if _stop_indexing_flag:
                        _kill_proc(_indexing_proc1, 'Blender/Refs')
                        _indexing_progress.update({'active': False, 'error': 'Cancelado por el usuario.'})
                        return
                _indexing_proc1.wait()
                print(f"[LegoVision GUI] Refs generadas. Codigo: {_indexing_proc1.returncode}")
            except Exception as ex:
                print(f"[LegoVision GUI WARN] Generacion refs: {ex}. Usando refs existentes...")

            # Chequear stop antes de iniciar indexacion
            if _stop_indexing_flag:
                _indexing_progress.update({'active': False, 'error': 'Cancelado por el usuario.'})
                return

            import glob
            multiangle_dir = os.path.join(project_root, "data", "ref_multiangle")
            renders_dir = os.path.join(project_root, "data", "synthetic_renders")
            total_files = (len(glob.glob(os.path.join(multiangle_dir, "ref_*.png"))) +
                           len(glob.glob(os.path.join(renders_dir, "render_*.png"))))
            _indexing_progress["total"] = max(1, total_files)
            _indexing_progress["current_file"] = "Indexando embeddings DINOv2..."
            print(f"[LegoVision GUI] PASO 2: Indexando {total_files} imagenes con DINOv2 ViT-S/14...")
            idx_script = os.path.join(project_root, "training", "index_synthetic_renders.py")
            cmd = [python_exec, idx_script, "--set_id_filter", set_id]
            _indexing_proc2 = subprocess.Popen(
                cmd, cwd=project_root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            for line in _indexing_proc2.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                ls = line.strip()
                if ls.startswith("OK") or "WRITTEN" in ls or ("[" in ls and "]" in ls):
                    _indexing_progress["current"] = min(_indexing_progress["current"] + 1, _indexing_progress["total"])
                    _indexing_progress["current_file"] = ls[:80]
                if _stop_indexing_flag:
                    _kill_proc(_indexing_proc2, 'DINOv2 Index')
                    _indexing_progress.update({'active': False, 'error': 'Cancelado por el usuario.'})
                    return
            _indexing_proc2.wait()
            if _indexing_proc2.returncode == 0:
                _indexing_progress.update({"current": _indexing_progress["total"], "done": True, "active": False})
            else:
                _indexing_progress.update({"error": f"Error codigo {_indexing_proc2.returncode}", "active": False})
            print(f"[LegoVision GUI] Indexacion completada. Codigo: {_indexing_proc2.returncode}")

        _indexing_thread = threading.Thread(target=_run, daemon=True)
        _indexing_thread.start()
        return {"status": "started", "message": f"Indexacion DINOv2 iniciada (set {set_id}): genera refs + embeddings en segundo plano."}


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

            # Si la generación de imágenes está activa, añadir detalles de progreso y forzar estado "generating"
            if _generation_progress.get("active"):
                run["status"] = "generating"
                run["generation_current"] = _generation_progress["current"]
                run["generation_total"] = _generation_progress["total"]
                run["generation_pct"] = round((_generation_progress["current"] / max(1, _generation_progress["total"])) * 100, 1)

            return {"status": "ok", "run": run}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    def is_training_active(self) -> bool:
        """Devuelve True si hay un hilo de entrenamiento corriendo activamente."""
        return bool(_training_thread and _training_thread.is_alive())

    def is_indexing_active(self) -> bool:
        """Devuelve True si hay un hilo de indexación corriendo activamente."""
        return bool(_indexing_thread and _indexing_thread.is_alive())
    def stop_training(self) -> dict:
        """Detiene el entrenamiento YOLO en curso (dataset Blender o YOLO train).
        Usa _kill_proc (SIGKILL al grupo) para garantizar que Blender y sus hijos mueren.
        """
        global _stop_training_flag
        _stop_training_flag = True
        killed = []
        for proc, name in [(_training_proc1, 'Blender/Dataset'), (_training_proc2, 'YOLO Train')]:
            if proc and proc.poll() is None:
                _kill_proc(proc, name)
                killed.append(name)
        if killed:
            return {'status': 'stopped', 'message': 'Detenido: ' + ', '.join(killed)}
        return {'status': 'idle', 'message': 'No habia entrenamiento activo.'}

    def stop_indexing(self) -> dict:
        """Detiene la indexacion DINOv2 en curso.
        Usa _kill_proc (SIGKILL al grupo) para garantizar que Blender y sus hijos mueren.
        """
        global _stop_indexing_flag
        _stop_indexing_flag = True
        killed = []
        for proc, name in [(_indexing_proc1, 'Blender/Refs'), (_indexing_proc2, 'DINOv2 Index')]:
            if proc and proc.poll() is None:
                _kill_proc(proc, name)
                killed.append(name)
        _indexing_progress.update({'active': False, 'error': 'Cancelado por el usuario.'})
        if killed:
            return {'status': 'stopped', 'message': 'Detenido: ' + ', '.join(killed)}
        return {'status': 'idle', 'message': 'No habia indexacion activa.'}


    def get_embedding_count(self) -> dict:
        """Cuenta cuántos embeddings de referencia hay en la BD."""
        try:
            count = supabase_client.count_embeddings()
            return {"status": "ok", "count": count}
        except Exception as e:
            return {"status": "error", "count": 0, "message": str(e)}

    def get_indexing_progress(self) -> dict:
        """Devuelve el progreso actual de la indexacion DINOv2."""
        pct = 0
        if _indexing_progress["total"] > 0:
            pct = round((_indexing_progress["current"] / _indexing_progress["total"]) * 100, 1)
        return {
            "current": _indexing_progress["current"],
            "total": _indexing_progress["total"],
            "pct": pct,
            "current_file": _indexing_progress.get("current_file", ""),
            "active": _indexing_progress["active"],
            "done": _indexing_progress["done"],
            "error": _indexing_progress.get("error", ""),
        }

    def get_eval_results(self) -> dict:
        """Devuelve los resultados de la ultima evaluacion post-entrenamiento (Opcion C)."""
        try:
            eval_path = os.path.join(project_root, "data", "eval_results_latest.json")
            if not os.path.exists(eval_path):
                return {"status": "not_available", "message": "No hay evaluacion disponible. Entrena el modelo primero."}
            import json as _json
            with open(eval_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if "error" in data:
                return {"status": "error", "message": data["error"]}
            return {"status": "ok", "eval": data}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_training_scope_info(self) -> dict:
        """Devuelve estadisticas del universo de piezas segun el scope seleccionado."""
        try:
            sys.path.insert(0, os.path.join(project_root, 'database'))
            from set_catalog import REAL_SETS
            # Calcular para set 75078-1
            set_data = REAL_SETS.get('75078-1', {})
            yolo_single = len(set(p['ref'] for p in set_data.get('parts', []) if 'stk' not in p['ref'].lower()))
            dino_single = len(set((p['ref'], p.get('color_hex','#A0A5A9')) for p in set_data.get('parts', []) if 'stk' not in p['ref'].lower()))
            # Calcular para todos los sets
            all_yolo = set()
            all_dino = set()
            for sid, sd in REAL_SETS.items():
                for pp in sd.get('parts', []):
                    ref = pp['ref']
                    if 'stk' not in ref.lower() and 'pb' not in ref.lower() and len(ref) < 15:
                        all_yolo.add(ref)
                        all_dino.add((ref, pp.get('color_hex', '#A0A5A9')))
            return {
                'status': 'ok',
                'single_set': {
                    'set_id': '75078-1',
                    'name': set_data.get('name', 'Imperial Troop Transport'),
                    'yolo_geometries': yolo_single,
                    'dino_pairs': dino_single,
                    'yolo_eta_min': 120,
                    'dino_eta_min': 30,
                    'total_eta_min': 150,
                },
                'all_sets': {
                    'set_count': len(REAL_SETS),
                    'yolo_geometries': len(all_yolo),
                    'dino_pairs': len(all_dino),
                    'yolo_eta_min': 360,
                    'dino_eta_min': 420,
                    'total_eta_min': 780,
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    # ------------------------------------------------------------------
    # Validación de Posiciones Estables (Fase 4)
    # ------------------------------------------------------------------

    def start_validation(self, runs: int = 20) -> dict:
        """Fase 4: Ejecuta simulación física en Blender para validar posiciones estables y contrastar con la BD."""
        global _validation_thread
        if _validation_thread and _validation_thread.is_alive():
            return {"status": "error", "message": "Ya hay un proceso de validación en curso."}

        def _run():
            global _validation_proc, _stop_validation_flag
            _stop_validation_flag = False
            
            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            if not os.path.exists(blender_path):
                _validation_progress.update({
                    "active": False,
                    "error": f"Ejecutable de Blender no encontrado en la ruta: {blender_path}. Por favor, configúralo en el archivo .env."
                })
                return

            val_script = os.path.join(project_root, "scripts", "validate_stable_poses.py")
            output_path = os.path.join(project_root, "data", "tmp", "stability_validation_results.json")
            
            # Reset progress
            _validation_progress.update({
                "current": 0,
                "total": 41, # Hay 41 piezas en el set
                "current_piece": "Iniciando validación...",
                "active": True,
                "done": False,
                "error": "",
                "results": None
            })
            
            cmd = [
                blender_path,
                "-b",
                "-P", val_script,
                "--",
                "--set_id", "75078-1",
                "--runs", str(runs),
                "--output_path", output_path
            ]
            
            print(f"[LegoVision GUI] Ejecutando validación en Blender: {' '.join(cmd)}")
            
            try:
                _validation_proc = subprocess.Popen(
                    cmd, cwd=project_root,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True,
                )
                
                for line in _validation_proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    ls = line.strip()
                    
                    # Parse line: e.g. "Piece 3/41: 3004 (Lego Brick)"
                    if ls.startswith("Piece ") and "/" in ls and ":" in ls:
                        match = re.match(r"Piece (\d+)/(\d+):\s*(.*)", ls)
                        if match:
                            curr_idx = int(match.group(1))
                            total_parts = int(match.group(2))
                            details = match.group(3)
                            _validation_progress["current"] = curr_idx
                            _validation_progress["total"] = total_parts
                            _validation_progress["current_piece"] = details
                            
                    if _stop_validation_flag:
                        _kill_proc(_validation_proc, 'Blender/Validation')
                        _validation_progress.update({'active': False, 'error': 'Cancelado por el usuario.'})
                        return
                        
                _validation_proc.wait()
                
                if _stop_validation_flag:
                    _validation_progress.update({'active': False, 'error': 'Cancelado por el usuario.'})
                    return
                    
                if _validation_proc.returncode == 0 and os.path.exists(output_path):
                    try:
                        with open(output_path, "r", encoding="utf-8") as f:
                            res_data = json.load(f)
                            
                        # Realizar comparación contra la base de datos desde la GUI
                        try:
                            from database import supabase_client
                            db_embeddings = supabase_client.get_all_embeddings()
                            db_faces = {}
                            for emb in db_embeddings:
                                r = emb["part_ref"]
                                f = emb["stable_face"]
                                if r not in db_faces:
                                    db_faces[r] = set()
                                db_faces[r].add(f)
                                
                            # Re-evaluar discrepancias
                            for item in res_data.get("report", []):
                                ref = item["part_ref"]
                                db_f = list(db_faces.get(ref, []))
                                exp_f = item["experimental_faces"]
                                
                                missing_in_db = [f for f in exp_f if f not in db_f]
                                extra_in_db = [f for f in db_f if f not in exp_f]
                                discrepancy = len(missing_in_db) > 0 or len(extra_in_db) > 0
                                
                                item["database_faces"] = db_f
                                item["missing_in_db"] = missing_in_db
                                item["extra_in_db"] = extra_in_db
                                item["discrepancy"] = discrepancy
                                
                            # Sobrescribir el reporte final con los datos de comparación reales
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(res_data, f, indent=2, ensure_ascii=False)
                        except Exception as dbe:
                            print(f"[LegoVision GUI ERROR] Falló la comparación BD en la GUI: {dbe}")
                            
                        _validation_progress.update({
                            "current": _validation_progress["total"],
                            "done": True,
                            "active": False,
                            "results": res_data
                        })
                    except Exception as e:
                        _validation_progress.update({
                            "error": f"Error leyendo reporte final: {str(e)}",
                            "active": False
                        })
                else:
                    _validation_progress.update({
                        "error": f"Blender falló con código {_validation_proc.returncode}",
                        "active": False
                    })
                    
            except Exception as ex:
                print(f"[LegoVision GUI ERROR] Validación falló: {ex}")
                _validation_progress.update({
                    "error": str(ex),
                    "active": False
                })
        _validation_thread = threading.Thread(target=_run, daemon=True)
        _validation_thread.start()
        return {"status": "started", "message": "Validación de estabilidad iniciada en segundo plano."}

    def get_validation_progress(self) -> dict:
        """Devuelve el progreso actual de la validación de posiciones estables."""
        pct = 0
        if _validation_progress["total"] > 0:
            pct = round((_validation_progress["current"] / _validation_progress["total"]) * 100, 1)
            
        if not _validation_progress["active"] and not _validation_progress["results"]:
            output_path = os.path.join(project_root, "data", "tmp", "stability_validation_results.json")
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        res_data = json.load(f)
                    _validation_progress["results"] = res_data
                    _validation_progress["done"] = True
                    _validation_progress["current"] = _validation_progress["total"]
                except Exception:
                    pass
                    
        return {
            "current": _validation_progress["current"],
            "total": _validation_progress["total"],
            "pct": pct,
            "current_piece": _validation_progress.get("current_piece", ""),
            "active": _validation_progress["active"],
            "done": _validation_progress["done"],
            "error": _validation_progress.get("error", ""),
            "results": _validation_progress.get("results")
        }

    def stop_validation(self) -> dict:
        """Detiene la validación en curso."""
        global _stop_validation_flag
        _stop_validation_flag = True
        if _validation_proc:
            _kill_proc(_validation_proc, 'Blender/Validation')
            _validation_progress.update({'active': False, 'error': 'Cancelado por el usuario.'})
            return {"status": "stopped", "message": "Proceso de validación detenido."}
        return {"status": "idle", "message": "No había validación activa."}

    # ------------------------------------------------------------------
    # Simulación FOV Inferencia e Indexación DINOv2 Dinámica
    # ------------------------------------------------------------------

    def start_dinov2_fov_simulation(self, part_ref: str, num_rotations: int, num_pieces: int, color_hex: str = "A0A5A9", clear_previous: bool = True) -> dict:
        """Lanza la simulación de la cámara de inferencia en el FOV y la vectorización con DINOv2."""
        global _dino_fov_thread
        import math
        if _dino_fov_thread and _dino_fov_thread.is_alive():
            return {"status": "error", "message": "Ya hay una simulación de FOV DINOv2 en curso."}

        # 1. Obtener posiciones estables de la BD
        poses = supabase_client.get_stable_poses(part_ref)
        if not poses:
            return {
                "status": "error",
                "message": f"No hay posiciones estables simuladas en la base de datos para la pieza {part_ref}. Por favor, ejecuta primero la Validación Física en la pestaña de Entrenamiento para generar poses estables empíricas."
            }

        # Guardar poses estables a JSON temporal
        tmp_dir = os.path.join(project_root, "data", "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        poses_json_path = os.path.join(tmp_dir, f"dino_stable_poses_{part_ref}.json")
        try:
            with open(poses_json_path, "w", encoding="utf-8") as f:
                json.dump(poses, f, indent=2)
        except Exception as e:
            return {"status": "error", "message": f"Error guardando JSON temporal de poses: {e}"}

        def _run():
            global _dino_fov_proc1, _dino_fov_proc2, _stop_dino_fov_flag
            _stop_dino_fov_flag = False

            # Inicializar progreso. Calculamos aproximadamente cuántos frames de renderizado haremos:
            # En cada render ponemos un layout de hasta 12 piezas (o más según el empaquetado).
            # Para estar seguros, estimamos el total basándonos en num_pieces y num_rotations.
            estimated_frames = num_rotations * max(1, math.ceil(num_pieces / 12))

            _dino_fov_progress.update({
                "current": 0,
                "total": estimated_frames,
                "current_file": "Iniciando Blender...",
                "active": True,
                "done": False,
                "error": "",
                "phase": "rendering"
            })

            # 1. Configurar Blender
            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            if not os.path.exists(blender_path):
                blender_path = "blender"
            
            output_dir = os.path.join(project_root, "data", "dino_scatter", part_ref)
            os.makedirs(output_dir, exist_ok=True)

            render_script = os.path.join(project_root, "scripts", "generate_dino_fov_renders.py")
            cmd_blender = [
                blender_path, "-b", "-P", render_script, "--",
                "--part_ref", part_ref,
                "--color_hex", color_hex,
                "--num_rotations", str(num_rotations),
                "--num_pieces", str(num_pieces),
                "--stable_poses_json", poses_json_path,
                "--output_dir", output_dir
            ]

            print(f"[LegoVision GUI] Ejecutando render FOV en Blender: {' '.join(cmd_blender)}")

            try:
                _dino_fov_proc1 = subprocess.Popen(
                    cmd_blender, cwd=project_root,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True
                )
                
                # Monitorear renderizado
                for line in _dino_fov_proc1.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    ls = line.strip()
                    if "[Blender] Renderizado:" in ls:
                        _dino_fov_progress["current_file"] = ls
                        _dino_fov_progress["current"] = min(_dino_fov_progress["current"] + 1, _dino_fov_progress["total"])
                    
                    if _stop_dino_fov_flag:
                        _kill_proc(_dino_fov_proc1, 'Blender/DINO-FOV')
                        _dino_fov_progress.update({"active": False, "phase": "error", "error": "Cancelado por el usuario."})
                        return

                _dino_fov_proc1.wait()
                if _dino_fov_proc1.returncode != 0:
                    _dino_fov_progress.update({
                        "active": False,
                        "phase": "error",
                        "error": f"Blender falló con código {_dino_fov_proc1.returncode}"
                    })
                    return
            except Exception as ex:
                _dino_fov_progress.update({"active": False, "phase": "error", "error": str(ex)})
                return

            if _stop_dino_fov_flag:
                _dino_fov_progress.update({"active": False, "phase": "error", "error": "Cancelado por el usuario."})
                return

            # 2. Configurar Indexador
            _dino_fov_progress.update({
                "current": 0,
                "total": 100,
                "current_file": "Iniciando indexador DINOv2...",
                "phase": "indexing"
            })

            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            idx_script = os.path.join(project_root, "scripts", "index_dino_fov_crops.py")
            
            cmd_idx = [
                python_exec, idx_script,
                "--part_ref", part_ref,
                "--color_hex", color_hex,
                "--output_dir", output_dir
            ]
            if clear_previous:
                cmd_idx.append("--clear_previous")

            print(f"[LegoVision GUI] Ejecutando indexador: {' '.join(cmd_idx)}")

            try:
                _dino_fov_proc2 = subprocess.Popen(
                    cmd_idx, cwd=project_root,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True
                )
                
                # Monitorear indexación
                for line in _dino_fov_proc2.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    ls = line.strip()
                    if ls.startswith("PROGRESS:"):
                        # Parse pct, e.g. "PROGRESS:25.5%"
                        match = re.search(r"PROGRESS:([\d\.]+)%", ls)
                        if match:
                            pct = float(match.group(1))
                            _dino_fov_progress["current"] = int(pct)
                            _dino_fov_progress["total"] = 100
                        _dino_fov_progress["current_file"] = ls.split("|")[-1].strip()
                    
                    if _stop_dino_fov_flag:
                        _kill_proc(_dino_fov_proc2, 'Indexer/DINO-FOV')
                        _dino_fov_progress.update({"active": False, "phase": "error", "error": "Cancelado por el usuario."})
                        return

                _dino_fov_proc2.wait()
                if _dino_fov_proc2.returncode != 0:
                    _dino_fov_progress.update({
                        "active": False,
                        "phase": "error",
                        "error": f"El indexador falló con código {_dino_fov_proc2.returncode}"
                    })
                    return
            except Exception as ex:
                _dino_fov_progress.update({"active": False, "phase": "error", "error": str(ex)})
                return

            # Completado con éxito
            _dino_fov_progress.update({
                "current": 100,
                "total": 100,
                "current_file": "Indexación DINOv2 completada con éxito.",
                "active": False,
                "done": True,
                "phase": "done"
            })

        _dino_fov_thread = threading.Thread(target=_run, daemon=True)
        _dino_fov_thread.start()
        return {"status": "started", "message": "Simulación y vectorización iniciada."}

    def get_dinov2_fov_progress(self) -> dict:
        """Devuelve el estado y progreso de la simulación FOV DINOv2."""
        return _dino_fov_progress

    def stop_dinov2_fov_simulation(self) -> dict:
        """Detiene la simulación y el indexado en curso."""
        global _stop_dino_fov_flag
        _stop_dino_fov_flag = True
        killed = []
        if _dino_fov_proc1 and _dino_fov_proc1.poll() is None:
            _kill_proc(_dino_fov_proc1, 'Blender/DINO-FOV')
            killed.append("Renderizador Blender")
        if _dino_fov_proc2 and _dino_fov_proc2.poll() is None:
            _kill_proc(_dino_fov_proc2, 'Indexer/DINO-FOV')
            killed.append("Indexador DINOv2")
            
        _dino_fov_progress.update({
            "active": False,
            "phase": "error",
            "error": "Proceso cancelado por el usuario."
        })
        if killed:
            return {"status": "stopped", "message": f"Detenido: {', '.join(killed)}"}
        return {"status": "idle", "message": "No había simulación activa."}

    # ---------------------------------------------------------------------------------------------------------------------------------
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
                    comp["image"] = f"http://localhost:{self.api_port}/renders/" + filename
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

    def simulate_physics_scatter(self, part_ref: str, color_hex: str, num_simulations: int = 15) -> dict:
        """
        Ejecuta la simulación de física de caída y estabilización en Blender
        para el número especificado de piezas de la referencia y color dados.
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

            crops_dir = os.path.dirname(output_path)
            stats_path = os.path.join(crops_dir, f"physics_scatter_{part_ref}_{clean_color}_stats.json")
            
            # Helper para verificar que existan todos los recortes
            def _crops_exist(n):
                for ci in range(n):
                    cp = os.path.join(crops_dir, f"physics_scatter_{part_ref}_{clean_color}_crop_{ci}.png")
                    if not os.path.exists(cp):
                        return False
                return True

            # Lógica de Cache Hit
            if os.path.exists(stats_path) and _crops_exist(num_simulations):
                try:
                    with open(stats_path, "r", encoding="utf-8") as sf:
                        stats = json.load(sf)
                    if stats.get("num_simulations") == num_simulations:
                        print(f"[LegoVision Physics] Cache hit: {num_simulations} crops y stats JSON ya existen para {part_ref}_{clean_color}.")
                        cached_crops = [f"http://localhost:{self.api_port}/renders/physics_scatter_{part_ref}_{clean_color}_crop_{ci}.png" for ci in range(num_simulations)]
                        return {
                            "status": "success",
                            "image_url": cached_crops[0],
                            "crops": cached_crops,
                            "output_path": os.path.join(crops_dir, f"physics_scatter_{part_ref}_{clean_color}_crop_0.png"),
                            "message": f"Crops cargados desde cache ({num_simulations} vistas de {part_ref}).",
                            "total_physics_time": stats.get("total_physics_time", 0.0),
                            "total_render_time": stats.get("total_render_time", 0.0),
                            "physics_time_per_piece": stats.get("physics_time_per_piece", 0.0),
                            "render_time_per_piece": stats.get("render_time_per_piece", 0.0)
                        }
                except Exception as ex:
                    print(f"[LegoVision Physics WARN] Error leyendo cache stats JSON: {ex}")

            cmd = [
                blender_path,
                "-b",
                "-P", script_path,
                "--",
                "--part_ref", part_ref,
                "--color_hex", color_hex,
                "--num_simulations", str(num_simulations),
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
                
            # Cargar la lista de recortes
            crops = []
            for i in range(num_simulations):
                crop_filename = f"physics_scatter_{part_ref}_{clean_color}_crop_{i}.png"
                crop_path = os.path.join(os.path.dirname(output_path), crop_filename)
                if os.path.exists(crop_path):
                    crops.append(f"http://localhost:{self.api_port}/renders/{crop_filename}")

            # Leer tiempos de simulación y renderizado desde el JSON de estadísticas
            stats = {}
            if os.path.exists(stats_path):
                try:
                    with open(stats_path, "r", encoding="utf-8") as sf:
                        stats = json.load(sf)
                except Exception as ex:
                    print(f"[LegoVision Physics WARN] Error leyendo stats JSON post-ejecución: {ex}")
            
            # Devolver URL de la imagen (fallback al crop 0) y los renders del carrusel junto con estadísticas de tiempo
            image_url = f"http://localhost:{self.api_port}/renders/{crop0_filename}"
            return {
                "status": "success",
                "image_url": image_url,
                "crops": crops,
                "output_path": crop0_path,
                "message": f"Simulación física (caída a 5cm) y renderizado a Z=10 completados para {num_simulations} vistas.",
                "total_physics_time": stats.get("total_physics_time", 0.0),
                "total_render_time": stats.get("total_render_time", 0.0),
                "physics_time_per_piece": stats.get("physics_time_per_piece", 0.0),
                "render_time_per_piece": stats.get("render_time_per_piece", 0.0)
            }
        except Exception as e:
            print(f"[LegoVision Physics ERROR] Excepción: {e}")
            return {"status": "error", "message": str(e)}

    def simulate_set_physics_scatter(self, set_id: str) -> dict:
        """
        Ejecuta la simulación física de caída y estabilización de todas las piezas
        del set especificado, renderizándolas sobre una cinta transportadora fotorrealista.
        """
        try:
            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            if not os.path.exists(blender_path):
                return {
                    "status": "error",
                    "message": f"Ejecutable de Blender no encontrado en la ruta: {blender_path}. Por favor, configúralo en el archivo .env."
                }

            script_path = os.path.join(project_root, "scripts", "physics_set_belt_generator.py")
            output_filename = f"set_scatter_{set_id}.png"
            output_path = os.path.join(project_root, "data", "synthetic_renders", output_filename)
            metadata_path = output_path.replace(".png", ".json")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Borrar la existente si existe para forzar regeneración cada vez
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"[LegoVision Set Physics] Borrado render existente: {output_path}")
                except Exception as e:
                    print(f"[WARN] No se pudo borrar el render existente: {e}")
            if os.path.exists(metadata_path):
                try:
                    os.remove(metadata_path)
                    print(f"[LegoVision Set Physics] Borrado metadata existente: {metadata_path}")
                except Exception as e:
                    print(f"[WARN] No se pudo borrar el metadata existente: {e}")

            cmd = [
                blender_path,
                "-b",
                "-P", script_path,
                "--",
                "--set_id", set_id,
                "--output_path", output_path
            ]

            print(f"[LegoVision Set Physics] Ejecutando simulación física de set en Blender: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"[LegoVision Set Physics ERROR] Blender falló con código {result.returncode}")
                return {
                    "status": "error",
                    "message": f"Blender falló con código de salida {result.returncode}",
                    "details": result.stderr[-500:] if result.stderr else "Sin detalles."
                }

            if not os.path.exists(output_path) or not os.path.exists(metadata_path):
                return {
                    "status": "error",
                    "message": "La simulación finalizó pero no se generó el render o los metadatos JSON esperados."
                }

            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

            return {
                "status": "success",
                "image_url": f"http://localhost:{self.api_port}/renders/{output_filename}",
                "metadata": meta_data,
                "message": "Simulación física del set y renderizado completados con éxito."
            }
        except Exception as e:
            print(f"[LegoVision Set Physics ERROR] Excepción: {e}")
            return {"status": "error", "message": str(e)}



    # ------------------------------------------------------------------
    # Generacion de Excel de Validacion de Posiciones Estables
    # ------------------------------------------------------------------
    def list_inference_renders(self, set_id: str = "75078-1") -> dict:
        """Lista renders de inferencia-test disponibles para un set."""
        try:
            renders_dir = os.path.join(project_root, "data", "synthetic_renders")
            set_tag = set_id.replace("-", "_")
            pattern = os.path.join(renders_dir, f"inference_test_{set_tag}_pf*.png")
            png_files = sorted(glob.glob(pattern))
            renders = []
            for png_path in png_files:
                filename = os.path.basename(png_path)
                json_path = png_path.replace(".png", ".json")
                import re as _re
                import datetime
                m = _re.search(r"_pf(\d+)", filename)
                pf = int(m.group(1)) if m else 0
                file_size_kb = round(os.path.getsize(png_path) / 1024, 1)
                meta = {}
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as fj:
                            meta = json.load(fj)
                    except Exception:
                        pass
                json_basename = os.path.basename(json_path)
                
                # Obtener la fecha de modificación del render
                mtime = os.path.getmtime(png_path)
                time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                renders.append({
                    "filename": filename,
                    "label": f"{set_id} — {pf} pzs/campo · {meta.get('pieces_placed', '?')} colocadas · {time_str} ({file_size_kb} KB)",
                    "pieces_in_field": pf,
                    "image_url": f"http://localhost:{self.api_port}/renders/{filename}",
                    "json_url": f"http://localhost:{self.api_port}/renders/{json_basename}" if os.path.exists(json_path) else None,
                    "metadata": meta,
                    "file_size_kb": file_size_kb,
                    "pieces_placed": meta.get("pieces_placed", 0),
                })
            return {"status": "success", "renders": renders, "count": len(renders)}
        except Exception as e:
            print(f"[ListRenders ERROR] {e}")
            return {"status": "error", "message": str(e), "renders": []}


    def generate_inference_test_render(self, pieces_in_field: int = 30, set_id: str = "75078-1", is_rolling: bool = True) -> dict:
        """
        Genera un render unico con todas las piezas del set sobre la cinta,
        usando orientaciones de stable_poses (placement directo, sin fisica).
        Comportamiento bloqueante - spinner en el UX mientras se genera.
        """
        try:
            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            if not os.path.exists(blender_path):
                return {
                    "status": "error",
                    "message": f"Blender no encontrado en: {blender_path}. Configura BLENDER_PATH en .env."
                }

            script_path = os.path.join(project_root, "scripts", "generate_inference_test_belt.py")
            if not os.path.exists(script_path):
                return {"status": "error", "message": "Script generate_inference_test_belt.py no encontrado."}

            renders_dir = os.path.join(project_root, "data", "synthetic_renders")
            os.makedirs(renders_dir, exist_ok=True)
            set_tag = set_id.replace("-", "_")
            
            import time
            timestamp = int(time.time())
            output_filename = f"inference_test_{set_tag}_pf{pieces_in_field}_{timestamp}.png"
            output_path = os.path.join(renders_dir, output_filename)
            metadata_path = output_path.replace(".png", ".json")

            cmd = [
                blender_path, "-b", "-P", script_path, "--",
                "--set_id", set_id,
                "--pieces_in_field", str(pieces_in_field),
                "--output_path", output_path,
                "--is_rolling", "true" if is_rolling else "false"
            ]

            print(f"[InferenceTest] Lanzando render Blender: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

            if result.returncode != 0:
                print(f"[InferenceTest ERROR] Blender fallo con codigo {result.returncode}")
                print(f"Stdout: {result.stdout[-800:]}")
                print(f"Stderr: {result.stderr[-400:]}")
                return {
                    "status": "error",
                    "message": f"Blender fallo (codigo {result.returncode})",
                    "details": result.stderr[-300:] if result.stderr else result.stdout[-300:]
                }

            if not os.path.exists(output_path):
                return {"status": "error", "message": "Render completado pero no se genero la imagen PNG."}

            meta_data = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                except Exception as e:
                    print(f"[InferenceTest WARN] Error leyendo metadata JSON: {e}")

            file_size_kb = round(os.path.getsize(output_path) / 1024, 1)
            print(f"[InferenceTest] Render generado: {output_path} ({file_size_kb} KB)")
            return {
                "status": "success",
                "image_url": f"http://localhost:{self.api_port}/renders/{output_filename}",
                "metadata": meta_data,
                "message": f"Render inferencia-test generado ({file_size_kb} KB, {meta_data.get('pieces_placed', '?')} piezas)"
            }
        except Exception as e:
            print(f"[InferenceTest ERROR] Excepcion: {e}")
            return {"status": "error", "message": str(e)}

    def delete_inference_render(self, filename: str) -> dict:
        """Elimina un render de inferencia y su archivo JSON de metadatos asociado."""
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
                return {"status": "success", "message": f"Render {filename} eliminado con éxito."}
            else:
                return {"status": "error", "message": f"No se encontró el archivo {filename}."}
        except Exception as e:
            print(f"[DeleteRender ERROR] {e}")
            return {"status": "error", "message": str(e)}

    def generate_validation_excel(self, set_id: str = "75078-1") -> dict:
        """
        Paso post-validacion: genera documento Excel comparativo de posiciones estables.
        1) Lanza Blender para renderizar las poses simuladas en data/validation_renders/
        2) Genera el Excel con imagenes BrickLink + renders + comparativa semantico vs experimental
        Output: data/validation_report_{set_id}.xlsx
        """
        try:
            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            validation_json = os.path.join(project_root, "data", "tmp", "stability_validation_results.json")
            renders_dir = os.path.join(project_root, "data", "validation_renders")
            output_excel = os.path.join(project_root, "data", "validation_report_" + set_id.replace("-", "") + ".xlsx")

            if not os.path.exists(validation_json):
                return {
                    "status": "error",
                    "message": "No hay resultados de validacion. Ejecuta la validacion (paso 4) primero."
                }

            # Paso 1: Renderizar poses simuladas con Blender (en directorio separado)
            os.makedirs(renders_dir, exist_ok=True)
            render_script = os.path.join(project_root, "scripts", "render_stable_poses_validation.py")

            if os.path.exists(blender_path) and os.path.exists(render_script):
                print("[LegoVision GUI] Renderizando poses simuladas en Blender...")
                render_cmd = [
                    blender_path, "-b", "-P", render_script, "--",
                    "--set_id", set_id,
                    "--output_dir", renders_dir,
                    "--validation_json", validation_json
                ]
                try:
                    result = subprocess.run(render_cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode != 0:
                        print("[LegoVision GUI WARN] Blender render retorno " + str(result.returncode) + ". Continuando con renders existentes.")
                except subprocess.TimeoutExpired:
                    print("[LegoVision GUI WARN] Timeout renderizando poses. Continuando con renders existentes.")
                except Exception as ex:
                    print("[LegoVision GUI WARN] Error en render Blender: " + str(ex))
            else:
                print("[LegoVision GUI WARN] Blender no disponible o script no encontrado. Generando Excel sin renders.")

            # Paso 2: Ejecutar algoritmo semantico LDraw
            semantic_json = os.path.join(project_root, "data", "tmp", "semantic_poses_" + set_id.replace("-", "") + ".json")
            semantic_script = os.path.join(project_root, "scripts", "analyze_stable_poses_ldraw.py")
            if os.path.exists(semantic_script):
                venv_python_s = os.path.join(project_root, ".venv", "bin", "python")
                python_exec_s = venv_python_s if os.path.exists(venv_python_s) else sys.executable
                try:
                    print("[LegoVision GUI] Ejecutando algoritmo semantico LDraw...")
                    result_s = subprocess.run(
                        [python_exec_s, semantic_script, "--set_id", set_id, "--output", semantic_json],
                        capture_output=True, text=True, timeout=300, cwd=project_root
                    )
                    if result_s.returncode == 0:
                        print("[LegoVision GUI] Algoritmo semantico completado: " + semantic_json)
                    else:
                        print("[LegoVision GUI WARN] Algoritmo semantico retorno " + str(result_s.returncode))
                except Exception as ex_s:
                    print("[LegoVision GUI WARN] Error en algoritmo semantico: " + str(ex_s))

            # Paso 3: Generar Excel
            excel_script = os.path.join(project_root, "scripts", "generate_validation_excel.py")
            if not os.path.exists(excel_script):
                return {"status": "error", "message": "Script generate_validation_excel.py no encontrado."}

            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable

            excel_cmd = [
                python_exec, excel_script,
                "--set_id", set_id,
                "--validation_json", validation_json,
                "--output", output_excel,
                "--renders_dir", renders_dir,
                "--semantic_json", semantic_json
            ]
            print("[LegoVision GUI] Generando Excel de validacion...")
            result = subprocess.run(excel_cmd, capture_output=True, text=True, timeout=300, cwd=project_root)

            if result.returncode != 0:
                print("[LegoVision GUI ERROR] Excel script error: " + result.stdout[-500:])
                return {
                    "status": "error",
                    "message": "Error generando Excel: " + result.stderr[-200:] if result.stderr else result.stdout[-200:]
                }

            if not os.path.exists(output_excel):
                return {"status": "error", "message": "El Excel no se genero correctamente."}

            file_size_kb = round(os.path.getsize(output_excel) / 1024, 1)
            print("[LegoVision GUI] Excel generado: " + output_excel + " (" + str(file_size_kb) + " KB)")
            return {
                "status": "success",
                "path": output_excel,
                "filename": os.path.basename(output_excel),
                "size_kb": file_size_kb,
                "message": "Excel generado correctamente (" + str(file_size_kb) + " KB)"
            }
        except Exception as e:
            print("[LegoVision GUI ERROR] generate_validation_excel: " + str(e))
            return {"status": "error", "message": str(e)}

    def open_validation_excel(self, set_id: str = "75078-1") -> dict:
        """Abre el Excel de validacion con la aplicacion del sistema."""
        try:
            output_excel = os.path.join(project_root, "data", "validation_report_" + set_id.replace("-", "") + ".xlsx")
            if not os.path.exists(output_excel):
                return {"status": "error", "message": "El Excel no existe. Genera el reporte primero."}
            import subprocess as sp
            sp.Popen(["open", output_excel])
            return {"status": "success", "message": "Abriendo " + os.path.basename(output_excel)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def open_training_specs(self) -> dict:
        """Abre el documento de especificaciones del entrenamiento."""
        try:
            specs_file = os.path.join(project_root, "docs", "setup_y_logs.md")
            if not os.path.exists(specs_file):
                return {"status": "error", "message": "El archivo de especificaciones no existe."}
            import subprocess as sp
            sp.Popen(["open", specs_file])
            return {"status": "success", "message": "Abriendo " + os.path.basename(specs_file)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_step5_pipeline(self, pipeline_id: str) -> dict:
        """Inicia la generación de renders y entrenamiento real de forma asíncrona para el Paso 5."""
        global _step5_threads, _step5_states
        
        # Inicializar diccionarios si no existen
        if '_step5_threads' not in globals():
            globals()['_step5_threads'] = {}
        if '_step5_states' not in globals():
            globals()['_step5_states'] = {}
            
        _step5_threads = globals()['_step5_threads']
        _step5_states = globals()['_step5_states']
        
        if pipeline_id in _step5_threads and _step5_threads[pipeline_id].is_alive():
            return {"status": "error", "message": "Este entrenamiento/indexado ya está en curso."}

        _step5_states[pipeline_id] = {
            "status": "⏳ Generando...",
            "progress": 0,
            "log": "Iniciando pipeline en segundo plano...\n",
            "done": False,
            "error": None
        }

        def _run():
            import subprocess as sp
            import time
            from datetime import datetime

            state = _step5_states[pipeline_id]
            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            blender_exec = "/opt/homebrew/bin/blender"
            if not os.path.exists(blender_exec):
                blender_exec = "blender"

            console_logs = []
            def log_and_append(msg):
                console_logs.append(msg)
                state["log"] += msg

            log_and_append(f"=== LEGOVISION PIPELINE {pipeline_id.upper()} ===\n")
            log_and_append(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            try:
                if pipeline_id == "yolo-cenital-render":
                    out_dir = os.path.join(project_root, "data", "yolo_cenital")
                    log_and_append(f"Generando renders YOLO Cenital en {out_dir}...\n")
                    cmd1 = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_yolo_training_dataset.py"), "--",
                            "--set_id", "75078-1", "--num_frames", "50", "--pieces", "1", "--center_spawn", "--output_dir", out_dir, "--camera_type", "cenital"]
                    p1 = sp.Popen(cmd1, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p1.stdout:
                        log_and_append(line)
                    p1.wait()

                elif pipeline_id == "yolo-cenital-train":
                    out_dir = os.path.join(project_root, "data", "yolo_cenital")
                    log_and_append("Iniciando entrenamiento YOLO Cenital (2 épocas para prueba rápida)...\n")
                    cmd2 = [python_exec, os.path.join(project_root, "training", "train_yolo.py"),
                            "--epochs", "2", "--batch", "16", "--dataset_size", "50", "--set_id", "75078-1",
                            "--raw_dataset_dir", out_dir, "--processed_dataset_dir", os.path.join(project_root, "data", "processed_yolo_cenital"),
                            "--model_name", "yolo_cenital"]
                    p2 = sp.Popen(cmd2, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p2.stdout:
                        log_and_append(line)
                    p2.wait()

                elif pipeline_id == "yolo-lateral-render":
                    out_dir = os.path.join(project_root, "data", "yolo_lateral")
                    log_and_append(f"Generando renders YOLO Lateral en {out_dir}...\n")
                    cmd1 = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_yolo_training_dataset.py"), "--",
                            "--set_id", "75078-1", "--num_frames", "50", "--pieces", "1", "--center_spawn", "--output_dir", out_dir, "--camera_type", "lateral"]
                    p1 = sp.Popen(cmd1, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p1.stdout:
                        log_and_append(line)
                    p1.wait()

                elif pipeline_id == "yolo-lateral-train":
                    out_dir = os.path.join(project_root, "data", "yolo_lateral")
                    log_and_append("Iniciando entrenamiento YOLO Lateral (2 épocas para prueba rápida)...\n")
                    cmd2 = [python_exec, os.path.join(project_root, "training", "train_yolo.py"),
                            "--epochs", "2", "--batch", "16", "--dataset_size", "50", "--set_id", "75078-1",
                            "--raw_dataset_dir", out_dir, "--processed_dataset_dir", os.path.join(project_root, "data", "processed_yolo_lateral"),
                            "--model_name", "yolo_lateral"]
                    p2 = sp.Popen(cmd2, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p2.stdout:
                        log_and_append(line)
                    p2.wait()

                elif pipeline_id == "dinov2-cenital":
                    out_dir = os.path.join(project_root, "data", "dinov2_cenital")
                    log_and_append(f"Paso 1: Generando referencias DINOv2 Cenital en {out_dir}...\n")
                    cmd1 = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_physics_ref_multiangle.py"), "--",
                            "--output_dir", out_dir, "--camera_type", "cenital", "--drops", "1", "--set_id", "75078-1"]
                    p1 = sp.Popen(cmd1, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p1.stdout:
                        log_and_append(line)
                    p1.wait()
                    
                    log_and_append("Paso 2: Iniciando indexación de embeddings DINOv2 Cenital...\n")
                    cmd2 = [python_exec, os.path.join(project_root, "training", "index_synthetic_renders.py"),
                            "--renders_dir", out_dir, "--multiangle_dir", out_dir]
                    p2 = sp.Popen(cmd2, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p2.stdout:
                        log_and_append(line)
                    p2.wait()

                elif pipeline_id == "dinov2-lateral":
                    out_dir = os.path.join(project_root, "data", "dinov2_lateral")
                    log_and_append(f"Paso 1: Generando referencias DINOv2 Lateral en {out_dir}...\n")
                    cmd1 = [blender_exec, "-b", "-P", os.path.join(project_root, "scripts", "generate_physics_ref_multiangle.py"), "--",
                            "--output_dir", out_dir, "--camera_type", "lateral", "--drops", "1", "--set_id", "75078-1"]
                    p1 = sp.Popen(cmd1, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p1.stdout:
                        log_and_append(line)
                    p1.wait()
                    
                    log_and_append("Paso 2: Iniciando indexación de embeddings DINOv2 Lateral...\n")
                    cmd2 = [python_exec, os.path.join(project_root, "training", "index_synthetic_renders.py"),
                            "--renders_dir", out_dir, "--multiangle_dir", out_dir]
                    p2 = sp.Popen(cmd2, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
                    for line in p2.stdout:
                        log_and_append(line)
                    p2.wait()

                specs_file = os.path.join(project_root, "docs", "setup_y_logs.md")
                if os.path.exists(specs_file):
                    log_summary = "\n\n"
                    log_summary += f"## Log de Ejecución: {pipeline_id.upper()} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
                    log_summary += f"* **Directorio de imágenes**: `data/{pipeline_id.replace('-', '_')}`\n"
                    log_summary += "* **Resultado**: Completado con éxito.\n"
                    log_summary += "* **Detalle de logs**:\n"
                    log_summary += "```\n"
                    log_summary += "".join(console_logs[-30:])
                    log_summary += "\n```\n"
                    with open(specs_file, "a", encoding="utf-8") as sf:
                        sf.write(log_summary)

                if pipeline_id.endswith("-render"):
                    state["status"] = "Listo"
                elif pipeline_id.endswith("-train"):
                    state["status"] = "Trained"
                else:
                    state["status"] = "Indexado"
                state["done"] = True

            except Exception as e:
                log_and_append(f"\n❌ Error durante la ejecución del pipeline: {e}\n")
                state["status"] = "Error"
                state["done"] = True
                state["error"] = str(e)

        _step5_threads[pipeline_id] = threading.Thread(target=_run, daemon=True)
        _step5_threads[pipeline_id].start()
        return {"status": "started"}

    def get_step5_pipeline_status(self, pipeline_id: str) -> dict:
        """Retorna el estado de la pipeline indicada en el Paso 5."""
        if '_step5_states' not in globals() or pipeline_id not in globals()['_step5_states']:
            return {"status": "Pendiente", "done": False}
        return globals()['_step5_states'][pipeline_id]

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
