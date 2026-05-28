import os
import sys
import json
import threading
import subprocess
import webview

# Añadir directorio raíz al path para importar de database e inference
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv(override=True)

from database import supabase_client

# Hilo de entrenamiento activo (solo uno a la vez)
_training_thread: threading.Thread | None = None
_indexing_thread: threading.Thread | None = None


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
        """Obtiene el inventario de un set, cargando o generando los renders 3D."""
        try:
            from database import set_catalog
            import base64

            set_data = set_catalog.get_set_data(set_id)

            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            ldraw_path = os.getenv("LDRAW_PATH", "./data/ldraw")
            if not os.path.isabs(ldraw_path):
                ldraw_path = os.path.abspath(os.path.join(project_root, ldraw_path))

            # Procesar cada pieza del set
            for part in set_data["parts"]:
                part_ref = part["ref"]
                color_code = part["color_code"]
                color_hex = part["color_hex"]

                # 1. Comprobar si ya existe en la base de datos
                image_data = supabase_client.get_part_render(part_ref, color_code)

                if image_data:
                    part["image"] = f"data:image/png;base64,{image_data}"
                else:
                    tmp_dir = os.path.join(project_root, "data", "tmp")
                    os.makedirs(tmp_dir, exist_ok=True)
                    output_png = os.path.join(tmp_dir, f"render_{part_ref}_{color_code}.png")

                    script_path = os.path.join(project_root, "blender_pipeline", "render_part.py")
                    cmd = [
                        blender_path,
                        "--background",
                        "--python", script_path,
                        "--",
                        "--part_id", part_ref,
                        "--color_hex", color_hex,
                        "--ldraw_path", ldraw_path,
                        "--output", output_png,
                    ]

                    print(f"[LegoVision GUI] Generando render bajo demanda: {part_ref} (Color {color_hex})...")
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    if os.path.exists(output_png):
                        with open(output_png, "rb") as f:
                            encoded_img = base64.b64encode(f.read()).decode("utf-8")
                        supabase_client.save_part_render(part_ref, color_code, encoded_img)
                        part["image"] = f"data:image/png;base64,{encoded_img}"
                        os.remove(output_png)
                    else:
                        part["image"] = ""

            return {
                "status": "success",
                "set_name": set_data["name"],
                "minifigures": set_data["minifigures"],
                "parts": set_data["parts"],
            }
        except Exception as e:
            print(f"[LegoVision GUI ERROR] get_set_inventory: {e}")
            return {"status": "error", "message": str(e)}

    def get_set_inventory_light(self, set_id: str) -> dict:
        """Obtiene el inventario de un set rápidamente sin generar renders 3D."""
        try:
            from database import set_catalog
            set_data = set_catalog.get_set_data(set_id)
            return {
                "status": "success",
                "set_name": set_data["name"],
                "minifigures": set_data["minifigures"],
                "parts": set_data["parts"],
            }
        except Exception as e:
            print(f"[LegoVision GUI ERROR] get_set_inventory_light: {e}")
            return {"status": "error", "message": str(e)}

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
        Inicia el proceso de indexación DINOv2 (multi-view renders + embedding extraction).
        Ejecuta en un hilo de fondo.
        """
        global _indexing_thread
        if _indexing_thread and _indexing_thread.is_alive():
            return {"status": "error", "message": "Ya hay un proceso de indexación en curso."}

        def _run():
            venv_python = os.path.join(project_root, ".venv", "bin", "python")
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable

            # Step 1: Generate multi-view renders with Blender
            blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
            ldraw_path = os.getenv("LDRAW_PATH", "./data/ldraw")
            if not os.path.isabs(ldraw_path):
                ldraw_path = os.path.abspath(os.path.join(project_root, ldraw_path))

            mv_script = os.path.join(project_root, "blender_pipeline", "generate_multi_view.py")
            output_dir = os.path.join(project_root, "data", "tmp", "ref_renders")
            os.makedirs(output_dir, exist_ok=True)

            print(f"[LegoVision GUI] Generando vistas multi-ángulo para set {set_id}...")
            cmd_blender = [
                blender_path, "--background",
                "--python", mv_script,
                "--",
                "--set_id", set_id,
                "--ldraw_path", ldraw_path,
                "--output_dir", output_dir,
            ]
            proc1 = subprocess.Popen(
                cmd_blender, cwd=project_root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc1.stdout:
                sys.stdout.write(line); sys.stdout.flush()
            proc1.wait()

            if proc1.returncode != 0:
                print(f"[LegoVision GUI ERROR] Generación multi-vista falló (código {proc1.returncode})")
                return

            # Step 2: Index embeddings with DINOv2
            idx_script = os.path.join(project_root, "training", "index_embeddings.py")
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

    # Arrancar bucle de interfaz
    webview.start(debug=True)


if __name__ == "__main__":
    main()
