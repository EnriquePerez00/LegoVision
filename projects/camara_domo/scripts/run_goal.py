import time
import subprocess
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
legovic_root = os.path.dirname(project_root)

print("=== LEGOVISION GOAL RUNNER ===")
print("1. Esperando a que el entrenamiento remoto termine y descargue los modelos...")

while True:
    result = subprocess.run(
        ["./.venv/bin/python", "camara_domo/scripts/remote_manager.py", "--action", "check"],
        capture_output=True, text=True, cwd=legovic_root
    )
    if "¡Todos los logs descargados exitosamente!" in result.stdout:
        print("✅ Modelos y logs descargados con éxito!")
        break
    elif "Aún no se han encontrado" in result.stdout:
        print("⏳ Entrenando... esperando 300s...")
        time.sleep(300)
    else:
        print("⚠️ Salida inesperada del manager. Reintentando en 300s...")
        print(result.stdout)
        print(result.stderr)
        time.sleep(300)

print("1.5. Descargando los logs de ejecución del Job (stdout)...")
# Recuperar logs de consola mediante el SDK y desactivando SSL temporalmente
log_fetch_script = """
import ssl
import os
from dotenv import load_dotenv
from lightning_sdk import Job

ssl._create_default_https_context = ssl._create_unverified_context
load_dotenv(".env")

teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "enriqueperezbcn1973")
username = "enriqueperezbcn1973"

try:
    # Obtener el job exacto usando list()
    from lightning_sdk import Studio
    studio_name = os.environ.get("LIGHTNING_STUDIO_NAME", "legovision-t4")
    
    # Obtener el job exacto usando el nombre guardado (con su ID)
    try:
        with open("camara_domo/logs/last_job_name.txt", "r") as fn:
            real_job_name = fn.read().strip()
    except Exception:
        real_job_name = "domo-yolo-pose-train"
        
    job = Job(name=real_job_name, teamspace=teamspace, user=username)
    logs_content = job.logs
    with open("camara_domo/logs/remote_training_stdout.log", "w", encoding="utf-8") as f:
        f.write(logs_content)
    print("Logs de consola guardados en camara_domo/logs/remote_training_stdout.log")
except Exception as e:
    print("No se pudo obtener el stdout del job remoto:", e)
"""
with open("fetch_logs_tmp.py", "w") as f:
    f.write(log_fetch_script)
subprocess.run(["./.venv/bin/python", "fetch_logs_tmp.py"], cwd=legovic_root)
os.remove("fetch_logs_tmp.py")

print("2. Ejecutando ciclo de inferencia comparativa (100 imágenes/frames)...")
# Modificamos run_comparative_inference on the fly o simplemente llamamos a inferencia_neuronal_v2 3 veces.
# Pero run_comparative_inference.py ya hace el bucle de los 3 modos. Lo editaremos localmente con replace_file_content antes de lanzar esto, para que use 100 frames.
subprocess.run(["./.venv/bin/python", "camara_domo/scripts/run_comparative_inference.py"], cwd=legovic_root, check=True)

print("3. Generando reporte completo de seguimiento (3 métodos)...")
subprocess.run(["./.venv/bin/python", "camara_domo/scripts/generate_general_report_tracking.py"], cwd=legovic_root, check=True)

print("=== GOAL COMPLETADO EXITOSAMENTE ===")
