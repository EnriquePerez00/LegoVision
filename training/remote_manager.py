import os
import sys
import time
import shutil
import subprocess
import argparse
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Cargar .env
load_dotenv(os.path.join(project_root, ".env"), override=True)

try:
    from lightning_sdk import Studio, Job, Machine
except ImportError:
    print("Por favor, instala lightning-sdk: pip install lightning-sdk")
    sys.exit(1)

def get_studio():
    studio_name = os.environ.get("LIGHTNING_STUDIO_NAME", "legovision-t4")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "enriqueperezbcn1973")
    user_id = os.environ.get("LIGHTNING_USER_ID", "")
    
    if not user_id:
        print("ERROR: LIGHTNING_USER_ID no está configurado en el .env")
        sys.exit(1)
        
    print(f"[Remote Manager] Conectando a Studio '{studio_name}' en teamspace '{teamspace}'...")
    studio = Studio(name=studio_name, teamspace=teamspace, user=user_id)
    return studio

def start_studio():
    studio = get_studio()
    print("[Remote Manager] Iniciando el Studio (puede tardar unos minutos si estaba apagado)...")
    studio.start()
    print("[Remote Manager] Studio iniciado y listo.")

def upload_dataset():
    """Comprime el dataset local y lo sube al Studio remoto"""
    studio = get_studio()
    
    local_data_dir = os.path.join(project_root, "data", "raw_dataset")
    if not os.path.exists(local_data_dir):
        print(f"ERROR: No se encontró el dataset en {local_data_dir}")
        return

    # Comprimir el dataset para subida más rápida
    zip_path = os.path.join(project_root, "data", "raw_dataset.zip")
    print(f"[Remote Manager] Comprimiendo {local_data_dir}...")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', local_data_dir)
    
    print(f"[Remote Manager] Subiendo {zip_path} al Studio...")
    # Usamos CLI para copiar ya que es más robusto para archivos grandes
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "enriqueperezbcn1973")
    user_id = os.environ.get("LIGHTNING_USER_ID", "")
    studio_name = os.environ.get("LIGHTNING_STUDIO_NAME", "legovision-t4")
    
    remote_lit_path = f"lit://{user_id}/{teamspace}/studios/{studio_name}/data/"
    
    subprocess.run(["lightning", "studio", "cp", zip_path, remote_lit_path], check=True)
    print("[Remote Manager] Subida de dataset completada.")
    
    # También subir código fuente relevante
    print("[Remote Manager] Sincronizando scripts de entrenamiento...")
    subprocess.run(["lightning", "studio", "cp", "-r", os.path.join(project_root, "training"), remote_lit_path.replace("/data/", "/")], check=True)

def run_training():
    studio = get_studio()
    print("[Remote Manager] Descomprimiendo dataset y lanzando entrenamiento...")
    
    # El comando remoto hace:
    # 1. Instalar dependencias si faltan
    # 2. Descomprimir dataset
    # 3. Ejecutar train_yolo.py con --remote
    
    remote_command = """
    cd /home/zeus/
    mkdir -p data/raw_dataset
    unzip -q -o data/raw_dataset.zip -d data/raw_dataset/
    pip install ultralytics python-dotenv
    python training/train_yolo.py --remote --epochs 35 --model_name yolo11_piece_detector
    """
    
    job = Job.run(
        command=remote_command,
        name="legovision-yolo-train",
        machine=Machine.T4,
        studio=studio
    )
    print(f"[Remote Manager] Trabajo enviado con éxito. Estado actual: {job.status}")
    print("Para verificar periódicamente y descargar los modelos, ejecuta: python training/remote_manager.py --action check")

def check_and_download():
    studio = get_studio()
    print(f"[Remote Manager] Verificando estado del Studio...")
    
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "enriqueperezbcn1973")
    user_id = os.environ.get("LIGHTNING_USER_ID", "")
    studio_name = os.environ.get("LIGHTNING_STUDIO_NAME", "legovision-t4")
    
    remote_best_pt = f"lit://{user_id}/{teamspace}/studios/{studio_name}/models/yolo11_piece_detector.pt"
    local_models_dir = os.path.join(project_root, "models")
    os.makedirs(local_models_dir, exist_ok=True)
    
    try:
        print("[Remote Manager] Intentando descargar modelo entrenado (yolo11_piece_detector.pt)...")
        subprocess.run(["lightning", "studio", "cp", remote_best_pt, f"{local_models_dir}/"], check=True)
        print(f"[Remote Manager] ¡Éxito! Modelo descargado en {local_models_dir}/yolo11_piece_detector.pt")
        
        # Intentar traer los logs de ultralytics
        print("[Remote Manager] Intentando descargar logs de Ultralytics...")
        remote_runs_dir = f"lit://{user_id}/{teamspace}/studios/{studio_name}/runs/"
        subprocess.run(["lightning", "studio", "cp", "-r", remote_runs_dir, os.path.join(project_root, "runs_remote/")], check=False)
        
    except subprocess.CalledProcessError:
        print("[Remote Manager] Aún no se ha encontrado el modelo final. Es posible que el entrenamiento siga en curso.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["start", "upload", "run", "check", "full"], required=True, help="Acción a realizar en el Studio remoto")
    args = parser.parse_args()
    
    if args.action == "start":
        start_studio()
    elif args.action == "upload":
        upload_dataset()
    elif args.action == "run":
        run_training()
    elif args.action == "check":
        check_and_download()
    elif args.action == "full":
        start_studio()
        upload_dataset()
        run_training()
