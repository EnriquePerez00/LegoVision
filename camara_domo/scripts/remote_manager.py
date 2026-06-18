import os
import sys
import shutil
import zipfile
import subprocess
import argparse
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    
    # user_id is the auth UUID, but Studio(user=) expects the username or org name.
    username = "enriqueperezbcn1973"
        
    print(f"[Remote Manager Domo] Conectando a Studio '{studio_name}' en teamspace '{teamspace}'...")
    studio = Studio(name=studio_name, teamspace=teamspace, user=username)
    return studio

def start_studio():
    studio = get_studio()
    print("[Remote Manager Domo] Iniciando el Studio (puede tardar unos minutos si estaba apagado)...")
    studio.start()
    print("[Remote Manager Domo] Studio iniciado y listo.")

def zip_selective_data(base_data_dir, target_zip, folders_to_include):
    import zipfile
    print(f"[Remote Manager Domo] Comprimiendo carpetas selectivas en {target_zip}...")
    with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_name in folders_to_include:
            folder_path = os.path.join(base_data_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"ADVERTENCIA: No se encontró la carpeta {folder_path}")
                continue
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # El path relativo dentro del zip
                    rel_path = os.path.relpath(file_path, base_data_dir)
                    zipf.write(file_path, rel_path)
    print(f"[Remote Manager Domo] Compresión completada: {os.path.getsize(target_zip) / (1024*1024):.2f} MB")

def upload_dataset():
    studio = get_studio()
    
    local_data_dir = os.path.join(project_root, "camara_domo", "data")
    zip_path = os.path.join(project_root, "camara_domo", "data", "domo_training_data.zip")
    
    folders_to_include = ["yolo_dataset_cenital", "yolo_dataset_frontal"]
    zip_selective_data(local_data_dir, zip_path, folders_to_include)
    
    print(f"[Remote Manager Domo] Subiendo {zip_path} al Studio...")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "enriqueperezbcn1973")
    username = "enriqueperezbcn1973"
    studio_name = os.environ.get("LIGHTNING_STUDIO_NAME", "legovision-t4")
    
    lightning_cli = os.path.join(os.path.dirname(sys.executable), "lightning")
    
    remote_lit_path = f"lit://{username}/{teamspace}/studios/{studio_name}/camara_domo/data/"
    
    subprocess.run([lightning_cli, "studio", "cp", zip_path, remote_lit_path], check=True)
    print("[Remote Manager Domo] Subida de dataset completada.")
    
    # Subir código de scripts (solo el necesario)
    print("[Remote Manager Domo] Sincronizando scripts de entrenamiento...")
    script_path = os.path.join(project_root, "camara_domo", "scripts", "train_yolo_pose_remote.py")
    subprocess.run([lightning_cli, "studio", "cp", script_path, f"lit://{username}/{teamspace}/studios/{studio_name}/camara_domo/scripts/"], check=True)

def run_training():
    studio = get_studio()
    print("[Remote Manager Domo] Descomprimiendo dataset y lanzando entrenamiento unificado...")
    
    remote_command = """
    mkdir -p camara_domo/data
    unzip -q -o camara_domo/data/domo_training_data.zip -d camara_domo/data/
    pip install ultralytics python-dotenv
    python camara_domo/scripts/train_yolo_pose_remote.py
    """
    
    job = Job.run(
        command=remote_command,
        name="domo-yolo-pose-train",
        machine=Machine.T4,
        studio=studio
    )
    
    # Guardar el nombre exacto del job (que Lightning modifica añadiendo un ID) para poder rastrearlo
    job_name = job.name
    log_dir = os.path.join(project_root, "camara_domo", "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "last_job_name.txt"), "w") as f:
        f.write(job_name)
        
    print(f"[Remote Manager Domo] Trabajo enviado con éxito.")
    print(f"[Remote Manager Domo] Nombre interno exacto del Job: {job_name}")
    print(f"[Remote Manager Domo] Estado actual: {job.status}")
    print("Para verificar periódicamente y descargar los modelos y logs, ejecuta: python camara_domo/scripts/remote_manager.py --action check")

def check_and_download():
    studio = get_studio()
    print(f"[Remote Manager Domo] Verificando estado del Studio...")
    
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE", "enriqueperezbcn1973")
    username = "enriqueperezbcn1973"
    studio_name = os.environ.get("LIGHTNING_STUDIO_NAME", "legovision-t4")
    lightning_cli = os.path.join(os.path.dirname(sys.executable), "lightning")
    
    local_models_dir = os.path.join(project_root, "camara_domo", "models")
    os.makedirs(local_models_dir, exist_ok=True)
    
    # Rutas remotas de pesos y de los runs
    remote_models_dir = f"lit://{username}/{teamspace}/studios/{studio_name}/camara_domo/models/train_runs/"
    remote_yolo_cen = f"lit://{username}/{teamspace}/studios/{studio_name}/camara_domo/models/yolo_cenital_pose.pt"
    remote_yolo_lat = f"lit://{username}/{teamspace}/studios/{studio_name}/camara_domo/models/yolo_frontal_pose.pt"
    
    try:
        print("[Remote Manager Domo] Intentando descargar modelo cenital entrenado...")
        subprocess.run([lightning_cli, "studio", "cp", remote_yolo_cen, f"{local_models_dir}/"], check=True)
        print(f"[Remote Manager Domo] ¡Modelo cenital descargado!")
        
        print("[Remote Manager Domo] Intentando descargar modelo frontal entrenado...")
        subprocess.run([lightning_cli, "studio", "cp", remote_yolo_lat, f"{local_models_dir}/"], check=True)
        print(f"[Remote Manager Domo] ¡Modelo frontal descargado!")
        
        # Descargar los logs (la carpeta train_runs entera que contiene las gráficas y métricas csv)
        print("[Remote Manager Domo] Descargando logs del entrenamiento (TensorBoard, imágenes y gráficas)...")
        subprocess.run([lightning_cli, "studio", "cp", "-r", remote_models_dir, f"{local_models_dir}/"], check=True)
        print("[Remote Manager Domo] ¡Todos los logs descargados exitosamente!")
        
    except subprocess.CalledProcessError:
        print("[Remote Manager Domo] Aún no se han encontrado todos los modelos finales. Es posible que el entrenamiento siga en curso.")

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
