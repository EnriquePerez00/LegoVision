import os
import sys
import subprocess
import argparse
import json
import time
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Cargar .env por defecto
load_dotenv(os.path.join(project_root, ".env"), override=True)

try:
    from lightning_sdk import Studio, Job, Machine
except ImportError:
    print("Por favor, instala lightning-sdk: pip install lightning-sdk")
    sys.exit(1)

# Disable SSL verification for Mac Python installations that complain
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

ACCOUNTS_FILE = os.path.join(project_root, "camara_domo", "scripts", "gpu_providers.json")

def load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return []
    with open(ACCOUNTS_FILE, "r") as f:
        return json.load(f)

def save_accounts(accounts):
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(accounts, f, indent=4)

def set_env_for_account(acc):
    """Inyecta las credenciales de la cuenta en el entorno local para el SDK y CLI."""
    os.environ["LIGHTNING_USER_ID"] = acc["user_id"]
    os.environ["LIGHTNING_API_KEY"] = acc["api_key"]
    if "LIGHTNING_TEAMSPACE" in os.environ:
        del os.environ["LIGHTNING_TEAMSPACE"]

def get_studio(acc, create_if_missing=False):
    studio_name = "legovision-t4"
    teamspace = acc["teamspace"]
    
    print(f"[Remote Manager Domo] Conectando a Studio '{studio_name}' en teamspace '{teamspace}'...")
    try:
        studio = Studio(name=studio_name, teamspace=teamspace, user=acc["username"])
    except Exception as e:
        print(f"[Remote Manager Domo] Fallback simple de Studio: {e}")
        studio = Studio(name=studio_name)
    
    return studio

def update_local_env(acc):
    """Actualiza el archivo .env para que los siguientes comandos usen la cuenta correcta."""
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        lines = f.readlines()
    
    with open(env_path, "w") as f:
        for line in lines:
            if line.startswith("LIGHTNING_API_KEY="):
                f.write(f"LIGHTNING_API_KEY={acc['api_key']}\n")
            elif line.startswith("LIGHTNING_USER_ID="):
                f.write(f"LIGHTNING_USER_ID={acc['user_id']}\n")
            elif line.startswith("LIGHTNING_TEAMSPACE="):
                f.write(f"LIGHTNING_TEAMSPACE={acc['teamspace']}\n")
            else:
                f.write(line)

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
                    rel_path = os.path.relpath(file_path, base_data_dir)
                    zipf.write(file_path, rel_path)
    print(f"[Remote Manager Domo] Compresión completada: {os.path.getsize(target_zip) / (1024*1024):.2f} MB")

import hashlib

def compute_dataset_hash(base_data_dir, folders_to_include):
    hasher = hashlib.md5()
    for folder_name in folders_to_include:
        folder_path = os.path.join(base_data_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
        for root, dirs, files in os.walk(folder_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_data_dir)
                stat = os.stat(file_path)
                hasher.update(f"{rel_path}_{stat.st_size}_{stat.st_mtime}".encode('utf-8'))
    return hasher.hexdigest()

def upload_dataset(acc, studio):
    local_data_dir = os.path.join(project_root, "camara_domo", "data")
    zip_path = os.path.join(local_data_dir, "domo_training_data.zip")
    folders_to_include = ["yolo_dataset_cenital", "yolo_dataset_frontal"]
    
    print("[Remote Manager Domo] Calculando hash del dataset local para comprobar si hay cambios...")
    current_hash = compute_dataset_hash(local_data_dir, folders_to_include)
    
    remote_hash_path = "camara_domo/data/domo_training_data_hash.txt"
    local_hash_temp = os.path.join(local_data_dir, "temp_remote_hash.txt")
    
    # Intentar descargar el hash remoto
    try:
        if os.path.exists(local_hash_temp):
            os.remove(local_hash_temp)
        studio.download_file(remote_hash_path, local_hash_temp)
        with open(local_hash_temp, "r") as f:
            remote_hash = f.read().strip()
    except Exception:
        remote_hash = None
        
    if current_hash == remote_hash:
        print("[Remote Manager Domo] El dataset en Lightning.ai ya está actualizado. Se omitirá la subida de datos.")
    else:
        print("[Remote Manager Domo] El dataset ha cambiado o no está subido. Comprimiendo datos...")
        zip_selective_data(local_data_dir, zip_path, folders_to_include)
        
        print(f"[Remote Manager Domo] Subiendo {zip_path} al Studio...")
        studio.upload_file(zip_path, "camara_domo/data/domo_training_data.zip")
        
        # Subir el nuevo hash
        local_hash_file = os.path.join(local_data_dir, "domo_training_data_hash.txt")
        with open(local_hash_file, "w") as f:
            f.write(current_hash)
        studio.upload_file(local_hash_file, remote_hash_path)
        print("[Remote Manager Domo] Subida de dataset y hash completada.")
    
    print("[Remote Manager Domo] Sincronizando scripts de entrenamiento...")
    script_path = os.path.join(project_root, "camara_domo", "scripts", "train_yolo_pose_remote.py")
    studio.upload_file(script_path, "camara_domo/scripts/train_yolo_pose_remote.py")

def run_on_lightning(acc):
    """Lógica de ejecución en Lightning AI."""
    print(f"\n[Remote Manager Domo] === Probando cuenta Lightning: {acc['email']} ===")
    set_env_for_account(acc)
    
    try:
        # Asegurarse de que el Studio existe (y si es nuevo, lo crea)
        studio = get_studio(acc, create_if_missing=True)
        
        # Como no sabemos si la máquina tiene los datos (puede ser una cuenta virgen o la anterior se corrompió)
        # Resubimos el dataset y el código por seguridad.
        upload_dataset(acc, studio)
        
        print("[Remote Manager Domo] Iniciando el Studio...")
        studio.start()
        
        print("[Remote Manager Domo] Descomprimiendo dataset y lanzando entrenamiento unificado...")
        remote_command = """bash -c '
        mkdir -p camara_domo/data
        unzip -q -o camara_domo/data/domo_training_data.zip -d camara_domo/data/
        pip install ultralytics python-dotenv
        mkdir -p camara_domo/models/train_runs
        python camara_domo/scripts/train_yolo_pose_remote.py > camara_domo/models/train_runs/training_job.log 2>&1
        '"""
        
        studio.run_and_detach(remote_command)
        
        # Si llegamos aquí sin excepciones, la cuenta tiene saldo y el comando se lanzó!
        job_name = "studio-run-detach"
        log_dir = os.path.join(project_root, "camara_domo", "logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "last_job_name.txt"), "w") as f:
            f.write(job_name)
            
        print(f"[Remote Manager Domo] Trabajo enviado con éxito en cuenta {acc['email']}.")
        print(f"[Remote Manager Domo] Nombre interno exacto del Job: {job_name}")
        
        # Actualizamos .env local para que los futuros --check funcionen automáticamente
        update_local_env(acc)
        return True
        
    except Exception as e:
        error_str = str(e).lower()
        print(f"[Remote Manager Domo] Error al lanzar en {acc['email']}: {e}")
        # Detectar si el error es de cuota/créditos
        if "credit" in error_str or "balance" in error_str or "fund" in error_str or "quota" in error_str or "insufficient" in error_str or "payment" in error_str or "billing" in error_str or "could not start" in error_str:
            print("[Remote Manager Domo] Detectado problema de créditos o arranque. Marcando cuenta como depleted.")
            acc["status"] = "depleted"
        else:
            print("[Remote Manager Domo] Error desconocido. Marcando cuenta como depleted temporalmente para hacer fallback.")
            acc["status"] = "depleted"
        return False

def run_on_kaggle(acc):
    """Lógica de ejecución en Kaggle."""
    print(f"\n[Remote Manager Domo] === Probando cuenta Kaggle: {acc['email']} ===")
    os.environ["KAGGLE_USERNAME"] = acc["username"]
    
    # Soporte para tokens modernos (KGAT_) vs Legacy API Keys
    if acc["api_key"].startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = acc["api_key"]
    else:
        os.environ["KAGGLE_KEY"] = acc["api_key"]
    
    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        print(f"[Remote Manager Domo] Error autenticando con Kaggle: {e}")
        acc["status"] = "depleted"
        return False
        
    local_data_dir = os.path.join(project_root, "camara_domo", "data")
    zip_path = os.path.join(local_data_dir, "domo_training_data.zip")
    folders_to_include = ["yolo_dataset_cenital", "yolo_dataset_frontal"]
    
    print("[Remote Manager Domo] Saltando compresión (usando el zip existente) para acelerar la subida...")
    # zip_selective_data(local_data_dir, zip_path, folders_to_include)
    
    dataset_name = "domo-training-data"
    dataset_id = f"{acc['username']}/{dataset_name}"
    dataset_dir = os.path.join(project_root, "camara_domo", "data", "kaggle_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    import shutil
    shutil.copy2(zip_path, os.path.join(dataset_dir, "domo_training_data.zip"))
    
    dataset_meta = {
        "title": "Domo Training Data",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(os.path.join(dataset_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_meta, f)
        
    try:
        print("[Remote Manager Domo] Actualizando dataset en Kaggle usando CLI...")
        kaggle_bin = os.path.join(project_root, ".venv", "bin", "kaggle")
        ret = os.system(f"{kaggle_bin} datasets version -p {dataset_dir} -m 'Update dataset' --dir-mode zip")
        if ret != 0:
            print("[Remote Manager Domo] El dataset no existe o falló, creando nuevo dataset usando CLI...")
            os.system(f"{kaggle_bin} datasets create -p {dataset_dir} --dir-mode zip")
    except Exception as e:
        print(f"[Remote Manager Domo] Error manejando dataset: {e}")
            
    print("[Remote Manager Domo] Preparando Kernel de Kaggle...")
    kernel_dir = os.path.join(project_root, "camara_domo", "scripts", "kaggle_kernel")
    os.makedirs(kernel_dir, exist_ok=True)
    
    # Intentamos obtener el ID del kernel existente (si lo hay) para no fallar con 409 Conflict
    # ya que Kaggle acorta enriqueperezbcn1973 a enriqueperezto
    real_username_slug = acc['username']
    try:
        import kaggle
        for k in kaggle.api.kernels_list(mine=True):
            if "domo-yolo-pose-train" in k.ref:
                real_username_slug = k.ref.split('/')[0]
                break
    except:
        pass
        
    kernel_meta = {
        "id": f"{real_username_slug}/domo-yolo-pose-train",
        "title": "Domo YOLO Pose Train",
        "code_file": "train_yolo_pose_remote.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [f"{real_username_slug}/{dataset_name}"],
        "competition_sources": [],
        "kernel_sources": []
    }
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_meta, f)
        
    import shutil
    shutil.copy2(os.path.join(project_root, "camara_domo", "scripts", "train_yolo_pose_remote.py"), 
                 os.path.join(kernel_dir, "train_yolo_pose_remote.py"))
        
    try:
        print("[Remote Manager Domo] Enviando trabajo (Kernel) a Kaggle...")
        try:
            # Try to pass the accelerator argument if the API supports it
            kaggle.api.kernels_push(kernel_dir, acc="NvidiaTeslaT4")
        except TypeError:
            # Fallback for older kaggle API versions that don't support the acc parameter
            kaggle.api.kernels_push(kernel_dir)
        print("[Remote Manager Domo] Trabajo enviado con éxito en Kaggle.")
        return True
    except Exception as e:
        print(f"[Remote Manager Domo] Error lanzando kernel en Kaggle: {e}")
        acc["status"] = "depleted"
        return False

def try_run_on_account(acc):
    """Intenta correr el proceso completo en una cuenta haciendo dispatch al provider."""
    provider = acc.get("provider", "lightning")
    if provider == "lightning":
        return run_on_lightning(acc)
    elif provider == "kaggle":
        return run_on_kaggle(acc)
    else:
        print(f"[Remote Manager Domo] Proveedor desconocido: {provider}")
        return False

def run_training_with_fallback():
    accounts = load_accounts()
    if not accounts:
        print("[Remote Manager Domo] No se encontró gpu_providers.json. Asegúrate de crearlo.")
        sys.exit(1)
        
    for acc in accounts:
        if acc.get("status") == "depleted":
            print(f"[Remote Manager Domo] Saltando cuenta agotada: {acc['email']}")
            continue
            
        success = try_run_on_account(acc)
        save_accounts(accounts) # Guardar el estado (depleted si falló)
        if success:
            print("[Remote Manager Domo] Proceso iniciado satisfactoriamente.")
            return
            
    print("[Remote Manager Domo] ERROR FATAL: Todas las cuentas disponibles se han agotado o han fallado.")

def check_lightning(active_acc):
    set_env_for_account(active_acc)
    studio = get_studio(active_acc, create_if_missing=False)
    
    local_models_dir = os.path.join(project_root, "camara_domo", "models")
    os.makedirs(local_models_dir, exist_ok=True)
    
    print("[Remote Manager Domo] Verificando estado del Studio...")
    
    remote_models_dir = "camara_domo/models/train_runs"
    remote_yolo_cen = "camara_domo/models/yolo_cenital_pose.pt"
    remote_yolo_lat = "camara_domo/models/yolo_frontal_pose.pt"
    
    try:
        print("[Remote Manager Domo] Intentando descargar modelo cenital entrenado...")
        studio.download_file(remote_yolo_cen, f"{local_models_dir}/yolo_cenital_pose.pt")
        print(f"[Remote Manager Domo] ¡Modelo cenital descargado!")
        
        print("[Remote Manager Domo] Intentando descargar modelo frontal entrenado...")
        studio.download_file(remote_yolo_lat, f"{local_models_dir}/yolo_frontal_pose.pt")
        print(f"[Remote Manager Domo] ¡Modelo frontal descargado!")
        
        print("[Remote Manager Domo] Descargando logs del entrenamiento...")
        # create train_runs dir if not exists
        os.makedirs(os.path.join(local_models_dir, "train_runs"), exist_ok=True)
        studio.download_folder(remote_models_dir, os.path.join(local_models_dir, "train_runs"))
        print("[Remote Manager Domo] ¡Todos los logs descargados exitosamente!")
        
    except Exception as e:
        print(f"[Remote Manager Domo] Aún no se han encontrado todos los modelos finales. Es posible que el entrenamiento siga en curso o haya fallado. Error: {e}")

def check_kaggle(active_acc):
    os.environ["KAGGLE_USERNAME"] = active_acc["username"]
    
    if active_acc["api_key"].startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = active_acc["api_key"]
    else:
        os.environ["KAGGLE_KEY"] = active_acc["api_key"]
    
    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        print(f"[Remote Manager Domo] Error autenticando con Kaggle: {e}")
        return
        
    # Buscar el ID real del kernel, ya que Kaggle puede reescribir el username (ej. enriqueperezto)
    kernels = kaggle.api.kernels_list(mine=True)
    kernel_id = None
    for k in kernels:
        if "domo-yolo-pose-train" in k.ref:
            kernel_id = k.ref
            break
            
    if not kernel_id:
        print("[Remote Manager Domo] No se encontró el Kernel 'domo-yolo-pose-train' en tu cuenta de Kaggle.")
        return
        
    print(f"[Remote Manager Domo] Verificando estado del Kernel en Kaggle ({kernel_id})...")
    
    try:
        status_res = kaggle.api.kernels_status(kernel_id)
        status = getattr(status_res, "status", "unknown")
        print(f"[Remote Manager Domo] Estado del Kernel en Kaggle: {status}")
        
        if status == "complete":
            print("[Remote Manager Domo] Descargando modelos y logs desde Kaggle...")
            local_models_dir = os.path.join(project_root, "camara_domo", "models")
            os.makedirs(local_models_dir, exist_ok=True)
            kaggle.api.kernels_output(kernel_id, path=local_models_dir)
            print("[Remote Manager Domo] ¡Modelos y logs descargados exitosamente desde Kaggle!")
            
            # TODO: Kaggle might put files in a slightly different tree depending on how we wrote them.
            # But they will be inside local_models_dir.
        elif status in ["error", "failed", "cancelled"]:
            print(f"[Remote Manager Domo] El Kernel ha fallado con estado: {status}")
            
    except Exception as e:
        print(f"[Remote Manager Domo] Error al consultar Kaggle: {e}")

def check_and_download():
    accounts = load_accounts()
    active_acc = None
    for acc in accounts:
        if acc.get("status") == "active":
            active_acc = acc
            break
            
    if not active_acc:
        print("[Remote Manager Domo] No hay cuentas activas para hacer el check.")
        return
        
    provider = active_acc.get("provider", "lightning")
    if provider == "lightning":
        check_lightning(active_acc)
    elif provider == "kaggle":
        check_kaggle(active_acc)
    else:
        print(f"[Remote Manager Domo] Proveedor desconocido: {provider}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["start", "upload", "run", "check", "full"], required=True, help="Acción a realizar en el Studio remoto")
    args = parser.parse_args()
    
    # Hemos simplificado para que 'start', 'upload', 'run' y 'full' dirijan al nuevo flujo multi-cuenta por robustez
    if args.action in ["start", "upload", "run", "full"]:
        run_training_with_fallback()
    elif args.action == "check":
        check_and_download()
