import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from core.db import supabase_client
from core.db.set_catalog import REAL_SETS

def run_migration():
    print("Iniciando migración de sets...")
    
    # 1. Inicializar las tablas
    try:
        supabase_client.init_database_tables()
    except Exception as e:
        print("Error inicializando tablas:", e)
        return
        
    # 2. Guardar cada set estático en la base de datos
    for set_id, set_data in REAL_SETS.items():
        print(f"Migrando set {set_id}: {set_data['name']}...")
        try:
            supabase_client.save_set_to_db(set_id, set_data)
        except Exception as e:
            print(f"Error migrando set {set_id}: {e}")
            
    print("Migración completada con éxito.")

if __name__ == "__main__":
    run_migration()
