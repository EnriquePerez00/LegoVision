# -*- coding: utf-8 -*-
"""
scripts/populate_topological_features.py
========================================

Orquesta el análisis topológico 3D de las piezas LEGO activas en la base de datos.
Parsea de manera recursiva los archivos LDraw (.dat) identificando primitivas
que correspondan a alguna de las 8 clases topológicas y actualiza el campo
`topological_features` en la tabla `lego_classes`.
"""

import os
import sys
import json
import logging

# Configurar logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("topological_features")

# Configurar path para imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))
sys.path.insert(0, os.path.join(project_root, "core", "db"))

from ldraw_mesh_parser import find_ldraw_file, _ref_fallback_candidates
from ldraw_primitives_map import PRIMITIVES_MAP_LOWER
from supabase_client import get_connection

def is_basic_primitive(resolved_path, sub_name):
    """
    Determina si un archivo LDraw es una primitiva geométrica básica no topológica
    (ej. círculos, rectángulos, líneas) para evitar la recursión innecesaria.
    """
    sub_name_lower = sub_name.lower().replace("\\", "/")
    if "/p/" in sub_name_lower or sub_name_lower.startswith("p/"):
        return True
    if resolved_path:
        resolved_path_lower = resolved_path.lower().replace("\\", "/")
        if "/p/" in resolved_path_lower:
            return True
    return False

def get_ldraw_part_name(path):
    """
    Extrae la primera línea del archivo LDraw para obtener una descripción/nombre legible.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("0 "):
                    name = line[2:].strip()
                    if name:
                        return name
    except Exception:
        pass
    return None

def parse_ldraw_topological_features(path, depth=0, seen=None):
    """
    Función recursiva para analizar las líneas de tipo 1 de un archivo .dat
    y acumular el conteo de las 8 características topológicas de interés.
    """
    if depth > 30:
        return {}
    if seen is None:
        seen = set()
    
    rp = os.path.realpath(path)
    if rp in seen:
        return {}
    seen.add(rp)

    counts = {
        "stud_solid": 0,
        "stud_hollow": 0,
        "technic_hole_round": 0,
        "technic_hole_cross": 0,
        "clip_jaw": 0,
        "bar_handle": 0,
        "bottom_tube": 0,
        "bottom_pin": 0
    }

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except Exception as e:
        logger.warning(f"No se pudo leer archivo {path}: {e}")
        return counts

    for raw in lines:
        p = raw.strip().split()
        if not p:
            continue
        tp = p[0]
        if tp == "1" and len(p) >= 15:
            sub_name = " ".join(p[14:])
            sub_bn = os.path.basename(sub_name).lower().strip()
            
            # 1. Comprobar si coincide con el diccionario de primitivas topológicas
            if sub_bn in PRIMITIVES_MAP_LOWER:
                feature = PRIMITIVES_MAP_LOWER[sub_bn]
                counts[feature] += 1
            else:
                # 2. Buscar archivo localmente
                sp = find_ldraw_file(sub_name)
                if sp:
                    # Si no coincide y no es primitiva geométrica básica, recursión
                    if not is_basic_primitive(sp, sub_name):
                        sub_counts = parse_ldraw_topological_features(sp, depth + 1, seen)
                        for k, v in sub_counts.items():
                            counts[k] += v
    return counts

def main():
    logger.info("Iniciando ETL de características topológicas...")
    
    conn = get_connection()
    if conn is None:
        logger.error("No se pudo conectar a la base de datos PostgreSQL/Supabase.")
        return

    try:
        # 1. Obtener la lista de todas las piezas en lego_classes
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ldraw_id FROM lego_classes;
            """)
            active_refs = [row["ldraw_id"] for row in cur.fetchall()]
        
        logger.info(f"Encontradas {len(active_refs)} referencias de piezas activas en la BD.")
        
        if not active_refs:
            logger.info("No hay referencias activas a procesar.")
            return

        # 2. Asegurar que las piezas activas existan en lego_classes (si no, las insertamos primero)
        with conn.cursor() as cur:
            for ref in active_refs:
                # Buscar archivo LDraw para extraer descripción/nombre
                part_name = f"Lego Piece {ref}"
                found_path = None
                for cand in _ref_fallback_candidates(ref):
                    p = find_ldraw_file(cand)
                    if p:
                        found_path = p
                        break
                if found_path:
                    extracted_name = get_ldraw_part_name(found_path)
                    if extracted_name:
                        part_name = extracted_name

                cur.execute("""
                    INSERT INTO lego_classes (ldraw_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (ldraw_id) DO NOTHING;
                """, (ref, part_name))
            conn.commit()
            logger.info("Verificación e inserción de catálogo lego_classes completada.")

        # 3. Procesar y parsear cada pieza
        results_to_update = []
        for idx, ref in enumerate(active_refs):
            # Encontrar el archivo de origen LDraw
            found_path = None
            for cand in _ref_fallback_candidates(ref):
                p = find_ldraw_file(cand)
                if p:
                    found_path = p
                    break
            
            if not found_path:
                logger.warning(f"[{idx+1}/{len(active_refs)}] Malla LDraw no encontrada para la pieza {ref}")
                continue

            # Parseo recursivo
            features = parse_ldraw_topological_features(found_path)
            
            # Limpiar entradas con conteo = 0 del JSON para mayor legibilidad y ahorro de espacio
            cleaned_features = {k: v for k, v in features.items() if v > 0}
            
            results_to_update.append((json.dumps(cleaned_features), ref))
            logger.info(f"[{idx+1}/{len(active_refs)}] Procesada pieza {ref}: {cleaned_features}")

        # 4. Actualizar en lotes de 100
        batch_size = 100
        total_updated = 0
        with conn.cursor() as cur:
            for i in range(0, len(results_to_update), batch_size):
                batch = results_to_update[i:i + batch_size]
                
                # Ejecutar lote de actualizaciones
                for feat_json, ref in batch:
                    cur.execute("""
                        UPDATE lego_classes
                        SET topological_features = %s
                        WHERE ldraw_id = %s;
                    """, (feat_json, ref))
                
                conn.commit()
                total_updated += len(batch)
                logger.info(f"Lote de actualización enviado a la BD. Progreso: {total_updated}/{len(results_to_update)}")

        logger.info(f"Proceso finalizado. Total de piezas enriquecidas con topología: {total_updated}")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error durante la ejecución del ETL: {e}", exc_info=True)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
