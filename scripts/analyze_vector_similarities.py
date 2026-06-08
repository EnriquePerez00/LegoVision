# -*- coding: utf-8 -*-
"""scripts/analyze_vector_similarities.py
Analyzes DINOv2 embeddings stored in Supabase to find visual collisions
(different pieces with very similar vectors).
"""
import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from database import supabase_client

def main():
    print("Cargando embeddings desde la base de datos...")
    records = supabase_client.get_all_embeddings()
    if not records:
        print("[ERROR] No se encontraron embeddings en la base de datos. Por favor, indexa primero.")
        return

    print(f"Se cargaron {len(records)} embeddings.")
    
    # Agrupar por pieza y pose/ángulo
    embeddings = []
    metadata = []
    
    for r in records:
        emb = r.get("embedding") or r.get("embedding_projected")
        if emb is None:
            continue
        emb_arr = np.array(emb, dtype=np.float32)
        # Normalizar por si acaso
        norm = np.linalg.norm(emb_arr)
        if norm > 1e-6:
            emb_arr /= norm
        embeddings.append(emb_arr)
        metadata.append({
            "part_ref": r["part_ref"],
            "stable_face": r["stable_face"],
            "rotation_angle": r["rotation_angle"],
            "color_hex": r.get("color_hex", "N/A"),
            "pose_index": r.get("pose_index")
        })
        
    if not embeddings:
        print("[ERROR] Los embeddings encontrados no contienen vectores válidos.")
        return
        
    embeddings = np.array(embeddings)
    
    # Calcular matriz de similitud de coseno (dot product puesto que están normalizados)
    print("Calculando matriz de similitudes de coseno...")
    sim_matrix = np.dot(embeddings, embeddings.T)
    
    collisions = []
    n = len(metadata)
    
    for i in range(n):
        for j in range(i + 1, n):
            meta_i = metadata[i]
            meta_j = metadata[j]
            
            # Solo buscar colisiones entre piezas distintas
            if meta_i["part_ref"] != meta_j["part_ref"]:
                sim = float(sim_matrix[i, j])
                collisions.append({
                    "piece_a": meta_i["part_ref"],
                    "pose_a": meta_i["stable_face"],
                    "rot_a": meta_i["rotation_angle"],
                    "color_a": meta_i["color_hex"],
                    "piece_b": meta_j["part_ref"],
                    "pose_b": meta_j["stable_face"],
                    "rot_b": meta_j["rotation_angle"],
                    "color_b": meta_j["color_hex"],
                    "similarity": sim
                })
                
    # Ordenar por similitud decreciente
    collisions.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Generar reporte Markdown
    report_path = os.path.join(project_root, "docs", "vector_similarities_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Reporte de Similitudes y Colisiones Vectoriales DINOv2\n\n")
        f.write(f"Fecha de análisis: {np.datetime64('now')}\n")
        f.write(f"Total de embeddings analizados: {n}\n\n")
        
        f.write("## Top 30 Colisiones Más Críticas (Piezas Diferentes con Similitud Alta)\n")
        f.write("Estas piezas tienen vectores DINOv2 muy parecidos y podrían ser confundidas por el clasificador.\n\n")
        f.write("| Pieza A | Pose/Rot A | Color A | Pieza B | Pose/Rot B | Color B | Similitud Coseno |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        
        count = 0
        for col in collisions[:100]:
            # Evitar reportar duplicados casi idénticos de las mismas piezas si ya las listamos
            f.write(f"| `{col['piece_a']}` | Pose {col['pose_a']} / {col['rot_a']}° | #{col['color_a']} | `{col['piece_b']}` | Pose {col['pose_b']} / {col['rot_b']}° | #{col['color_b']} | **{col['similarity']:.4f}** |\n")
            count += 1
            if count >= 30:
                break
                
        f.write("\n## Conclusiones y Recomendaciones\n")
        if collisions and collisions[0]["similarity"] > 0.90:
            f.write("> [!WARNING]\n")
            f.write("> Se detectaron similitudes superiores a 0.90 entre algunas piezas diferentes.\n")
            f.write("> Esto indica riesgo de confusión visual. Recomendamos:\n")
            f.write("> 1. **Consenso Multicámara**: Asegurar que las cámaras laterales y cenitales ponderen juntas para resolver simetrías.\n")
            f.write("> 2. **Filtrado por Huella Física**: Utilizar la relación de aspecto (alto/ancho) para descartar candidatos de piezas con volúmenes muy diferentes.\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> La separación vectorial de DINOv2 es excelente en este set. No se aprecian colisiones críticas (< 0.90).\n")

    print(f"\n✅ Análisis completado. Reporte guardado en: {report_path}")
    if collisions:
        print(f"Máxima similitud encontrada entre piezas distintas: {collisions[0]['similarity']:.4f}")
        print(f"  Pieza A: {collisions[0]['piece_a']} (Pose {collisions[0]['pose_a']})")
        print(f"  Pieza B: {collisions[0]['piece_b']} (Pose {collisions[0]['pose_b']})")

if __name__ == "__main__":
    main()
