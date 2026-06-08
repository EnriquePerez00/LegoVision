# Algoritmo Semantico de Posiciones Estables

## Descripcion General
Determina teoricamente cuantas posiciones estables tiene cada pieza LEGO
basandose en la geometria 3D del fichero .dat de LDraw.

## Tabla BD: piece_embeddings
- part_ref: referencia LDraw (e.g. "3004")
- stable_face: 0=Top (studs arriba), 1=Side (de lado), 2=Bottom (invertida)
- rotation_angle: 0, 30, 60... 330 grados
- embedding: vector DINOv2 384-dim

El numero de stable_face distintos por part_ref = posiciones estables segun el algoritmo semantico.

## Scripts
- Generacion renders: data/tmp/ref_renders/part_{ref}_face{0,1,2}_rot{ang}.png
- Indexacion: training/index_embeddings.py
- Validacion experimental: scripts/validate_stable_poses.py
- Renders validacion: scripts/render_stable_poses_validation.py (NUEVO)
- Excel comparativo: scripts/generate_validation_excel.py (NUEVO)
- Directorio renders validacion: data/validation_renders/ (separado de ref_multiangle/)
- Output Excel: data/validation_report_{set_id}.xlsx

## Comparacion Semantico vs Experimental
Semantico (BD): clasificacion determinista 0/1/2 desde renders LDraw indexados con DINOv2
Experimental: simulacion fisica Blender, agrupacion por vector local_up < 15 grados
