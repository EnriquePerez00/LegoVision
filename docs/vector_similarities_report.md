# Reporte de Similitudes y Colisiones Vectoriales DINOv2

Fecha de análisis: 2026-06-03T05:58:03
Total de embeddings analizados: 2824

## Top 30 Colisiones Más Críticas (Piezas Diferentes con Similitud Alta)
Estas piezas tienen vectores DINOv2 muy parecidos y podrían ser confundidas por el clasificador.

| Pieza A | Pose/Rot A | Color A | Pieza B | Pose/Rot B | Color B | Similitud Coseno |
| --- | --- | --- | --- | --- | --- | --- |
| `3003` | Pose 1 / 300° | #A0A5A9 | `3039` | Pose 1 / 300° | #A0A5A9 | **0.9997** |
| `3037` | Pose 1 / 90° | #A0A5A9 | `3001` | Pose 1 / 90° | #A0A5A9 | **0.9996** |
| `3003` | Pose 1 / 45° | #A0A5A9 | `3039` | Pose 1 / 45° | #A0A5A9 | **0.9995** |
| `3037` | Pose 1 / 45° | #A0A5A9 | `3001` | Pose 1 / 45° | #A0A5A9 | **0.9994** |
| `3003` | Pose 1 / 315° | #A0A5A9 | `3039` | Pose 1 / 315° | #A0A5A9 | **0.9993** |
| `3037` | Pose 1 / 0° | #A0A5A9 | `3001` | Pose 1 / 0° | #A0A5A9 | **0.9993** |
| `3001` | Pose 1 / 60° | #A0A5A9 | `3037` | Pose 1 / 60° | #A0A5A9 | **0.9993** |
| `3003` | Pose 1 / 180° | #A0A5A9 | `3039` | Pose 1 / 180° | #A0A5A9 | **0.9993** |
| `3001` | Pose 1 / 195° | #A0A5A9 | `3037` | Pose 1 / 195° | #A0A5A9 | **0.9991** |
| `3001` | Pose 1 / 240° | #A0A5A9 | `3037` | Pose 1 / 240° | #A0A5A9 | **0.9991** |
| `3037` | Pose 1 / 270° | #A0A5A9 | `3001` | Pose 1 / 270° | #A0A5A9 | **0.9991** |
| `3003` | Pose 1 / 285° | #A0A5A9 | `3039` | Pose 1 / 285° | #A0A5A9 | **0.9991** |
| `3003` | Pose 1 / 30° | #A0A5A9 | `3039` | Pose 1 / 30° | #A0A5A9 | **0.9991** |
| `3001` | Pose 1 / 165° | #A0A5A9 | `3037` | Pose 1 / 165° | #A0A5A9 | **0.9991** |
| `3001` | Pose 1 / 15° | #A0A5A9 | `3037` | Pose 1 / 15° | #A0A5A9 | **0.9990** |
| `3037` | Pose 1 / 180° | #A0A5A9 | `3001` | Pose 1 / 180° | #A0A5A9 | **0.9990** |
| `3001` | Pose 1 / 345° | #A0A5A9 | `3037` | Pose 1 / 345° | #A0A5A9 | **0.9990** |
| `3003` | Pose 1 / 60° | #A0A5A9 | `3039` | Pose 1 / 60° | #A0A5A9 | **0.9990** |
| `3001` | Pose 1 / 285° | #A0A5A9 | `3037` | Pose 1 / 285° | #A0A5A9 | **0.9989** |
| `3037` | Pose 1 / 135° | #A0A5A9 | `3001` | Pose 1 / 135° | #A0A5A9 | **0.9989** |
| `3003` | Pose 1 / 270° | #A0A5A9 | `3039` | Pose 1 / 270° | #A0A5A9 | **0.9987** |
| `3039` | Pose 1 / 135° | #A0A5A9 | `3003` | Pose 1 / 135° | #A0A5A9 | **0.9987** |
| `3003` | Pose 1 / 0° | #A0A5A9 | `3039` | Pose 1 / 0° | #A0A5A9 | **0.9986** |
| `3001` | Pose 1 / 105° | #A0A5A9 | `3037` | Pose 1 / 105° | #A0A5A9 | **0.9985** |
| `3001` | Pose 1 / 75° | #A0A5A9 | `3037` | Pose 1 / 75° | #A0A5A9 | **0.9985** |
| `3003` | Pose 1 / 195° | #A0A5A9 | `3039` | Pose 1 / 195° | #A0A5A9 | **0.9985** |
| `3003` | Pose 1 / 165° | #A0A5A9 | `3039` | Pose 1 / 165° | #A0A5A9 | **0.9983** |
| `3001` | Pose 1 / 210° | #A0A5A9 | `3037` | Pose 1 / 210° | #A0A5A9 | **0.9982** |
| `3037` | Pose 1 / 225° | #A0A5A9 | `3001` | Pose 1 / 225° | #A0A5A9 | **0.9982** |
| `3003` | Pose 1 / 345° | #A0A5A9 | `3039` | Pose 1 / 345° | #A0A5A9 | **0.9982** |

## Conclusiones y Recomendaciones
> [!WARNING]
> Se detectaron similitudes superiores a 0.90 entre algunas piezas diferentes.
> Esto indica riesgo de confusión visual. Recomendamos:
> 1. **Consenso Multicámara**: Asegurar que las cámaras laterales y cenitales ponderen juntas para resolver simetrías.
> 2. **Filtrado por Huella Física**: Utilizar la relación de aspecto (alto/ancho) para descartar candidatos de piezas con volúmenes muy diferentes.
