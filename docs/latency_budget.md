# LegoVision — Análisis de Latencia y Velocidad de Cinta

Este documento describe los cálculos de velocidad de la cinta transportadora y el presupuesto de latencia asignado al módulo de inferencia.

## 1. Parámetros Físicos
*   **Velocidad de la cinta**: Variable, máximo de 5 m/min.
*   **Conversión a unidades métricas**:
    $$\text{Velocidad} = \frac{5\text{ metros}}{60\text{ segundos}} = 0.0833\text{ m/s} = 83.3\text{ mm/s}$$
*   **Campo de Visión (FOV) longitudinal**: 250 mm.

---

## 2. Tiempo de Tránsito
El tiempo total que una pieza tarda en cruzar completamente el campo de visión de la cámara se calcula como:
$$\text{Tiempo de tránsito} = \frac{\text{FOV}}{\text{Velocidad}} = \frac{250\text{ mm}}{83.3\text{ mm/s}} \approx 3.0\text{ segundos}$$

---

## 3. Presupuesto de Latencia
Para garantizar un mínimo de **3 capturas válidas** por pieza mientras cruza la cinta:
*   **Frecuencia mínima de inferencia**: 1 FPS (1 frame por segundo, equivale a 1 inferencia cada 1000ms).
*   **Frecuencia de diseño objetivo (Target)**: **5 FPS** (1 inferencia cada 200ms).
*   **Presupuesto de Inferencia por Frame**:
    *   **Inferencia YOLOv8 (Metal / MPS en M4)**: ~15-30 ms.
    *   **Captura de Frame (GigE Vision / USB3)**: ~10 ms.
    *   **Postprocesamiento y guardado en base de datos**: ~20 ms (asíncrono en FastAPI).
    *   **Latencia Total**: ~50 ms (muy por debajo del presupuesto de 200 ms).

Por tanto, el procesador M4 local tiene capacidad de sobra para procesar el flujo de la cinta en tiempo real a la velocidad máxima de 5 m/min.
