# LegoVision — Especificaciones del Setup Físico e Industrial

Este documento detalla los componentes físicos del sistema de visión artificial y la configuración del hardware de adquisición de imágenes.

## 1. Cámara Industrial
*   **Sensor**: Sony IMX264 (2/3" CMOS)
*   **Resolución**: 5 Megapíxeles (2448 × 2048 px)
*   **Tamaño de Píxel**: 3.45 µm × 3.45 µm
*   **Interfaz**: GigE Vision o USB3 Vision
*   **Velocidad de captura**: Hasta 35 FPS en resolución completa.

---

## 2. Lente y Óptica
*   **Montura**: C-mount
*   **Distancia Focal**: 12.0 mm
*   **Distancia de Trabajo (WD)**: 355 mm (distancia recomendada desde el sensor hasta la cinta transportadora).
*   **Campo de Visión (FOV)**:
    *   **Horizontal**: ~250 mm
    *   **Vertical**: ~200 mm

---

## 3. Iluminación Industrial
Para minimizar sombras fuertes y reflejos especulares que dificulten la detección del plástico brillante de las piezas LEGO, se utiliza una configuración de iluminación difusa uniforme:
*   **Tipo**: Domo de luz LED industrial o luz anular difusa de alta frecuencia.
*   **Color**: Blanco Frío (~6000K).
*   **Montaje**: Coaxial o cenital rodeando la cámara, cubriendo un radio de ~180mm a una altura de Z = 300mm.

---

## 4. Cinta Transportadora
*   **Ancho de la cinta**: 200 mm
*   **Color**: Negro mate (opacidad >95% para maximizar el contraste con piezas de cualquier color).
*   **Velocidad**: Variable (máximo de 5 metros por minuto, equivalente a 83.3 mm/s).
*   **Sincronización**: Modo trigger por encoder óptico rotativo para disparar la cámara exactamente cuando la pieza ingresa al centro del FOV.
