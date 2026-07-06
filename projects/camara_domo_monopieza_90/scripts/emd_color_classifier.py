# -*- coding: utf-8 -*-
"""
emd_color_classifier.py
================================================================================
Clasificador no paramétrico de color para el set 75078-1.
Implementa:
1. Clasificación por distancia Delta-E en el espacio CIELAB cromático.
2. Comparación de histogramas Lab (Bhattacharyya / Earth Mover's Distance)
   usando muestras del dataset de calibración.
================================================================================
"""

import os
import json
import numpy as np
import cv2

class ColorEMDClassifier:
    def __init__(self):
        # Colores de referencia del set 75078-1 (definiciones oficiales en RGB/HEX)
        self.ref_colors = {
            "1":  {"name": "White",             "hex": "#F9F9F9", "rgb": [249, 249, 249]},
            "11": {"name": "Black",             "hex": "#202020", "rgb": [32, 32, 32]},
            "13": {"name": "Trans-Brown",       "hex": "#625E51", "rgb": [98, 94, 81]},
            "17": {"name": "Trans-Red",         "hex": "#C91A09", "rgb": [201, 26, 9]},
            "85": {"name": "Dark Bluish Gray",  "hex": "#595D60", "rgb": [89, 93, 96]},
            "86": {"name": "Light Bluish Gray", "hex": "#A2A1A3", "rgb": [162, 161, 163]}
        }
        # Convertir referencias a CIELAB
        self.ref_lab = {}
        for code, info in self.ref_colors.items():
            self.ref_lab[code] = self._rgb_to_lab(info["rgb"])

    def _rgb_to_lab(self, rgb):
        # Conversión analítica estándar de sRGB a CIELAB
        r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
        r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
        g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
        b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505

        x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

        fx = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
        fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
        fz = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)

        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return np.array([L, a, b])

    def predict_delta_e(self, mean_lab):
        """Predice la clase de color calculando la menor distancia Delta-E en el espacio Lab."""
        if mean_lab is None:
            return "86", "Light Bluish Gray" # Fallback por defecto
            
        best_code = None
        min_dist = float("inf")
        
        for code, ref in self.ref_lab.items():
            # Distancia Delta-E clásica (Euclidiana en Lab)
            # Damos un ligero peso menor a L para ser más tolerantes a cambios de brillo
            dist = np.sqrt(0.3 * (mean_lab[0] - ref[0])**2 + (mean_lab[1] - ref[1])**2 + (mean_lab[2] - ref[2])**2)
            if dist < min_dist:
                min_dist = dist
                best_code = code
                
        return best_code, self.ref_colors[best_code]["name"]

    def compare_histograms(self, query_pixels_rgb):
        """Compara el histograma 3D Lab del query contra histogramas Gaussianos sintéticos."""
        if query_pixels_rgb is None or len(query_pixels_rgb) == 0:
            return "86", "Light Bluish Gray"

        # Convertir query a Lab usando OpenCV
        query_rgb_reshaped = query_pixels_rgb.reshape(-1, 1, 3).astype(np.uint8)
        query_lab = cv2.cvtColor(query_rgb_reshaped, cv2.COLOR_RGB2Lab).reshape(-1, 3)

        # Calcular histograma 3D en Lab (canales L:0, a:1, b:2)
        # Bins: 8 bins por canal para evitar esparcimiento por baja muestra
        hist_query = cv2.calcHist([query_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_query, hist_query, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        best_code = None
        min_distance = float("inf")

        for code, ref in self.ref_lab.items():
            # Construir histograma 3D sintético de referencia (distribución gaussiana estrecha en torno al Lab de la referencia)
            ref_pixels_lab = np.random.normal(loc=ref, scale=[2.0, 1.0, 1.0], size=(1000, 3))
            ref_pixels_lab = np.clip(ref_pixels_lab, 0, 255).astype(np.float32)
            
            hist_ref = cv2.calcHist([ref_pixels_lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_ref, hist_ref, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Comparación mediante distancia de Bhattacharyya (HISTCMP_BHATTACHARYYA)
            dist = cv2.compareHist(hist_query, hist_ref, cv2.HISTCMP_BHATTACHARYYA)
            if dist < min_distance:
                min_distance = dist
                best_code = code

        return best_code, self.ref_colors[best_code]["name"]
