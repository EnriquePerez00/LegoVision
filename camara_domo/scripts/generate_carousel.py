# -*- coding: utf-8 -*-
"""
Generador de visualizador interactivo (carrusel / slider temporal)
para las cámaras cenital y frontal de camara_domo.
Optimizado con rutas absolutas de imagen, superposición de Bounding Boxes (BBoxes)
y cajas de visualización fijas.
"""
import json
import os
import sys

def main():
    metadata_path = "camara_domo/data/data100/simulation_metadata.json"
    output_html = "camara_domo/data/data100/report/carousel.html"
    
    if not os.path.exists(metadata_path):
        print(f"Error: No se encuentra {metadata_path}")
        sys.exit(1)
        
    print(f"Cargando metadatos desde {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Obtener la ruta absoluta de las imágenes
    img_dir_abs = os.path.abspath("camara_domo/data/data100")
    # Formatear el prefijo de URL de archivo absoluto para compatibilidad multiplataforma
    # En Windows, abspath puede tener barras invertidas; las reemplazamos por barras normales
    img_dir_abs_url = img_dir_abs.replace("\\", "/")
    if not img_dir_abs_url.startswith("/"):
        img_dir_abs_url = "/" + img_dir_abs_url
    file_url_prefix = f"file://{img_dir_abs_url}/"
        
    frames_compact = []
    for f in data.get("frames", []):
        pieces = []
        for p in f.get("visible_pieces", []):
            pieces.append({
                "ref": p.get("ref"),
                "color": p.get("color_name"),
                "bbox_cenital": p.get("bbox_cenital_norm"),
                "bbox_frontal": p.get("bbox_frontal_norm"),
                "x": round(p.get("x_belt_local_mm", 0), 1),
                "y": round(p.get("y_belt_local_mm", 0), 1)
            })
        frames_compact.append({
            "idx": f.get("frame_index"),
            "offset": round(f.get("belt_offset_mm", 0), 1),
            "img_cenital": file_url_prefix + f.get("file_name"),
            "img_frontal": file_url_prefix + f.get("file_name_frontal"),
            "pieces": pieces
        })
        
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Visualizador de BBoxes LEGO - Cenital vs Frontal</title>
    <style>
        :root {{
            --bg-primary: #090d16;
            --bg-secondary: #121829;
            --accent: #4f46e5;
            --accent-hover: #6366f1;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --glass-bg: rgba(18, 24, 41, 0.75);
            --glass-border: rgba(255, 255, 255, 0.08);
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        
        header {{
            background: #05080f;
            border-bottom: 1px solid var(--glass-border);
            padding: 12px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        header h1 {{
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(to right, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .main-container {{
            display: flex;
            flex: 1;
            padding: 20px;
            gap: 20px;
            overflow: hidden;
        }}
        
        .viewer-pane {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        
        .views-grid {{
            display: flex;
            gap: 20px;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
        }}
        
        .view-card {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            /* Forzar relación de aspecto exacta y tamaño cuadrado fijo */
            height: calc(100vh - 220px);
            width: calc(100vh - 220px);
            max-height: 600px;
            max-width: 600px;
            aspect-ratio: 1 / 1;
        }}
        
        .view-label {{
            padding: 8px 16px;
            background: rgba(0, 0, 0, 0.4);
            border-bottom: 1px solid var(--glass-border);
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #818cf8;
            z-index: 10;
        }}
        
        .image-wrapper {{
            flex: 1;
            position: relative;
            background: #020306;
            overflow: hidden;
        }}
        
        .image-wrapper img {{
            width: 100%;
            height: 100%;
            object-fit: cover; /* Cubrir completamente el contenedor cuadrado fijo */
            display: block;
        }}
        
        /* Contenedor de overlays de BBox absoluto */
        .bbox-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 5;
        }}
        
        .bbox-rect {{
            position: absolute;
            border: 2px solid #38bdf8;
            box-sizing: border-box;
            box-shadow: 0 0 4px rgba(0,0,0,0.5);
            transition: all 0.05s ease-out;
        }}
        
        .bbox-label {{
            position: absolute;
            top: -16px;
            left: -2px;
            background: rgba(0, 0, 0, 0.85);
            color: #fff;
            font-size: 8px;
            padding: 1px 4px;
            border-radius: 2px;
            white-space: nowrap;
            border: 1px solid rgba(255, 255, 255, 0.15);
            font-family: monospace;
        }}
        
        .sidebar-pane {{
            width: 260px;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }}
        
        .section-title {{
            font-size: 0.8rem;
            font-weight: 700;
            color: #f3f4f6;
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 4px;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 0.8rem;
        }}
        
        .metric-val {{
            font-family: monospace;
            font-weight: bold;
            color: #60a5fa;
        }}
        
        /* Control Bar styling */
        .controls-container {{
            background: #05080f;
            border-top: 1px solid var(--glass-border);
            padding: 12px 40px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .timeline-wrapper {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .timeline-slider {{
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: #27272a;
            border-radius: 4px;
            outline: none;
            cursor: pointer;
        }}
        
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent);
            cursor: pointer;
            border: 2px solid #fff;
        }}
        
        .timeline-slider::-webkit-slider-thumb:hover {{
            background: var(--accent-hover);
        }}
        
        .buttons-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .playback-controls {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        button {{
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--glass-border);
            padding: 6px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        button:hover {{
            background: var(--accent);
            border-color: var(--accent);
        }}
        
        .direction-toggle {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            color: #818cf8;
            cursor: pointer;
            background: rgba(79, 70, 229, 0.08);
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid rgba(79, 70, 229, 0.15);
        }}
        
        .keyboard-hint {{
            font-size: 0.7rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>

    <header>
        <div>
            <h1>Simulador Dinámico de Cinta LEGO (BBoxes Activos)</h1>
            <div class="subtitle">Control Manual de Cinta a 5 m/s</div>
        </div>
        <div class="keyboard-hint">
            Mover: <b>← / A</b> y <b>→ / D</b>
        </div>
    </header>
    
    <div class="main-container">
        <!-- Visualizadores Fijos Cuadrados -->
        <div class="viewer-pane">
            <div class="views-grid">
                <!-- Cámara Cenital -->
                <div class="view-card">
                    <div class="view-label">Cámara Cenital (X=0)</div>
                    <div class="image-wrapper">
                        <img id="img-cenital" src="" alt="Cámara Cenital">
                        <div id="overlay-cenital" class="bbox-overlay"></div>
                    </div>
                </div>
                
                <!-- Cámara Frontal -->
                <div class="view-card">
                    <div class="view-label">Cámara Frontal (X=23.3 cm)</div>
                    <div class="image-wrapper">
                        <img id="img-frontal" src="" alt="Cámara Frontal">
                        <div id="overlay-frontal" class="bbox-overlay"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Sidebar Simplificado sin lista de piezas visibles -->
        <div class="sidebar-pane">
            <div>
                <div class="section-title">Parámetros Físicos</div>
                <div class="metric-row">
                    <span>Velocidad Cinta</span>
                    <span class="metric-val">5.0 m/s</span>
                </div>
                <div class="metric-row">
                    <span>Ancho Cinta</span>
                    <span class="metric-val">200 mm</span>
                </div>
                <div class="metric-row">
                    <span>Lente Cenital</span>
                    <span class="metric-val">55.0 mm</span>
                </div>
                <div class="metric-row">
                    <span>Lente Frontal</span>
                    <span class="metric-val">55.0 mm</span>
                </div>
            </div>
            
            <div>
                <div class="section-title">Telemetría de Frame</div>
                <div class="metric-row">
                    <span>Fotograma</span>
                    <span id="label-frame" class="metric-val">000 / 000</span>
                </div>
                <div class="metric-row">
                    <span>Offset Cinta</span>
                    <span id="label-offset" class="metric-val">0.0 mm</span>
                </div>
                <div class="metric-row">
                    <span>Piezas en FoV</span>
                    <span id="label-count" class="metric-val">0</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Barra de Control -->
    <div class="controls-container">
        <div class="timeline-wrapper">
            <span style="font-size:0.75rem; font-family:monospace; color: var(--text-secondary);" id="lbl-slider-start">Inicio</span>
            <input type="range" id="timeline" class="timeline-slider" min="0" max="0" value="0">
            <span style="font-size:0.75rem; font-family:monospace; color: var(--text-secondary);" id="lbl-slider-end">Fin</span>
        </div>
        
        <div class="buttons-row">
            <div class="playback-controls">
                <button id="btn-prev10">⏪ -10</button>
                <button id="btn-prev">◀ Anterior</button>
                <button id="btn-next">Siguiente ▶</button>
                <button id="btn-next10">+10 ⏩</button>
            </div>
            
            <label class="direction-toggle">
                <input type="checkbox" id="chk-invert" checked>
                <span>Invertir Sentido (Aproximación)</span>
            </label>
        </div>
    </div>

    <script>
        const originalFrames = {json.dumps(frames_compact, ensure_ascii=False)};
        const totalFrames = originalFrames.length;
        
        let currentSliderVal = 0;
        let invertFlow = true;
        
        // Elementos DOM
        const imgCenital = document.getElementById('img-cenital');
        const imgFrontal = document.getElementById('img-frontal');
        const overlayCenital = document.getElementById('overlay-cenital');
        const overlayFrontal = document.getElementById('overlay-frontal');
        
        const timeline = document.getElementById('timeline');
        const labelFrame = document.getElementById('label-frame');
        const labelOffset = document.getElementById('label-offset');
        const labelCount = document.getElementById('label-count');
        const chkInvert = document.getElementById('chk-invert');
        
        const btnPrev10 = document.getElementById('btn-prev10');
        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');
        const btnNext10 = document.getElementById('btn-next10');
        
        const lblSliderStart = document.getElementById('lbl-slider-start');
        const lblSliderEnd = document.getElementById('lbl-slider-end');
        
        timeline.max = totalFrames - 1;
        
        function getFrameForSliderValue(val) {{
            return invertFlow ? originalFrames[totalFrames - 1 - val] : originalFrames[val];
        }}
        
        // Genera colores distintos para los BBoxes
        function getBBoxColor(ref) {{
            const colors = ['#f43f5e', '#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899', '#14b8a6'];
            const val = parseInt(ref) || 0;
            return colors[val % colors.length];
        }}
        
        function drawBBoxes(pieces, overlay, type) {{
            overlay.innerHTML = '';
            pieces.forEach(p => {{
                const bbox = type === 'cenital' ? p.bbox_cenital : p.bbox_frontal;
                if (!bbox) return;
                
                // Formato: [xmin, ymin, xmax, ymax]
                const xmin = bbox[0];
                const ymin = bbox[1];
                const xmax = bbox[2];
                const ymax = bbox[3];
                
                const width = xmax - xmin;
                const height = ymax - ymin;
                
                // Si la caja no tiene área, omitir
                if (width <= 0 || height <= 0) return;
                
                const color = getBBoxColor(p.ref);
                
                const rect = document.createElement('div');
                rect.className = 'bbox-rect';
                rect.style.left = (xmin * 100) + '%';
                rect.style.top = (ymin * 100) + '%';
                rect.style.width = (width * 100) + '%';
                rect.style.height = (height * 100) + '%';
                rect.style.borderColor = color;
                
                const label = document.createElement('div');
                label.className = 'bbox-label';
                label.style.backgroundColor = color;
                label.textContent = p.ref + ' | ' + p.color;
                rect.appendChild(label);
                
                overlay.appendChild(rect);
            }});
        }}
        
        function updateUI() {{
            const frame = getFrameForSliderValue(currentSliderVal);
            if (!frame) return;
            
            // Actualizar imágenes
            imgCenital.src = frame.img_cenital;
            imgFrontal.src = frame.img_frontal;
            
            // Dibujar BBoxes
            drawBBoxes(frame.pieces, overlayCenital, 'cenital');
            drawBBoxes(frame.pieces, overlayFrontal, 'frontal');
            
            timeline.value = currentSliderVal;
            
            labelFrame.textContent = 'F-' + String(frame.idx).padStart(3, '0') + ' (' + (currentSliderVal + 1) + '/' + totalFrames + ')';
            labelOffset.textContent = frame.offset.toFixed(1) + ' mm';
            labelCount.textContent = frame.pieces.length;
            
            if (invertFlow) {{
                lblSliderStart.textContent = "Lejos (F-" + String(originalFrames[totalFrames-1].idx).padStart(3, '0') + ")";
                lblSliderEnd.textContent = "Cerca (F-000)";
            }} else {{
                lblSliderStart.textContent = "Inicio (F-000)";
                lblSliderEnd.textContent = "Fin (F-" + String(originalFrames[totalFrames-1].idx).padStart(3, '0') + ")";
            }}
        }}
        
        function changeSliderVal(delta) {{
            let newVal = currentSliderVal + delta;
            if (newVal < 0) newVal = 0;
            if (newVal >= totalFrames) newVal = totalFrames - 1;
            currentSliderVal = newVal;
            updateUI();
        }}
        
        timeline.addEventListener('input', (e) => {{
            currentSliderVal = parseInt(e.target.value);
            updateUI();
        }});
        
        chkInvert.addEventListener('change', (e) => {{
            invertFlow = e.target.checked;
            currentSliderVal = totalFrames - 1 - currentSliderVal;
            updateUI();
        }});
        
        btnPrev10.addEventListener('click', () => changeSliderVal(-10));
        btnPrev.addEventListener('click', () => changeSliderVal(-1));
        btnNext.addEventListener('click', () => changeSliderVal(1));
        btnNext10.addEventListener('click', () => changeSliderVal(10));
        
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{
                changeSliderVal(-1);
            }} else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {{
                changeSliderVal(1);
            }}
        }});
        
        invertFlow = chkInvert.checked;
        updateUI();
    </script>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Visualizador manual generado con éxito en: {output_html}")

if __name__ == '__main__':
    main()
