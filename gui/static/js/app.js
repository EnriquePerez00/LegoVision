// =============================================================
// LegoVision — Dashboard Logic, Belt Simulation & API Integration
// =============================================================

const API_BASE = "http://localhost:8005";

// ── Estado global ──
let sessionActive    = false;
let sessionId        = null;
let simulationRafId  = null;   // requestAnimationFrame handle
let lastInferenceTime = 0;     // para throttle de 500ms

// Bboxes detectadas en el último frame: [{x1,y1,x2,y2, label, conf}]
let activeBboxes = [];

// Umbral de confianza de detección YOLO (controlado por slider)
let liveConfThreshold = 0.30;

// Historial de loss/mAP para el chart de entrenamiento
const chartHistory = { epochs: [], loss: [], map50: [] };

// Mapeo de códigos de color LDraw → BrickLink (para construir URLs de imagen)
const LDRAW_TO_BRICKLINK_COLOR = {
    "0": "11",   // Black
    "1": "7",    // Blue
    "4": "5",    // Red
    "14": "3",   // Yellow
    "15": "1",   // White
    "84": "85",  // Dark Bluish Gray
    "85": "86",  // Light Bluish Gray
    "36": "17",  // Trans-Red
    "2": "6",    // Green
    "10": "10",  // Bright Green
    "25": "8",   // Orange
    "26": "26",  // Pink
    "27": "34",  // Lime Green
    "17": "40",  // Trans-Clear
    "73": "42",  // Medium Blue
};

// =============================================================
// 1. Inicialización
// =============================================================
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    checkDatabaseConnection();
    startStatsPolling();
    initLiveView();
    initSessionControls();
    initSetSearch();
    initDrawer();
    initTrainingPanel();
    initBeltSpeedSlider();
    initLiveConfSlider();
    initHistoryView();
    initTargetSetSelector();
});

// =============================================================
// 2. Navegación entre vistas
// =============================================================
function initNavigation() {
    const navItems  = document.querySelectorAll(".nav-item");
    const pageViews = document.querySelectorAll(".page-view");

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            navItems.forEach(n => n.classList.remove("active"));
            item.classList.add("active");

            const target = item.getAttribute("data-view");
            pageViews.forEach(v => {
                v.classList.remove("active");
                if (v.id === `view-${target}`) v.classList.add("active");
            });
        });
    });
}

// =============================================================
// 3. Conexión a base de datos
// =============================================================
function checkDatabaseConnection() {
    const dot  = document.getElementById("db-status-dot");
    const text = document.getElementById("db-status-text");

    window.addEventListener("pywebviewready", async () => {
        try {
            const connected = await pywebview.api.check_connection();
            if (connected) {
                dot.className  = "dot online";
                text.innerText = "Base de datos Online";
            } else {
                dot.className  = "dot";
                text.innerText = "Error en Base de datos";
            }
        } catch (e) { console.error("Bridge error:", e); }
    });
}

// =============================================================
// 4. Polling de estadísticas del dashboard
// =============================================================
function startStatsPolling() {
    setInterval(updateStats, 4000);
    window.addEventListener("pywebviewready", updateStats);
}

async function updateStats() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const stats = await pywebview.api.get_historical_stats();
        if (stats.status !== "success") return;

        renderDetectionsList(stats.recent_detections);
        renderTopClasses(stats.top_classes);

        const total = stats.recent_detections.length;
        document.getElementById("metric-total-pieces").innerText = total;

        if (total > 0) {
            const avgConf = stats.recent_detections.reduce((a, c) => a + c.confidence, 0) / total;
            document.getElementById("metric-confidence").innerText = `${(avgConf * 100).toFixed(1)}%`;
            const unique = new Set(stats.recent_detections.map(d => d.piece_class)).size;
            document.getElementById("metric-classes").innerText = unique;
        }
    } catch (e) { console.error("Stats error:", e); }
}

function renderDetectionsList(detections) {
    const list = document.getElementById("recent-detections-list");
    if (!detections || detections.length === 0) {
        list.innerHTML = `<div class="recent-item">No hay detecciones recientes</div>`;
        return;
    }
    list.innerHTML = detections.map(d => {
        const cls     = d.confidence > 0.75 ? "high" : "low";
        const dateStr = d.detected_at ? new Date(d.detected_at).toLocaleTimeString() : "N/A";
        return `
            <div class="recent-item">
                <div class="piece-info">
                    <span class="piece-class">Pieza ${d.piece_class}</span>
                    <span class="piece-name">${d.piece_name || "Sin descripción"} — ${dateStr}</span>
                </div>
                <span class="piece-conf ${cls}">${(d.confidence * 100).toFixed(0)}%</span>
            </div>`;
    }).join("");
}

function renderTopClasses(topClasses) {
    const list = document.getElementById("top-classes-list");
    if (!topClasses || topClasses.length === 0) {
        list.innerHTML = `<div class="recent-item">Esperando datos de entrenamiento</div>`;
        return;
    }
    list.innerHTML = topClasses.map(c => `
        <div class="recent-item">
            <div class="piece-info">
                <span class="piece-class">Pieza ${c.piece_class}</span>
                <span class="piece-name">${c.piece_name || "Clase Lego"}</span>
            </div>
            <span class="piece-conf high">${c.count} uds</span>
        </div>`).join("");
}

// =============================================================
// 5. SIMULACIÓN DE CINTA TRANSPORTADORA (canvas)
// =============================================================

// Paleta de colores LEGO reales
const LEGO_COLORS = [
    { name: "Rojo",     hex: "#C91A09" },
    { name: "Azul",     hex: "#1C6DCD" },
    { name: "Amarillo", hex: "#F2CD37" },
    { name: "Verde",    hex: "#257A3E" },
    { name: "Naranja",  hex: "#FE8A18" },
    { name: "Blanco",   hex: "#F4F4F4" },
    { name: "Negro",    hex: "#1B2A34" },
    { name: "Gris",     hex: "#9BA19D" },
    { name: "Morado",   hex: "#6C468A" },
    { name: "Marrón",   hex: "#7B5B3A" },
];

// Piezas del Set 75078-1 (Star Wars Rebels Troop Transport) completas
const SET_75078_1_PARTS = [
    { ref: "sw0614", name: "Stormtrooper (Rebels)", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", qty: 4 },
    { ref: "3004", name: "Brick 1x2", w: 28, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 8 },
    { ref: "3001", name: "Brick 2x4", w: 56, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3020", name: "Plate 2x4", w: 56, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 4 },
    { ref: "3022", name: "Plate 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 12 },
    { ref: "2877", name: "Brick 1x2 Grille", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 4 },
    { ref: "59900", name: "Cone 1x1", w: 14, h: 14, color: "#C91A09", colorName: "Rojo Trans.", qty: 4 },
    { ref: "3003", name: "Brick 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 }
];

// Fallback inventories for dropdown options in pure client-side mode
const LOCAL_SET_CATALOG = {
    "75078-1": {
        name: "Imperial Troop Transport (Star Wars Rebels)",
        parts: [...SET_75078_1_PARTS]
    },
    "75280-1": {
        name: "501st Legion Clone Troopers",
        parts: [
            { ref: "sw1093", name: "501st Legion Clone Trooper", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", qty: 3 },
            { ref: "sw1094", name: "501st Legion Jet Trooper", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", qty: 1 },
            { ref: "3023", name: "Plate 1x2", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 15 },
            { ref: "3024", name: "Plate 1x1", w: 14, h: 14, color: "#0A3C9F", colorName: "Azul", qty: 20 },
            { ref: "2420", name: "Plate 2x2 Corner", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 6 },
            { ref: "3003", name: "Brick 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 8 }
        ]
    },
    "75218-1": {
        name: "X-Wing Starfighter",
        parts: [
            { ref: "sw0949", name: "Luke Skywalker (Pilot)", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", qty: 1 },
            { ref: "sw0950", name: "Biggs Darklighter (Pilot)", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", qty: 1 },
            { ref: "3020", name: "Plate 2x4", w: 56, h: 28, color: "#FFFFFF", colorName: "Blanco", qty: 12 },
            { ref: "3022", name: "Plate 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 25 },
            { ref: "3023", name: "Plate 1x2", w: 28, h: 14, color: "#C91A09", colorName: "Rojo", qty: 18 },
            { ref: "3001", name: "Brick 2x4", w: 56, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 10 }
        ]
    },
    "75337-1": {
        name: "AT-TE Walker",
        parts: [
            { ref: "sw1219", name: "Commander Cody (Phase 2)", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", qty: 1 },
            { ref: "sw1220", name: "212th Attack Battalion Clone Trooper", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", qty: 3 },
            { ref: "3001", name: "Brick 2x4", w: 56, h: 28, color: "#5A5A5A", colorName: "Gris Oscuro", qty: 15 },
            { ref: "3020", name: "Plate 2x4", w: 56, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 30 },
            { ref: "3022", name: "Plate 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 50 },
            { ref: "3003", name: "Brick 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 20 }
        ]
    }
};

let currentSetId = "75078-1";
let currentSetName = "Imperial Troop Transport (Star Wars Rebels)";
let currentSetParts = [...SET_75078_1_PARTS];

let simulationSpawnPool = [];
let sessionIdentifiedCounts = {};
let spawnedCount = 0;
let exitedCount = 0;

// Inicializar el conteo a 0 para cada ref
function resetSessionIdentifiedCounts() {
    sessionIdentifiedCounts = {};
    currentSetParts.forEach(p => {
        sessionIdentifiedCounts[p.ref] = 0;
    });
    updateSessionInventoryTable();
}

function refillSimulationSpawnPool() {
    simulationSpawnPool = [];
    currentSetParts.forEach(part => {
        for (let i = 0; i < part.qty; i++) {
            simulationSpawnPool.push({ ...part });
        }
    });
    // Barajar (Fisher-Yates)
    for (let i = simulationSpawnPool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [simulationSpawnPool[i], simulationSpawnPool[j]] = [simulationSpawnPool[j], simulationSpawnPool[i]];
    }
}

function updateSessionInventoryTable() {
    const tbody = document.getElementById("session-inventory-tbody");
    if (!tbody) return;

    tbody.innerHTML = currentSetParts.map(p => {
        const count = sessionIdentifiedCounts[p.ref] || 0;
        let statusBadge = "";
        
        if (count === p.qty) {
            statusBadge = `<span class="badge-status ok">✓ Completo</span>`;
        } else if (count < p.qty) {
            statusBadge = `<span class="badge-status less">⚠️ Falta ${p.qty - count} (De menos)</span>`;
        } else {
            statusBadge = `<span class="badge-status more">🚨 Exceso +${count - p.qty} (De más)</span>`;
        }

        return `
            <tr>
                <td style="padding: 10px; font-weight: bold; color: var(--accent);">${p.ref}</td>
                <td style="padding: 10px; color: #fff;">${p.name} (${p.colorName})</td>
                <td style="padding: 10px; text-align: center; color: var(--text-secondary); font-weight: bold;">${p.qty}</td>
                <td style="padding: 10px; text-align: center; color: #fff; font-size: 1rem; font-weight: bold;">${count}</td>
                <td style="padding: 10px; text-align: center;">${statusBadge}</td>
            </tr>
        `;
    }).join("");
}

let canvas, ctx;
let beltPieces = [];   // array de piezas en la simulación
let beltSpeed  = 4.0;  // m/min → px/frame calculado en spawnPiece

// píxeles por metro en la simulación (FOV 250mm = canvas height)
const FOV_MM   = 250;
const CANVAS_H = 480;
const CANVAS_W = 640;
const PX_PER_MM = CANVAS_H / FOV_MM;     // ≈ 1.92 px/mm
const MM_PER_MIN_TO_PX_PER_FRAME = PX_PER_MM / 60 / 30; // @ 30fps

function initLiveView() {
    canvas = document.getElementById("live-canvas");
    ctx    = canvas.getContext("2d");
    canvas.width  = CANVAS_W;
    canvas.height = CANVAS_H;
    drawBeltBackground();

    // Click en el canvas → clasificar la pieza en la bbox
    canvas.addEventListener("click", onCanvasClick);

    resetSessionIdentifiedCounts();
}

function drawBeltBackground() {
    // Fondo oscuro tipo cinta
    const grad = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
    grad.addColorStop(0,   "#0e1220");
    grad.addColorStop(0.5, "#111827");
    grad.addColorStop(1,   "#0e1220");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Textura de cinta: líneas de guía laterales
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth   = 2;
    for (let x = 20; x < CANVAS_W; x += 60) {
        ctx.beginPath();
        ctx.moveTo(x, 0); ctx.lineTo(x, CANVAS_H);
        ctx.stroke();
    }

    // Línea central de referencia
    ctx.strokeStyle = "rgba(37,99,235,0.08)";
    ctx.lineWidth   = 1;
    ctx.setLineDash([12, 8]);
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, 0); ctx.lineTo(CANVAS_W / 2, CANVAS_H);
    ctx.stroke();
    ctx.setLineDash([]);

    // Bordes laterales de la cinta
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(8, 0);          ctx.lineTo(8, CANVAS_H);
    ctx.moveTo(CANVAS_W - 8, 0); ctx.lineTo(CANVAS_W - 8, CANVAS_H);
    ctx.stroke();
}

const GRAVITY = 0.45;

function updatePiecePhysics(p, vy) {
    if (p.z > 0) {
        // En el aire: cae por gravedad y avanza a menor velocidad de cinta (inercia/resistencia del aire)
        p.vz -= GRAVITY;
        p.z += p.vz;
        p.y += vy * 0.35; // velocidad de avance Y reducida en el aire
        p.angle += p.spin;

        if (p.z <= 0) {
            // Colisión con la cinta
            p.z = 0;
            p.vz = -p.vz * p.restitution;

            // Perturbación física al rebotar (desplazamientos y giros bruscos)
            p.spin = (Math.random() - 0.5) * 0.08;
            p.angle += (Math.random() - 0.5) * 0.15;
            p.x += (Math.random() - 0.5) * 6;
            p.y += (Math.random() - 0.5) * 4;

            // Mantener dentro de los límites de la cinta
            const margin = 20 + Math.max(p.w, p.h) / 2;
            p.x = Math.max(margin, Math.min(CANVAS_W - margin, p.x));

            // Si el rebote es muy leve, se asienta por completo
            if (Math.abs(p.vz) < 0.6) {
                p.z = 0;
                p.vz = 0;
                p.spin = (Math.random() - 0.5) * 0.002;
            }
        }
    } else {
        // Asentada en la cinta: avanza y vibra
        p.y += vy;
        p.angle += p.spin;

        // Micro-vibraciones mecánicas por el motor de la cinta (traqueteo)
        p.x += (Math.random() - 0.5) * 0.35;
        p.y += (Math.random() - 0.5) * 0.35;
        p.angle += (Math.random() - 0.5) * 0.001;

        // Mantener dentro de los límites de la cinta
        const margin = 20 + Math.max(p.w, p.h) / 2;
        p.x = Math.max(margin, Math.min(CANVAS_W - margin, p.x));
    }
}

function spawnPiece() {
    if (simulationSpawnPool.length === 0) {
        refillSimulationSpawnPool();
    }
    const part = simulationSpawnPool.pop() || currentSetParts[0];
    const angle = Math.random() * Math.PI * 2;  // rotación aleatoria 360°

    // Distribuir piezas en X dentro de la cinta
    const margin = 20 + Math.max(part.w, part.h) / 2;
    const x = margin + Math.random() * (CANVAS_W - 2 * margin);

    return {
        x, 
        y: -100 - Math.random() * 80,  // Nace más arriba para que caiga y rebote antes de entrar en el campo visual
        w: part.w, h: part.h,
        color: part.color,
        colorName: part.colorName,
        ref: part.ref,
        name: part.name,
        angle,
        vy: beltSpeed * 1000 / 60 / 30 * PX_PER_MM,
        spin: (Math.random() - 0.5) * 0.05, // Rotación activa en el aire
        isDetected: false,
        hasBeenCounted: false,
        z: 140 + Math.random() * 60, // altura inicial de caída
        vz: 0,
        restitution: 0.2 + Math.random() * 0.15 // rebote
    };
}

function initBeltPieces() {
    refillSimulationSpawnPool();
    beltPieces = [];
    // Pre-poblar con 6 piezas — solapamiento máximo permitido: 20%
    const MIN_INIT_DIST = 0; // sin margen extra; la repulsión limita el solapamiento a ≤20%
    for (let i = 0; i < 6; i++) {
        let p;
        let placed = false;
        for (let attempt = 0; attempt < 30; attempt++) {
            p = spawnPiece();
            p.z = 0;
            p.vz = 0;
            p.spin = (Math.random() - 0.5) * 0.002;
            p.y = 40 + Math.random() * (CANVAS_H - 80);
            let ok = true;
            for (const bp of beltPieces) {
                // Sólo rechazar si hay más del 20% de solapamiento
                const minD = (Math.max(bp.w, bp.h) + Math.max(p.w, p.h)) / 2 * 0.80;
                if (Math.hypot(bp.x - p.x, bp.y - p.y) < minD) { ok = false; break; }
            }
            if (ok) { placed = true; break; }
        }
        if (placed) beltPieces.push(p);
    }
}

function drawPiece(p) {
    // 1. Proyectar sombra sobre la cinta (z = 0)
    // El desplazamiento y desenfoque de la sombra dependen de la altura z
    const shadowOpacity = Math.max(0.08, 0.45 - p.z * 0.0025);
    const shadowShiftX = p.z * 0.18;
    const shadowShiftY = 3 + p.z * 0.26;
    const shadowBlur = 4 + p.z * 0.14;

    ctx.save();
    ctx.translate(p.x + shadowShiftX, p.y + shadowShiftY);
    ctx.rotate(p.angle);
    ctx.fillStyle = `rgba(0, 0, 0, ${shadowOpacity})`;
    ctx.shadowColor = `rgba(0, 0, 0, ${shadowOpacity})`;
    ctx.shadowBlur = shadowBlur;
    
    // Forma de la sombra basada en el cuerpo de la pieza
    const r = 3;
    ctx.beginPath();
    ctx.roundRect(-p.w / 2, -p.h / 2, p.w, p.h, r);
    ctx.fill();
    ctx.restore();

    // 2. Dibujar el cuerpo principal de la pieza (escalado según la altura z)
    ctx.save();
    // Efecto de paralaje vertical: desplazar ligeramente hacia arriba según z
    const parallaxY = -p.z * 0.18;
    ctx.translate(p.x, p.y + parallaxY);
    
    const scale = 1 + p.z / 320;
    ctx.scale(scale, scale);
    ctx.rotate(p.angle);

    // Cuerpo principal
    ctx.beginPath();
    ctx.roundRect(-p.w/2, -p.h/2, p.w, p.h, r);
    ctx.fillStyle = p.color;
    ctx.fill();

    // Cara superior más clara (efecto de relieve)
    ctx.beginPath();
    ctx.roundRect(-p.w/2 + 2, -p.h/2 + 2, p.w - 4, p.h * 0.4, 2);
    ctx.fillStyle = hexToRgba(p.color, 0.25);
    ctx.fill();

    // Borde de la pieza
    ctx.beginPath();
    ctx.roundRect(-p.w/2, -p.h/2, p.w, p.h, r);
    ctx.strokeStyle = hexToRgba(p.color, 0.6);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Studs (pivotes)
    drawStuds(ctx, p.w, p.h, p.color);

    ctx.restore();
}

function drawStuds(ctx, w, h, color) {
    const studR = 4;
    const cols  = Math.max(1, Math.round(w / 16));
    const rows  = Math.max(1, Math.round(h / 16));
    const stepX = w / (cols + 1);
    const stepY = h / (rows + 1);

    ctx.fillStyle   = hexToRgba(color, 0.7);
    ctx.strokeStyle = hexToRgba(color, 0.4);
    ctx.lineWidth   = 1;

    for (let r = 1; r <= rows; r++) {
        for (let c = 1; c <= cols; c++) {
            const sx = -w/2 + stepX * c;
            const sy = -h/2 + stepY * r;
            ctx.beginPath();
            ctx.arc(sx, sy, studR, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        }
    }
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function drawBboxOverlay() {
    if (!activeBboxes.length) return;
    canvas.classList.add("has-detections");

    activeBboxes.forEach(bb => {
        // Bbox parpadeante estilo industrial
        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth   = 2;
        ctx.shadowColor = "rgba(0,255,136,0.4)";
        ctx.shadowBlur  = 6;
        ctx.strokeRect(bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1);
        ctx.shadowBlur = 0;

        // Etiqueta
        const label = `${bb.label} ${(bb.conf * 100).toFixed(0)}%`;
        ctx.font = "bold 11px Outfit, sans-serif";
        const tw = ctx.measureText(label).width + 10;
        ctx.fillStyle = "#00ff88";
        ctx.fillRect(bb.x1, bb.y1 - 18, tw, 17);
        ctx.fillStyle = "#000";
        ctx.fillText(label, bb.x1 + 5, bb.y1 - 5);
    });
} 

function animationLoop(timestamp) {
    simulationRafId = requestAnimationFrame(animationLoop);

    // Limpiar y redibujar fondo
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    drawBeltBackground();

    // Calcular velocidad de la cinta en px/frame
    const vy = beltSpeed * 1000 / 60 / 30 * PX_PER_MM;

    if (sessionActive) {
        const totalParts = currentSetParts.reduce((sum, p) => sum + p.qty, 0);
        // MODO BATCH EN SESIÓN ACTIVA: Spawning para alta densidad (10-15 en FOV)
        if (spawnedCount < totalParts) {
            let canSpawn = false;
            let spawnX = 0;
            let spawnY = 0;

            if (beltPieces.length === 0) {
                canSpawn = true;
            } else {
                if (simulationSpawnPool.length === 0) {
                    refillSimulationSpawnPool();
                }
                const nextPiece = simulationSpawnPool[simulationSpawnPool.length - 1];
                
                // Intentar encontrar un lugar válido con min 1cm de separación real
                // Probamos varias posiciones aleatorias cerca del inicio
                for (let k = 0; k < 15; k++) {
                    const margin = 20 + Math.max(nextPiece.w, nextPiece.h) / 2;
                    let cx = margin + Math.random() * (CANVAS_W - 2 * margin);
                    let cy = -Math.max(nextPiece.w, nextPiece.h) - 50 - Math.random() * 40; // Nace arriba
                    
                    let tooClose = false;
                    for (let i = 0; i < beltPieces.length; i++) {
                        let bp = beltPieces[i];
                        let dx = bp.x - cx;
                        let dy = bp.y - cy;
                        let dist = Math.sqrt(dx*dx + dy*dy);
                        // Distancia mínima: buffer de 2mm (permite hasta 20% de solapamiento)
                        let minDist = (Math.max(bp.w, bp.h) / 2) + (Math.max(nextPiece.w, nextPiece.h) / 2) + (2 * PX_PER_MM);
                        if (dist < minDist) {
                            tooClose = true;
                            break;
                        }
                    }
                    if (!tooClose) {
                        spawnX = cx;
                        spawnY = cy;
                        canSpawn = true;
                        break;
                    }
                }
            }

            if (canSpawn) {
                const p = spawnPiece();
                if (spawnY !== 0) {
                    p.x = spawnX;
                    p.y = spawnY;
                } else {
                    p.y = -p.h / 2 - 100;
                }
                beltPieces.push(p);
                spawnedCount++;
            }
        }

        // Actualizar posiciones, detectar salida de piezas de la cinta y no reciclar en lote
        let remainingPieces = [];
        beltPieces.forEach(p => {
            updatePiecePhysics(p, vy);
            
            if (p.y > CANVAS_H) {
                if (p.isDetected && !p.hasBeenCounted) {
                    p.hasBeenCounted = true;
                    sessionIdentifiedCounts[p.ref] = (sessionIdentifiedCounts[p.ref] || 0) + 1;
                    updateSessionInventoryTable();
                    exitedCount++;
                }
            }

            if (p.y <= CANVAS_H + 60) {
                remainingPieces.push(p);
            } else {
                if (!p.hasBeenCounted) {
                    exitedCount++;
                }
            }
        });
        beltPieces = remainingPieces;

        // ── Repulsión física suave ──
        // Permite hasta el 20% de solapamiento entre piezas asentadas.
        // Sólo empuja cuando el solapamiento supera ese umbral.
        for (let i = 0; i < beltPieces.length; i++) {
            const pi = beltPieces[i];
            if (pi.z > 5) continue;
            for (let j = i + 1; j < beltPieces.length; j++) {
                const pj = beltPieces[j];
                if (pj.z > 5) continue;
                const dx = pi.x - pj.x;
                const dy = pi.y - pj.y;
                const dist = Math.hypot(dx, dy) || 0.001;
                // umbral al 80% de la suma de radios ⇒ solapamiento máximo del 20%
                const minD = (Math.max(pi.w, pi.h) + Math.max(pj.w, pj.h)) / 2 * 0.80;
                if (dist < minD) {
                    const force = (minD - dist) * 0.20;
                    const nx = dx / dist;
                    const ny = dy / dist;
                    pi.x += nx * force;
                    pi.y += ny * force;
                    pj.x -= nx * force;
                    pj.y -= ny * force;
                    const mx  = 20 + Math.max(pi.w, pi.h) / 2;
                    const mxj = 20 + Math.max(pj.w, pj.h) / 2;
                    pi.x = Math.max(mx,  Math.min(CANVAS_W - mx,  pi.x));
                    pj.x = Math.max(mxj, Math.min(CANVAS_W - mxj, pj.x));
                }
            }
        }

        // Auto-detener al completar la tanda de todas las piezas
        if (exitedCount >= totalParts && beltPieces.length === 0) {
            const btnToggle = document.getElementById("btn-toggle-session");
            if (btnToggle && btnToggle.innerText.includes("Detener")) {
                btnToggle.click();
            }
        }
    } else {
        // MODO IDLE: Simulación infinita con reciclado (para mantener la UX viva)
        beltPieces.forEach(p => {
            updatePiecePhysics(p, vy);
            if (p.y > CANVAS_H + 60) {
                const sp = spawnPiece();
                // 50% de probabilidad de dejar caer en modo idle
                if (Math.random() < 0.5) {
                    sp.z = 0;
                    sp.vz = 0;
                    sp.spin = (Math.random() - 0.5) * 0.002;
                }
                Object.assign(p, sp);
            }
        });
    }

    beltPieces.forEach(drawPiece);

    // Actualizar la posición de las bboxes activas en vivo para evitar decalaje espacial
    activeBboxes.forEach(bb => {
        bb.y1 += vy;
        bb.y2 += vy;
    });

    // Inferencia YOLO a 500ms (no bloqueante)
    // Se ejecuta ANTES de pintar el overlay para capturar un frame limpio
    if (sessionActive && timestamp - lastInferenceTime > 500) {
        lastInferenceTime = timestamp;
        captureAndDetect();
    }

    // Dibujar bboxes de la última inferencia en pantalla después de capturar
    drawBboxOverlay();
}

function startBeltSimulation() {
    if (simulationRafId) cancelAnimationFrame(simulationRafId);
    initBeltPieces();
    simulationRafId = requestAnimationFrame(animationLoop);
}

function stopBeltSimulation() {
    if (simulationRafId) {
        cancelAnimationFrame(simulationRafId);
        simulationRafId = null;
    }
    activeBboxes = [];
    canvas.classList.remove("has-detections");
    drawBeltBackground();
}

// =============================================================
// 6. Slider de velocidad de cinta
// =============================================================
function initBeltSpeedSlider() {
    const slider  = document.getElementById("belt-speed-slider");
    const display = document.getElementById("speed-value-display");
    if (!slider) return;

    slider.addEventListener("input", () => {
        beltSpeed = parseFloat(slider.value);
        display.innerText = `${beltSpeed.toFixed(1)} m/min`;
        // Actualizar velocidad de todas las piezas en vuelo
        const vy = beltSpeed * 1000 / 60 / 30 * PX_PER_MM;
        beltPieces.forEach(p => p.vy = vy);
    });
}

// =============================================================
// 6b. Slider de Umbral de Confianza en vivo
// =============================================================
function initLiveConfSlider() {
    const slider  = document.getElementById("live-conf-slider");
    const display = document.getElementById("live-conf-value");
    if (!slider) return;
    liveConfThreshold = parseFloat(slider.value);
    display.innerText = `${Math.round(liveConfThreshold * 100)}%`;
    slider.addEventListener("input", () => {
        liveConfThreshold = parseFloat(slider.value);
        display.innerText = `${Math.round(liveConfThreshold * 100)}%`;
    });
}

// =============================================================
// 7. Inferencia YOLO → captura el frame y lo envía al API
// =============================================================
async function captureAndDetect() {
    return new Promise(resolve => {
        canvas.toBlob(async blob => {
            const formData = new FormData();
            formData.append("file", blob, "live_frame.png");

            const start = performance.now();
            try {
                const res  = await fetch(`${API_BASE}/detect?conf=${liveConfThreshold}`, { method: "POST", body: formData });
                const data = await res.json();

                const latency = performance.now() - start;
                document.getElementById("live-latency").innerText      = `${latency.toFixed(0)} ms`;
                document.getElementById("live-frame-pieces").innerText = data.detections_count;
                document.getElementById("metric-fps").innerText        = `${(1000 / latency).toFixed(1)} FPS`;

                // Calcular compensación espacial por latencia del API
                const vy = beltSpeed * 1000 / 60 / 30 * PX_PER_MM;
                const frameLatency = latency / (1000 / 30);
                const spatialOffset = vy * frameLatency;

                // Convertir detecciones YOLO (normalizado) → píxeles absolutos con offset de latencia
                activeBboxes = (data.detections || []).map(d => {
                    const [xc, yc, wn, hn] = d.bbox;
                    return {
                        x1:    (xc - wn/2) * CANVAS_W,
                        y1:    (yc - hn/2) * CANVAS_H + spatialOffset,
                        x2:    (xc + wn/2) * CANVAS_W,
                        y2:    (yc + hn/2) * CANVAS_H + spatialOffset,
                        label: d.name || "lego_piece",
                        conf:  d.confidence || 0,
                    };
                });

                // Centroid matching con las piezas del simulador
                activeBboxes.forEach(bb => {
                    const bb_cx = (bb.x1 + bb.x2) / 2;
                    const bb_cy = (bb.y1 + bb.y2) / 2;
                    
                    let bestPiece = null;
                    let minDist = 999999;
                    beltPieces.forEach(p => {
                        // Evitar asociar piezas que aún están cayendo muy alto en el aire
                        if (p.z > 20) return;

                        const dist = Math.hypot(p.x - bb_cx, p.y - bb_cy);
                        // Priorizar matching si el label de YOLO coincide exactamente con la ref de la pieza simulada
                        const labelMatch = (bb.label === p.ref);
                        const maxDist = labelMatch ? 45 : 25; // 1cm = 19px, así que 25px previene falsas asociaciones

                        if (dist < minDist && dist < maxDist) {
                            minDist = dist;
                            bestPiece = p;
                        }
                    });
                    if (bestPiece) {
                        bestPiece.isDetected = true;
                    }
                });

                document.getElementById("live-bbox-count").innerText = activeBboxes.length;
                canvas.classList.toggle("has-detections", activeBboxes.length > 0);

            } catch (e) {
                // API offline: usar piezas simuladas como bboxes de demo
                activeBboxes = beltPieces.map(p => {
                    const cos = Math.cos(p.angle), sin = Math.sin(p.angle);
                    const hw = p.w/2, hh = p.h/2;
                    const R = Math.sqrt(hw*hw + hh*hh);
                    return {
                        x1: p.x - R, y1: p.y - R,
                        x2: p.x + R, y2: p.y + R,
                        label: p.ref,
                        conf:  0.85 + Math.random() * 0.13,
                    };
                });

                // En modo offline marcamos las piezas de la simulación como detectadas
                beltPieces.forEach(p => {
                    p.isDetected = true;
                });

                canvas.classList.toggle("has-detections", activeBboxes.length > 0);
                document.getElementById("live-bbox-count").innerText = activeBboxes.length;
            }
            resolve();
        }, "image/png");
    });
}

// =============================================================
// 8. Click en canvas → abrir drawer de clasificación
// =============================================================
function onCanvasClick(event) {
    if (!activeBboxes.length) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const mx = (event.clientX - rect.left) * scaleX;
    const my = (event.clientY - rect.top)  * scaleY;

    // Encontrar la primera bbox que contenga el punto del click
    const hit = activeBboxes.find(bb =>
        mx >= bb.x1 && mx <= bb.x2 && my >= bb.y1 && my <= bb.y2
    );

    if (!hit) return;

    // Extraer el crop del frame actual del canvas
    const cw = hit.x2 - hit.x1;
    const ch = hit.y2 - hit.y1;
    if (cw < 4 || ch < 4) return;

    const offscreen = document.createElement("canvas");
    offscreen.width  = cw;
    offscreen.height = ch;
    const octx = offscreen.getContext("2d");
    octx.drawImage(canvas, hit.x1, hit.y1, cw, ch, 0, 0, cw, ch);
    const cropDataUrl = offscreen.toDataURL("image/png");

    // Capturar frame completo en base64 para enviar al API
    const frameDataUrl = canvas.toDataURL("image/png");
    const frameB64 = frameDataUrl.replace(/^data:image\/\w+;base64,/, "");

    openClassifierModal(cropDataUrl, frameB64, hit);
}

// =============================================================
// 9. Drawer lateral de clasificación DINOv2
// =============================================================
function initDrawer() {
    const overlay = document.getElementById("classifier-overlay");
    const btnClose = document.getElementById("btn-close-drawer");

    overlay.addEventListener("click", closeClassifierDrawer);
    btnClose.addEventListener("click", closeClassifierDrawer);
}

function openClassifierDrawer(cropDataUrl, frameB64, bbox) {
    const overlay  = document.getElementById("classifier-overlay");
    const drawer   = document.getElementById("classifier-drawer");
    const loading  = document.getElementById("drawer-loading");
    const result   = document.getElementById("drawer-result");
    const noEmbed  = document.getElementById("drawer-no-embeddings");

    // Mostrar loading
    loading.style.display = "flex";
    result.style.display  = "none";
    noEmbed.style.display = "none";
    overlay.classList.add("visible");
    drawer.classList.add("open");

    // Mostrar crop detectado inmediatamente
    document.getElementById("drawer-crop-img").src = cropDataUrl;

    // Llamar al endpoint /classify_crop
    classifyWithDINOv2(frameB64, bbox)
        .then(data => {
            loading.style.display = "none";

            if (data.status === "not_ready") {
                noEmbed.style.display = "flex";
                return;
            }

            if (!data.best_match) {
                result.style.display = "flex";
                document.getElementById("drawer-part-name").innerText = "Sin coincidencia";
                document.getElementById("drawer-part-ref").innerText  = "—";
                document.getElementById("drawer-confidence").innerText = "0%";
                document.getElementById("drawer-conf-bar").style.width = "0%";
                document.getElementById("candidate-list").innerHTML = "";
                document.getElementById("drawer-ref-wrap").innerHTML = `<div class="drawer-img-placeholder">Sin imagen</div>`;
                return;
            }

            const best = data.best_match;
            result.style.display = "flex";

            // Nombre y referencia
            document.getElementById("drawer-part-name").innerText = best.part_name;
            document.getElementById("drawer-part-ref").innerText  = `LDraw: ${best.part_ref}`;

            // Imagen de referencia (BrickLink u otra, sin render local)
            const refWrap = document.getElementById("drawer-ref-wrap");
            refWrap.innerHTML = `<div class="drawer-img-placeholder">Sin render 3D</div>`;

            // Barra de confianza (animada con delay para efecto visual)
            const pct = Math.round(best.score * 100);
            document.getElementById("drawer-confidence").innerText = `${pct}%`;
            requestAnimationFrame(() => {
                document.getElementById("drawer-conf-bar").style.width = `${pct}%`;
            });

            // Top-3 candidatos
            const candidateList = document.getElementById("candidate-list");
            candidateList.innerHTML = (data.top3 || []).map(c => {
                const scorePct = Math.round(c.score * 100);
                const thumbHtml = `<div class="candidate-thumb" style="display:flex;align-items:center;justify-content:center;color:rgba(255,255,255,0.2);font-size:1.2rem;">🧱</div>`;
                return `
                    <div class="candidate-card ${c.rank === 1 ? 'rank-1' : ''}">
                        <div class="candidate-rank">${c.rank}</div>
                        ${thumbHtml}
                        <div class="candidate-info">
                            <div class="candidate-ref">${c.part_name}</div>
                            <div class="candidate-score-text">Ref: ${c.part_ref} — Similitud: ${scorePct}%</div>
                        </div>
                        <div class="candidate-score-bar">
                            <div class="candidate-score-bar-fill" style="width:${scorePct}%"></div>
                        </div>
                    </div>`;
            }).join("");
        })
        .catch(err => {
            console.error("Error en clasificación:", err);
            loading.style.display = "none";
            noEmbed.style.display = "flex";
        });
}

function closeClassifierDrawer() {
    document.getElementById("classifier-overlay").classList.remove("visible");
    document.getElementById("classifier-drawer").classList.remove("open");
}

async function classifyWithDINOv2(frameB64, bbox, filename = null) {
    const res = await fetch(`${API_BASE}/classify_crop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            bbox:      [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            frame_b64: frameB64,
            filename:  filename
        }),
    });
    return res.json();
}

// =============================================================
// 10. Control de Sesión de Inferencia
// =============================================================
function initSessionControls() {
    const btnToggle  = document.getElementById("btn-toggle-session");
    const btnCapture = document.getElementById("btn-capture-frame");
    const badge      = document.getElementById("session-badge");
    const liveSessId = document.getElementById("live-session-id");

    btnToggle.addEventListener("click", async () => {
        if (!sessionActive) {
            // Reset historial y conteo al iniciar una nueva corrida
            resetSessionIdentifiedCounts();
            spawnedCount = 0;
            exitedCount = 0;
            beltPieces = []; // vaciar cinta para iniciar lote desde arriba
            historyImages = [];
            currentHistoryIndex = 0;
            renderHistoryThumbnails();
            clearHistoryCanvas();
            document.getElementById("history-page-num").innerText = "Sin fotos capturadas";

            try {
                const res = await fetch(`${API_BASE}/session/start`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model_version:   "yolov8n_synthetic",
                        belt_speed_mm_s: beltSpeed * 1000 / 60,
                    }),
                });
                const data = await res.json();

                if (data.status === "success") {
                    sessionActive = true;
                    sessionId     = data.session_id;

                    btnToggle.innerText       = "Detener Inferencia";
                    btnToggle.style.background = "var(--accent-red)";
                    badge.innerText            = "Sesión Activa";
                    badge.className            = "piece-conf high";
                    liveSessId.innerText       = sessionId;

                    startBeltSimulation();
                    startHistoryPolling();
                }
            } catch (e) {
                // API offline: arrancar simulación de demo igualmente
                console.warn("API offline, arrancando simulación demo...");
                sessionActive = true;
                btnToggle.innerText        = "Detener Simulación";
                btnToggle.style.background = "var(--accent-red)";
                badge.innerText            = "Demo (sin API)";
                badge.className            = "piece-conf high";
                startBeltSimulation();
                startHistoryPolling();
            }
        } else {
            // Detener
            try {
                await fetch(`${API_BASE}/session/stop`, { method: "POST" });
            } catch (e) { /* ignorar si la API no está online */ }

            sessionActive = false;
            sessionId     = null;

            btnToggle.innerText        = "Iniciar Sesión de Inferencia";
            btnToggle.style.background = "var(--accent)";
            badge.innerText            = "Sin Sesión Activa";
            badge.className            = "piece-conf low";
            liveSessId.innerText       = "Ninguna";

            stopBeltSimulation();
            initBeltPieces(); // repoblar cinta con piezas idle para mantener viva la UX
            
            // Hacer un último poll para recolectar las imágenes finales y detener polling
            setTimeout(async () => {
                await pollHistoryImages();
                stopHistoryPolling();
            }, 500);

            updateStats();
        }
    });

    btnCapture.addEventListener("click", () => captureAndDetect());
}

// =============================================================
// 11. Búsqueda de Sets Reales
// =============================================================
function initSetSearch() {
    const btnSearch   = document.getElementById("btn-search-set");
    const inputSearch = document.getElementById("set-search-input");
    if (!btnSearch || !inputSearch) return;

    const triggerSearch = async () => {
        const set_id = inputSearch.value.trim();
        if (!set_id) return;

        const loader    = document.getElementById("set-loading");
        const container = document.getElementById("set-result-container");
        loader.style.display    = "flex";
        container.style.display = "none";

        try {
            if (typeof pywebview === "undefined" || !pywebview.api) {
                setTimeout(() => { loader.style.display = "none"; alert("Abre la app con python gui/app.py"); }, 1000);
                return;
            }
            const res = await pywebview.api.get_set_inventory(set_id);
            loader.style.display = "none";

            if (res.status === "success") {
                container.style.display = "flex";
                document.getElementById("set-title").innerText = res.set_name;

                const partsGrid = document.getElementById("set-parts-grid");
                partsGrid.innerHTML = res.parts.map(p => {
                    const imgHtml = p.image
                        ? `<img src="${p.image}" class="part-img" alt="${p.ref}">`
                        : `<div style="color:var(--text-secondary);font-size:0.8rem;height:100px;display:flex;align-items:center;justify-content:center;">Sin vista 3D</div>`;
                    return `
                        <div class="part-card">
                            <div class="part-img-container">${imgHtml}</div>
                            <div class="part-info-grid">
                                <span class="part-title">Pieza ${p.ref}</span>
                                <span class="part-ref">${p.color_name}</span>
                                <div class="part-meta">
                                    <span style="color:var(--text-secondary);">Cantidad:</span>
                                    <span class="part-qty-badge">${p.qty}</span>
                                </div>
                            </div>
                        </div>`;
                }).join("");

                const minifigsList = document.getElementById("set-minifigures-list");
                if (res.minifigures && res.minifigures.length > 0) {
                    minifigsList.innerHTML = res.minifigures.map(m => `
                        <div class="recent-item">
                            <div class="piece-info">
                                <span class="piece-class">Fig. ${m.ref}</span>
                                <span class="piece-name">${m.name}</span>
                            </div>
                            <span class="piece-conf high">${m.qty} uds</span>
                        </div>`).join("");
                } else {
                    minifigsList.innerHTML = `<div class="recent-item">Sin minifiguras en este set</div>`;
                }
            } else {
                alert("Error: " + res.message);
            }
        } catch (e) {
            loader.style.display = "none";
            console.error("Error set search:", e);
        }
    };

    btnSearch.addEventListener("click", triggerSearch);
    inputSearch.addEventListener("keypress", e => { if (e.key === "Enter") triggerSearch(); });
}

// =============================================================
// 12. Panel de Entrenamiento
// =============================================================

let trainingPollInterval = null;

function initTrainingPanel() {
    document.getElementById("btn-start-training").addEventListener("click", async () => {
        const epochs  = parseInt(document.getElementById("train-epochs").value)  || 15;
        const batch   = parseInt(document.getElementById("train-batch").value)   || 16;
        const dsize   = parseInt(document.getElementById("train-dataset-size").value) || 200;

        setStepStatus("step-1", "running", "▶ Ejecutando...");
        appendLog(`\n> Iniciando entrenamiento: ${epochs} épocas, batch=${batch}, ${dsize} imágenes sintéticas...`);

        try {
            if (typeof pywebview !== "undefined" && pywebview.api) {
                const res = await pywebview.api.start_training(epochs, dsize, batch);
                appendLog(`> ${res.message}`);
            } else {
                appendLog("> [DEMO] API de PyWebView no disponible en modo navegador.");
                setStepStatus("step-1", "error", "Sin API");
                return;
            }

            // Iniciar polling de progreso
            if (trainingPollInterval) clearInterval(trainingPollInterval);
            trainingPollInterval = setInterval(pollTrainingStatus, 1000);
        } catch (e) {
            appendLog(`> ERROR: ${e.message}`);
            setStepStatus("step-1", "error", "Error");
        }
    });

    document.getElementById("btn-start-indexing").addEventListener("click", async () => {
        setStepStatus("step-2", "running", "▶ Indexando...");
        appendLog("\n> Iniciando indexación DINOv2 (multi-view renders + embeddings)...");

        try {
            if (typeof pywebview !== "undefined" && pywebview.api) {
                const res = await pywebview.api.start_indexing();
                appendLog(`> ${res.message}`);
                setTimeout(refreshEmbeddingCount, 5000);
            } else {
                appendLog("> [DEMO] PyWebView no disponible.");
                setStepStatus("step-2", "error", "Sin API");
            }
        } catch (e) {
            appendLog(`> ERROR: ${e.message}`);
            setStepStatus("step-2", "error", "Error");
        }
    });

    document.getElementById("btn-refresh-embeddings").addEventListener("click", refreshEmbeddingCount);
    document.getElementById("btn-clear-logs").addEventListener("click", () => {
        document.getElementById("training-logs").textContent = "> LegoVision Training Terminal\n> Logs limpios.\n";
    });

    // Intentar cargar estado inicial
    window.addEventListener("pywebviewready", () => {
        pollTrainingStatus();
        refreshEmbeddingCount();
    });
}

async function pollTrainingStatus() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;

    try {
        const res = await pywebview.api.get_training_status();
        if (res.status !== "ok" || !res.run) return;

        const run = res.run;

        // Badge del header
        const badge = document.getElementById("train-status-badge");
        badge.innerText = run.status === "running" ? "Entrenando..." :
                          run.status === "completed" ? "Completado ✓" :
                          run.status === "failed"    ? "Error ✗" : run.status;
        badge.className = run.status === "running"   ? "piece-conf high" :
                          run.status === "completed" ? "piece-conf high" :
                          run.status === "failed"    ? "piece-conf low"  : "piece-conf low";

        // Paso 1 status
        if (run.status === "running") {
            setStepStatus("step-1", "running", `Época ${run.current_epoch}/${run.epochs}`);
            if (!trainingPollInterval)
                trainingPollInterval = setInterval(pollTrainingStatus, 1000);
        } else if (run.status === "completed") {
            setStepStatus("step-1", "done", "Completado ✓");
            if (trainingPollInterval) { clearInterval(trainingPollInterval); trainingPollInterval = null; }
        } else if (run.status === "failed") {
            setStepStatus("step-1", "error", "Error ✗");
            if (trainingPollInterval) { clearInterval(trainingPollInterval); trainingPollInterval = null; }
        }

        // Barra de progreso de épocas
        const pct = run.epochs > 0 ? (run.current_epoch / run.epochs) * 100 : 0;
        document.getElementById("epoch-progress-bar").style.width = `${pct}%`;
        document.getElementById("epoch-label").innerText =
            `Época ${run.current_epoch} / ${run.epochs}`;

        const lossStr  = run.loss  != null ? run.loss.toFixed(4)  : "—";
        const map50Str = run.map50 != null ? run.map50.toFixed(4) : "—";
        document.getElementById("metrics-label").innerText =
            `Loss: ${lossStr} | mAP50: ${map50Str}`;

        // Añadir punto al historial del chart
        if (run.current_epoch > 0 && (chartHistory.epochs.length === 0 ||
            chartHistory.epochs[chartHistory.epochs.length-1] !== run.current_epoch)) {
            chartHistory.epochs.push(run.current_epoch);
            chartHistory.loss.push(run.loss || 0);
            chartHistory.map50.push(run.map50 || 0);
            drawTrainingChart();
        }

        // Logs: mostrar las últimas líneas
        if (run.logs) {
            const terminal = document.getElementById("training-logs");
            terminal.textContent = run.logs.slice(-3000); // últimos 3000 chars
            terminal.scrollTop   = terminal.scrollHeight;
        }

    } catch (e) {
        console.error("Error polling training status:", e);
    }
}

async function refreshEmbeddingCount() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const res = await pywebview.api.get_embedding_count();
        const count = res.count || 0;
        document.getElementById("step-3-status").innerText =
            count > 0 ? `${count} embeddings` : "0 embeddings";
        setStepStatus("step-2", count > 0 ? "done" : "pending",
            count > 0 ? "Indexado ✓" : "Pendiente");
    } catch (e) { console.error("Embedding count error:", e); }
}

// =============================================================
// 13. Gráfico de entrenamiento (canvas HTML5)
// =============================================================
function drawTrainingChart() {
    const chartCanvas = document.getElementById("training-chart");
    if (!chartCanvas) return;

    // Fijar dimensiones en píxeles según el CSS
    chartCanvas.width  = chartCanvas.offsetWidth  || 640;
    chartCanvas.height = chartCanvas.offsetHeight || 160;

    const cctx = chartCanvas.getContext("2d");
    const W = chartCanvas.width, H = chartCanvas.height;
    const PAD = { top: 16, right: 16, bottom: 28, left: 44 };
    const cw = W - PAD.left - PAD.right;
    const ch = H - PAD.top  - PAD.bottom;

    cctx.clearRect(0, 0, W, H);

    if (chartHistory.epochs.length < 2) return;

    const n      = chartHistory.epochs.length;
    const maxLoss = Math.max(...chartHistory.loss, 0.01);

    // Grid
    cctx.strokeStyle = "rgba(255,255,255,0.05)";
    cctx.lineWidth   = 1;
    for (let i = 0; i <= 4; i++) {
        const y = PAD.top + ch * (1 - i / 4);
        cctx.beginPath(); cctx.moveTo(PAD.left, y); cctx.lineTo(PAD.left + cw, y); cctx.stroke();
        cctx.fillStyle  = "rgba(255,255,255,0.3)";
        cctx.font       = "10px Outfit,sans-serif";
        cctx.textAlign  = "right";
        cctx.fillText((maxLoss * i / 4).toFixed(2), PAD.left - 6, y + 4);
    }

    // Helper: trazar línea
    function plotLine(data, color, maxVal) {
        cctx.beginPath();
        data.forEach((v, i) => {
            const x = PAD.left + (i / (n - 1)) * cw;
            const y = PAD.top + ch * (1 - v / maxVal);
            i === 0 ? cctx.moveTo(x, y) : cctx.lineTo(x, y);
        });
        cctx.strokeStyle = color;
        cctx.lineWidth   = 2;
        cctx.stroke();

        // Puntos
        data.forEach((v, i) => {
            const x = PAD.left + (i / (n - 1)) * cw;
            const y = PAD.top + ch * (1 - v / maxVal);
            cctx.beginPath();
            cctx.arc(x, y, 3, 0, Math.PI * 2);
            cctx.fillStyle = color;
            cctx.fill();
        });
    }

    plotLine(chartHistory.loss,  "#3b82f6", maxLoss);
    plotLine(chartHistory.map50, "#10b981", 1.0);

    // Leyenda
    cctx.font      = "10px Outfit,sans-serif";
    cctx.textAlign = "left";
    cctx.fillStyle = "#3b82f6";
    cctx.fillText("● Loss train", PAD.left, H - 8);
    cctx.fillStyle = "#10b981";
    cctx.fillText("● mAP50", PAD.left + 80, H - 8);
}

// =============================================================
// Helpers
// =============================================================
function appendLog(msg) {
    const terminal = document.getElementById("training-logs");
    if (!terminal) return;
    terminal.textContent += msg + "\n";
    terminal.scrollTop    = terminal.scrollHeight;
}

function setStepStatus(stepId, state, label) {
    const statusEl = document.getElementById(`${stepId}-status`);
    if (!statusEl) return;
    statusEl.className = `step-status ${state}`;
    statusEl.innerText = label;
}


// =============================================================
// 14. HISTORIAL DE IMÁGENES Y MODAL DETALLE (VISTA EN VIVO)
// =============================================================
let historyImages = [];
let currentHistoryIndex = 0;
let historyBboxes = [];
let historyCanvas, historyCtx;
let historyPollingTimer = null;

function initHistoryView() {
    historyCanvas = document.getElementById("history-canvas");
    if (!historyCanvas) return;
    historyCtx = historyCanvas.getContext("2d");
    historyCanvas.width = CANVAS_W;
    historyCanvas.height = CANVAS_H;

    clearHistoryCanvas();

    document.getElementById("btn-history-prev").addEventListener("click", () => navigateHistoryImage(-1));
    document.getElementById("btn-history-next").addEventListener("click", () => navigateHistoryImage(1));
    historyCanvas.addEventListener("click", onHistoryCanvasClick);

    // Configurar modal pop-up
    const modal = document.getElementById("classifier-modal");
    const overlay = document.getElementById("classifier-overlay");
    const btnCloseModal = document.getElementById("btn-close-modal");

    const closeModal = () => {
        modal.classList.remove("visible");
        overlay.classList.remove("visible");
    };

    if (btnCloseModal) btnCloseModal.addEventListener("click", closeModal);
    if (overlay) overlay.addEventListener("click", closeModal);

    // Iniciar polling para reflejar el estado actual
    startHistoryPolling();
}

function clearHistoryCanvas() {
    if (!historyCtx) return;
    const grad = historyCtx.createLinearGradient(0, 0, 0, CANVAS_H);
    grad.addColorStop(0,   "#0e1220");
    grad.addColorStop(0.5, "#111827");
    grad.addColorStop(1,   "#0e1220");
    historyCtx.fillStyle = grad;
    historyCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Textura de cinta
    historyCtx.strokeStyle = "rgba(255,255,255,0.03)";
    historyCtx.lineWidth   = 2;
    for (let x = 20; x < CANVAS_W; x += 60) {
        historyCtx.beginPath();
        historyCtx.moveTo(x, 0); historyCtx.lineTo(x, CANVAS_H);
        historyCtx.stroke();
    }
}

function startHistoryPolling() {
    if (historyPollingTimer) clearInterval(historyPollingTimer);
    historyPollingTimer = setInterval(pollHistoryImages, 1500);
}

function stopHistoryPolling() {
    if (historyPollingTimer) {
        clearInterval(historyPollingTimer);
        historyPollingTimer = null;
    }
}

async function pollHistoryImages() {
    try {
        const res = await fetch(`${API_BASE}/inference-run/images`);
        const data = await res.json();
        const newImages = data.images || [];

        if (JSON.stringify(newImages) !== JSON.stringify(historyImages)) {
            const oldLength = historyImages.length;
            historyImages = newImages;
            renderHistoryThumbnails();
            
            if (oldLength === 0 && historyImages.length > 0) {
                loadHistoryImage(historyImages.length - 1);
            } else if (historyImages.length > 0) {
                updateHistoryNavigationControls();
            }
        }
    } catch (e) {
        console.error("Error polling history images:", e);
    }
}

function renderHistoryThumbnails() {
    const container = document.getElementById("history-thumbnails");
    if (!container) return;

    if (!historyImages.length) {
        container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-secondary); font-size: 0.85rem; padding: 20px;">Sin frames capturados</div>`;
        return;
    }

    container.innerHTML = historyImages.map((filename, index) => {
        const isActive = index === currentHistoryIndex ? 'active' : '';
        const thumbUrl = `${API_BASE}/inference-run/image/${filename}`;
        return `
            <div class="history-thumb-item ${isActive}" onclick="loadHistoryImage(${index})" style="position: relative; cursor: pointer; aspect-ratio: 4/3; border-radius: 6px; border: 2px solid ${index === currentHistoryIndex ? 'var(--accent)' : 'var(--border)'}; overflow: hidden; background: #000; transition: var(--transition);">
                <img src="${thumbUrl}" style="width: 100%; height: 100%; object-fit: cover;" alt="Frame ${index + 1}">
                <div style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.6); color: #fff; font-size: 10px; text-align: center; padding: 2px 0;">
                    #${index + 1}
                </div>
            </div>
        `;
    }).join('');
}

async function loadHistoryImage(index) {
    if (!historyImages.length || index < 0 || index >= historyImages.length) return;
    currentHistoryIndex = index;

    // Actualizar borde de miniaturas
    const items = document.querySelectorAll(".history-thumb-item");
    items.forEach((item, idx) => {
        if (idx === index) {
            item.style.borderColor = "var(--accent)";
            item.classList.add("active");
        } else {
            item.style.borderColor = "var(--border)";
            item.classList.remove("active");
        }
    });

    const filename = historyImages[index];
    const pageNumEl = document.getElementById("history-page-num");
    if (pageNumEl) {
        pageNumEl.innerText = `Imagen ${index + 1} de ${historyImages.length}`;
    }

    updateHistoryNavigationControls();
    clearHistoryCanvas();

    historyCtx.fillStyle = "rgba(255,255,255,0.8)";
    historyCtx.font = "16px Outfit, sans-serif";
    historyCtx.textAlign = "center";
    historyCtx.fillText("Cargando imagen e inferencia YOLO...", CANVAS_W / 2, CANVAS_H / 2);

    try {
        const detectRes = await fetch(`${API_BASE}/inference-run/detect/${filename}`, { method: "POST" });
        const detectData = await detectRes.json();

        const img = new Image();
        img.src = `${API_BASE}/inference-run/image/${filename}?t=${new Date().getTime()}`;
        img.onload = () => {
            historyCtx.drawImage(img, 0, 0, CANVAS_W, CANVAS_H);

            historyBboxes = (detectData.detections || []).map(d => {
                const [xc, yc, wn, hn] = d.bbox;
                const x1 = (xc - wn/2) * CANVAS_W;
                const y1 = (yc - hn/2) * CANVAS_H;
                const x2 = (xc + wn/2) * CANVAS_W;
                const y2 = (yc + hn/2) * CANVAS_H;
                const cw = x2 - x1;
                const ch = y2 - y1;

                // Dibujar el BBox visible
                historyCtx.strokeStyle = "var(--accent, #3b82f6)";
                historyCtx.lineWidth = 2;
                historyCtx.strokeRect(x1, y1, cw, ch);
                
                // Fondo semi-transparente al hacer hover (opcional, por ahora solo dibujamos stroke)
                historyCtx.fillStyle = "rgba(59, 130, 246, 0.15)";
                historyCtx.fillRect(x1, y1, cw, ch);

                return {
                    x1:    x1,
                    y1:    y1,
                    x2:    x2,
                    y2:    y2,
                    label: d.name || "lego_piece",
                    conf:  d.confidence || 0,
                };
            });

            // Bboxes ya están dibujadas en la imagen persistida en el backend
            // Solo necesitamos popular historyBboxes para los clicks
        };
    } catch (e) {
        console.error("Error al cargar imagen del historial:", e);
        clearHistoryCanvas();
        historyCtx.fillStyle = "#ef4444";
        historyCtx.fillText("Error al procesar el frame del historial.", CANVAS_W / 2, CANVAS_H / 2);
    }
}

function updateHistoryNavigationControls() {
    const btnPrev = document.getElementById("btn-history-prev");
    const btnNext = document.getElementById("btn-history-next");
    if (btnPrev) btnPrev.disabled = (currentHistoryIndex === 0 || historyImages.length === 0);
    if (btnNext) btnNext.disabled = (currentHistoryIndex === historyImages.length - 1 || historyImages.length === 0);
}

function navigateHistoryImage(direction) {
    const nextIndex = currentHistoryIndex + direction;
    if (nextIndex >= 0 && nextIndex < historyImages.length) {
        loadHistoryImage(nextIndex);
    }
}

function onHistoryCanvasClick(event) {
    if (!historyBboxes.length) return;

    const rect = historyCanvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const mx = (event.clientX - rect.left) * scaleX;
    const my = (event.clientY - rect.top)  * scaleY;

    const hit = historyBboxes.find(bb =>
        mx >= bb.x1 && mx <= bb.x2 && my >= bb.y1 && my <= bb.y2
    );

    if (!hit) return;

    const cw = hit.x2 - hit.x1;
    const ch = hit.y2 - hit.y1;
    if (cw < 4 || ch < 4) return;

    const offscreen = document.createElement("canvas");
    offscreen.width  = cw;
    offscreen.height = ch;
    const octx = offscreen.getContext("2d");
    octx.drawImage(historyCanvas, hit.x1, hit.y1, cw, ch, 0, 0, cw, ch);
    const cropDataUrl = offscreen.toDataURL("image/png");

    const frameDataUrl = historyCanvas.toDataURL("image/png");
    const frameB64 = frameDataUrl.replace(/^data:image\/\w+;base64,/, "");
    const filename = historyImages[currentHistoryIndex] || null;

    openClassifierModal(cropDataUrl, frameB64, hit, filename);
}

function openClassifierModal(cropDataUrl, frameB64, bbox, filename = null) {
    const modal = document.getElementById("classifier-modal");
    const overlay = document.getElementById("classifier-overlay");
    const loading = document.getElementById("modal-loading");
    const result = document.getElementById("modal-result");
    const noEmbed = document.getElementById("modal-no-embeddings");

    if (!modal) return;

    loading.style.display = "flex";
    result.style.display  = "none";
    noEmbed.style.display = "none";
    if (overlay) overlay.classList.add("visible");
    modal.classList.add("visible");

    document.getElementById("modal-crop-img").src = cropDataUrl;

    classifyWithDINOv2(frameB64, bbox, filename)
        .then(data => {
            loading.style.display = "none";

            if (data.status === "not_ready") {
                noEmbed.style.display = "flex";
                return;
            }

            if (!data.best_match) {
                result.style.display = "flex";
                document.getElementById("modal-part-name").innerText = "Sin coincidencia";
                document.getElementById("modal-part-ref").innerText  = "—";
                document.getElementById("modal-confidence").innerText = "0%";
                document.getElementById("modal-conf-bar").style.width = "0%";
                document.getElementById("modal-bricklink-wrap").innerHTML = `<div class="drawer-img-placeholder">Sin imagen</div>`;
                document.getElementById("modal-color-badge").style.background = "#475569";
                document.getElementById("modal-color-text").innerText = "—";
                document.getElementById("modal-bricklink-url").href = "#";
                return;
            }

            const best = data.best_match;
            result.style.display = "flex";

            // ── Nombre y referencia ──
            document.getElementById("modal-part-name").innerText = best.part_name || `Pieza LDraw ${best.part_ref}`;
            document.getElementById("modal-part-ref").innerText  = `Referencia LDraw: ${best.part_ref}`;

            // ── Color badge ──
            const colorHex  = best.color_hex  || "#A0A5A9";
            const colorName = best.color_name || "Unknown";
            const colorCode = best.detected_color || "85";
            document.getElementById("modal-color-badge").style.background = colorHex;
            document.getElementById("modal-color-text").innerText = `${colorCode} — ${colorName}`;

            // ── Imagen BrickLink ──
            const blColorId = LDRAW_TO_BRICKLINK_COLOR[String(colorCode)] || "86";
            const partRef   = best.part_ref;
            const isMinifig = /^(sw|fig|cty|cas|pi|tor|wr|hp|arc|jw|ind|idea|tlm|col|dim|sh|njo|tnt|gam|utl|bat|sp|rac|dk|fst|trn|pln|cre|elf|gal|flm|lco|min|lia|alp|aqu|hol|hrf|crf|col|rck|knf|elf|cmd|air|for|sea|tur|fra|mck|dst|ntd|hnt)/.test(partRef);
            let blImgUrl, blPageUrl;
            if (isMinifig) {
                blImgUrl  = `https://img.bricklink.com/ItemImage/MN/0/${partRef}.png`;
                blPageUrl = `https://www.bricklink.com/v2/catalog/catalogitem.page?M=${partRef}`;
            } else {
                blImgUrl  = `https://img.bricklink.com/ItemImage/PN/${blColorId}/${partRef}.png`;
                blPageUrl = `https://www.bricklink.com/v2/catalog/catalogitem.page?P=${partRef}`;
            }

            // Intentar cargar imagen de BrickLink (puede fallar por CORS o parte inexistente)
            const blWrap = document.getElementById("modal-bricklink-wrap");
            const blImg  = new Image();
            blImg.onload = () => {
                blImg.style.cssText = "max-width:100%;max-height:110px;object-fit:contain;border-radius:6px;";
                blImg.alt = `BrickLink ${partRef}`;
                blWrap.innerHTML = "";
                blWrap.appendChild(blImg);
            };
            blImg.onerror = () => {
                blWrap.innerHTML = `<div class="drawer-img-placeholder" style="font-size:0.75rem;text-align:center;">Imagen BrickLink<br>no disponible</div>`;
            };
            blImg.src = blImgUrl;

            // Enlace al catálogo BrickLink
            document.getElementById("modal-bricklink-url").href = blPageUrl;

            // ── Barra de confianza ──
            const pct = Math.round(best.score * 100);
            document.getElementById("modal-confidence").innerText = `${pct}%`;
            requestAnimationFrame(() => {
                document.getElementById("modal-conf-bar").style.width = `${pct}%`;
            });
        })
        .catch(err => {
            console.error("Error classifying in modal:", err);
            loading.style.display = "none";
        });
}

// =============================================================
// 15. TARGET SET SELECTOR & ESTIMATION HELPERS
// =============================================================

function estimatePartDimensions(ref, name) {
    const n = (name || "").toLowerCase();
    
    // Check if it's a minifigure
    if (ref.startsWith("sw") || ref.startsWith("fig") || n.includes("minifig")) {
        return { w: 32, h: 42 };
    }
    
    // Check for standard Brick sizes in name (e.g. "Brick 2 x 4", "Plate 1 x 2")
    const match = n.match(/(?:brick|plate|tile|slope|wedge)\s+(\d+)\s*x\s*(\d+)/i);
    if (match) {
        const x = parseInt(match[1]);
        const y = parseInt(match[2]);
        // Scale factor: standard unit stud is ~14px wide
        return { w: Math.max(14, x * 14), h: Math.max(14, y * 14) };
    }
    
    // Fallbacks based on common LDraw reference names if description is simple
    if (ref === "3001") return { w: 56, h: 28 }; // Brick 2x4
    if (ref === "3002") return { w: 42, h: 28 }; // Brick 2x3
    if (ref === "3003") return { w: 28, h: 28 }; // Brick 2x2
    if (ref === "3004") return { w: 28, h: 14 }; // Brick 1x2
    if (ref === "3005") return { w: 14, h: 14 }; // Brick 1x1
    if (ref === "3010") return { w: 56, h: 14 }; // Brick 1x4
    if (ref === "3020") return { w: 56, h: 28 }; // Plate 2x4
    if (ref === "3021") return { w: 42, h: 14 }; // Plate 2x3
    if (ref === "3022") return { w: 28, h: 28 }; // Plate 2x2
    if (ref === "3023") return { w: 28, h: 14 }; // Plate 1x2
    if (ref === "3024") return { w: 14, h: 14 }; // Plate 1x1
    if (ref === "3070" || ref === "3070b") return { w: 14, h: 14 }; // Tile 1x1
    if (ref === "3069" || ref === "3069b") return { w: 28, h: 14 }; // Tile 1x2
    if (ref === "3068" || ref === "3068b") return { w: 28, h: 28 }; // Tile 2x2
    if (ref === "2420") return { w: 28, h: 28 }; // Plate 2x2 Corner
    if (ref === "3710") return { w: 42, h: 14 }; // Plate 1x3
    if (ref === "3666") return { w: 84, h: 14 }; // Plate 1x6
    if (ref === "3795") return { w: 84, h: 28 }; // Plate 2x6
    if (ref === "4073") return { w: 14, h: 14 }; // Plate Round 1x1
    if (ref === "6141") return { w: 14, h: 14 }; // Plate Round 1x1
    if (ref === "15573") return { w: 28, h: 14 }; // Plate Modified Jumper 1x2
    if (ref === "18674") return { w: 28, h: 28 }; // Plate Round 2x2
    if (ref === "32000") return { w: 28, h: 14 }; // Technic Brick 1x2
    if (ref === "2780") return { w: 20, h: 8 };   // Technic Pin
    if (ref === "3673") return { w: 20, h: 8 };   // Technic Pin
    if (ref === "4274") return { w: 14, h: 8 };   // Technic Pin
    
    // Default fallback
    return { w: 24, h: 24 };
}

async function changeTargetSet(setId) {
    if (typeof pywebview !== "undefined" && pywebview.api) {
        try {
            const res = await pywebview.api.get_set_inventory_light(setId);
            if (res.status === "success") {
                currentSetId = setId;
                currentSetName = res.set_name;
                
                // Map minifigures to standard part structure
                const minifigsMapped = (res.minifigures || []).map(m => {
                    const dims = estimatePartDimensions(m.ref, m.name);
                    return {
                        ref: m.ref,
                        name: m.name,
                        w: dims.w,
                        h: dims.h,
                        color: "#FFFFFF",
                        colorName: "Blanco",
                        qty: m.qty
                    };
                });
                
                // Map parts
                const partsMapped = (res.parts || []).map(p => {
                    const dims = estimatePartDimensions(p.ref, p.desc || p.name || "");
                    return {
                        ref: p.ref,
                        name: p.desc || p.name || `Pieza LDraw ${p.ref}`,
                        w: dims.w,
                        h: dims.h,
                        color: p.color_hex || "#808080",
                        colorName: p.color_name || "Color",
                        qty: p.qty
                    };
                });
                
                currentSetParts = [...minifigsMapped, ...partsMapped];
                
                // Update dynamic set title in index.html
                const titleSpan = document.getElementById("inference-set-title");
                if (titleSpan) {
                    const totalPieces = currentSetParts.reduce((sum, p) => sum + p.qty, 0);
                    titleSpan.innerText = `${currentSetName} (${totalPieces} piezas)`;
                }
                
                // Reset simulation variables
                resetSessionIdentifiedCounts();
                refillSimulationSpawnPool();
                if (sessionActive) {
                    beltPieces = [];
                    spawnedCount = 0;
                    exitedCount = 0;
                } else {
                    initBeltPieces();
                }
                console.log(`Set switched to ${setId} with ${currentSetParts.length} parts.`);
                return;
            }
        } catch (e) {
            console.error("Error loading set inventory light:", e);
        }
    }
    
    // Fallback if pywebview is not ready/browser mode
    console.warn("Using fallback local inventory for set ID:", setId);
    const localData = LOCAL_SET_CATALOG[setId];
    if (localData) {
        currentSetId = setId;
        currentSetName = localData.name;
        currentSetParts = [...localData.parts];
        
        const titleSpan = document.getElementById("inference-set-title");
        if (titleSpan) {
            const totalPieces = currentSetParts.reduce((sum, p) => sum + p.qty, 0);
            titleSpan.innerText = `${currentSetName} (${totalPieces} piezas)`;
        }
        
        resetSessionIdentifiedCounts();
        refillSimulationSpawnPool();
        if (sessionActive) {
            beltPieces = [];
            spawnedCount = 0;
            exitedCount = 0;
        } else {
            initBeltPieces();
        }
    }
}

function initTargetSetSelector() {
    const selectSet = document.getElementById("select-target-set");
    if (!selectSet) return;
    
    selectSet.addEventListener("change", async () => {
        const setId = selectSet.value;
        await changeTargetSet(setId);
    });
    
    // Trigger initial load when bridge is ready
    window.addEventListener("pywebviewready", async () => {
        await changeTargetSet(selectSet.value);
    });
    
    // Fallback immediate call for pure static HTML representation
    setTimeout(async () => {
        if (currentSetParts.length === 0 || currentSetId !== selectSet.value) {
            await changeTargetSet(selectSet.value);
        }
    }, 100);
}


