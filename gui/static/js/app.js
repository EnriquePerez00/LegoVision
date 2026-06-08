// =============================================================
// LegoVision — Dashboard Logic, Belt Simulation & API Integration
// =============================================================

let API_BASE = "http://localhost:8005";

// ── Estado global ──
let sessionActive    = false;
let sessionId        = null;
let simulationRafId  = null;   // requestAnimationFrame handle
let lastInferenceTime = 0;     // para throttle de 500ms

// Bboxes detectadas en el último frame: [{x1,y1,x2,y2, label, conf}]
let activeBboxes = [];

// ── Estado de Simulación Fotorrealista de Set Completo ──
let setSimulationActive = false;
let setSimulationImage  = null;
let setSimulationMeta   = null;
let setScrollY          = 0;
let lastFrameTime       = 0;
let setSequentialImages = [];


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
    initInferenceTestPanel();
    initMulticamPanel();
    initStep5Pipelines();
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
            if (pywebview.api.get_api_base) {
                const base = await pywebview.api.get_api_base();
                if (base) {
                    API_BASE = base;
                    console.log("[LegoVision JS] API_BASE reconfigurada dinámicamente a:", API_BASE);
                }
            }
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
    { ref: "3022", name: "Plate 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 14 },
    { ref: "2877", name: "Brick 1x2 Grille", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 4 },
    { ref: "59900", name: "Cone 1x1", w: 14, h: 14, color: "#C91A09", colorName: "Rojo Trans.", qty: 4 },
    { ref: "3003", name: "Brick 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3002", name: "Brick 2x3", w: 42, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3005", name: "Brick 1x1", w: 14, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 4 },
    { ref: "3010", name: "Brick 1x4", w: 56, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3021", name: "Plate 2x3", w: 42, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3023", name: "Plate 1x2", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 10 },
    { ref: "3024", name: "Plate 1x1", w: 14, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 6 },
    { ref: "2420", name: "Plate 2x2 Corner", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3710", name: "Plate 1x4", w: 56, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 3 },
    { ref: "3622", name: "Brick 1x3", w: 42, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3665", name: "Slope Inverted 45 2x1", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 2 },
    { ref: "3039", name: "Slope 45 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "4070", name: "Brick 1x1 Headlight", w: 14, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "6141", name: "Plate 1x1 Round", w: 14, h: 14, color: "#C91A09", colorName: "Rojo", qty: 4 },
    { ref: "15573", name: "Plate 1x2 with 1 Stud", w: 28, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "2412", name: "Tile 1x2 Grille", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 2 },
    { ref: "3069", name: "Tile 1x2", w: 28, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3068", name: "Tile 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "60478", name: "Plate 1x2 with Handle", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 2 },
    { ref: "48336", name: "Plate 1x2 with Handle Side", w: 28, h: 14, color: "#1B1B1B", colorName: "Negro", qty: 2 },
    { ref: "32000", name: "Brick 1x2 with 2 Holes", w: 28, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3700", name: "Brick 1x2 with Hole", w: 28, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3701", name: "Brick 1x4 with 3 Holes", w: 56, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 1 },
    { ref: "4032", name: "Plate 2x2 Round", w: 28, h: 28, color: "#1B1B1B", colorName: "Negro", qty: 2 },
    { ref: "3062", name: "Brick 1x1 Round", w: 14, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "85984", name: "Slope 30 1x2x2/3", w: 28, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "54200", name: "Slope 30 1x1x2/3", w: 14, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 3 },
    { ref: "99206", name: "Plate 2x2x2/3 side studs", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "3037", name: "Slope 45 2x4", w: 56, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 1 },
    { ref: "3298", name: "Slope 33 3x2", w: 42, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 1 },
    { ref: "11477", name: "Slope Curved 2x1", w: 28, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "15068", name: "Slope Curved 2x2", w: 28, h: 28, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "98138", name: "Tile 1x1 Round", w: 14, h: 14, color: "#C91A09", colorName: "Rojo", qty: 3 },
    { ref: "2431", name: "Tile 1x4", w: 56, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 2 },
    { ref: "6636", name: "Tile 1x6", w: 84, h: 14, color: "#A0A5A9", colorName: "Gris", qty: 1 }
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
    },
    "911943-1": {
        name: "Luke Skywalker foil pack #1 (Star Wars)",
        parts: [
            { ref: "sw0778", name: "Luke Skywalker (Tatooine)", w: 32, h: 42, color: "#FFFFFF", colorName: "Blanco", colorCode: "15", qty: 1 },
            { ref: "64567", name: "Lightsaber Hilt Straight", w: 22, h: 22, color: "#899395", colorName: "Flat Silver", colorCode: "67", qty: 1 },
            { ref: "30374", name: "Bar 4L (Lightsaber Blade)", w: 10, h: 56, color: "#AEE9EF", colorName: "Trans-Light Blue", colorCode: "15", qty: 1 }
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
let maxPiecesInField = 30; // Valor del slider de piezas en campo

// píxeles por metro en la simulación (Cinta de 200mm = canvas width)
// El UX representa una cinta con un ancho de 20cm (200mm).
const BELT_WIDTH_MM = 200;
const CANVAS_W = 640;
const CANVAS_H = 640;
const PX_PER_MM = CANVAS_W / BELT_WIDTH_MM; // 640 / 200 = 3.2 px/mm
const MM_PER_MIN_TO_PX_PER_FRAME = PX_PER_MM / 60 / 30; // @ 30fps

function initLiveView() {
    canvas = document.getElementById("live-canvas");
    if (!canvas) {
        console.warn("Element 'live-canvas' not found, skipping live view canvas initialization.");
        return;
    }
    ctx    = canvas.getContext("2d");
    canvas.width  = CANVAS_W;
    canvas.height = CANVAS_H;
    drawBeltBackground();

    // Click en el canvas → clasificar la pieza en la bbox
    canvas.addEventListener("click", onCanvasClick);

    resetSessionIdentifiedCounts();
}


function drawBeltBackground() {
    // Fondo azul petróleo claro tipo cinta
    const grad = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
    grad.addColorStop(0,   "#1b303f");
    grad.addColorStop(0.5, "#254154");
    grad.addColorStop(1,   "#1b303f");
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

    const ref = part.ref;
    const colorHex = part.color ? part.color.replace("#", "") : "808080";
    const img = new Image();
    img.src = `${API_BASE}/renders/render_${ref}_${colorHex}.png`;

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
        restitution: 0.2 + Math.random() * 0.15, // rebote
        img: img
    };
}

function initBeltPieces() {
    refillSimulationSpawnPool();
    beltPieces = [];
    // Pre-poblar con la cantidad indicada por el slider
    const MIN_INIT_DIST = 0; // sin margen extra; la repulsión limita el solapamiento a ≤20%
    for (let i = 0; i < maxPiecesInField; i++) {
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

    // Cuerpo principal o Imagen 3D
    if (p.img && p.img.complete && p.img.naturalWidth !== 0) {
        ctx.drawImage(p.img, -p.w/2, -p.h/2, p.w, p.h);
    } else {
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
    }

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
    if (sessionActive) {
        // Detener recursión del loop IDLE si hay una sesión de inferencia física activa
        return;
    }
    simulationRafId = requestAnimationFrame(animationLoop);

    // Calcular delta time en segundos
    if (!lastFrameTime) lastFrameTime = timestamp;
    const dt = (timestamp - lastFrameTime) / 1000;
    lastFrameTime = timestamp;

    // Limpiar y redibujar fondo
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    if (setSimulationActive && setSimulationImage) {
        // --- NUEVO MODO: SIMULACIÓN DE SET COMPLETO CON RENDER DE BLENDER ---
        
        // Calcular velocidad física exacta:
        // beltSpeed es en m/min.
        // Velocidad real = beltSpeed * 1000 / 60 mm/s.
        // En el UX, 20 cm (200 mm) = 640 px (ancho de la cinta), por lo que la escala es 3.2 px/mm.
        const speed_mm_s = sessionActive ? ((beltSpeed * 1000) / 60) : 0;
        const speed_px_s = speed_mm_s * 3.2;
        
        // Solo avanza si la sesión está activa
        setScrollY += speed_px_s * dt;

        const imgW = CANVAS_W;
        // Altura proporcional al aspect ratio original
        const imgH = setSimulationImage.height * (CANVAS_W / setSimulationImage.width);

        // Bucle infinito para visualización continua
        if (setScrollY > imgH) {
            setScrollY = 0;
            // Si la sesión está activa y terminó el lote de piezas, se reinician los disparadores
            if (sessionActive && setSimulationMeta && setSimulationMeta.detections) {
                setSimulationMeta.detections.forEach(d => {
                    d.hasBeenCaptured = false;
                    d.hasBeenCounted = false;
                });
            }
        }

        // Dibujar la cinta larga fotorrealista desplazándose
        ctx.drawImage(setSimulationImage, 0, setScrollY - imgH, imgW, imgH);
        
        // Si el final de la cinta está pasando por arriba, rellenar el inicio para que no quede fondo negro
        if (setScrollY > 0) {
            ctx.drawImage(setSimulationImage, 0, setScrollY - 2 * imgH, imgW, imgH);
        }

        // Proyectar bounding boxes activas basadas en la posición de scroll
        activeBboxes = [];
        if (setSimulationMeta && setSimulationMeta.detections) {
            const viewportYMin = 0;
            const viewportYMax = CANVAS_H;

            setSimulationMeta.detections.forEach(d => {
                const [x1_n, y1_n, x2_n, y2_n] = d.bbox_norm;
                
                // Coordenadas absolutas de la pieza en la cinta larga escalada
                const y1_img = y1_n * imgH;
                const y2_img = y2_n * imgH;
                
                // Coordenadas en el Canvas de pantalla en función del scroll
                const x1_canvas = x1_n * CANVAS_W;
                const x2_canvas = x2_n * CANVAS_W;
                
                // Calculamos dos posibles posiciones en pantalla por el scrolling infinito
                let y1_canvas = y1_img + (setScrollY - imgH);
                let y2_canvas = y2_img + (setScrollY - imgH);

                // Si esta copia se salió de pantalla pero la de la siguiente sección ya entró:
                if (y2_canvas < viewportYMin && setScrollY > 0) {
                    y1_canvas = y1_img + (setScrollY - 2 * imgH);
                    y2_canvas = y2_img + (setScrollY - 2 * imgH);
                }

                // Si la pieza está en el rango visible
                if (y2_canvas >= viewportYMin && y1_canvas <= viewportYMax) {
                    activeBboxes.push({
                        x1: x1_canvas,
                        y1: y1_canvas,
                        x2: x2_canvas,
                        y2: y2_canvas,
                        label: d.ref,
                        conf: d.similarity,
                        name: d.name,
                        color_hex: d.color_hex
                    });

                    // Si la sesión está activa y entra en la zona de la cámara (centro de la pantalla)
                    const camYMin = CANVAS_H * 0.35;
                    const camYMax = CANVAS_H * 0.55;
                    const cy = (y1_canvas + y2_canvas) / 2.0;

                    if (sessionActive && cy >= camYMin && cy <= camYMax) {
                        // 1. Marcar como contada si no se ha hecho
                        if (!d.hasBeenCounted) {
                            d.hasBeenCounted = true;
                            sessionIdentifiedCounts[d.ref] = (sessionIdentifiedCounts[d.ref] || 0) + 1;
                            updateSessionInventoryTable();
                            
                            // Emitir sonido metálico/industrial leve al detectar
                            try {
                                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                                const osc = audioCtx.createOscillator();
                                const gain = audioCtx.createGain();
                                osc.connect(gain);
                                gain.connect(audioCtx.destination);
                                osc.type = 'sine';
                                osc.frequency.setValueAtTime(880, audioCtx.currentTime); // tono agudo corto
                                gain.gain.setValueAtTime(0.03, audioCtx.currentTime);
                                gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.15);
                                osc.start();
                                osc.stop(audioCtx.currentTime + 0.15);
                            } catch(e) {}
                        }

                        // 2. Capturar imagen secuencial del visor si no se ha capturado en esta pasada
                        if (!d.hasBeenCaptured) {
                            d.hasBeenCaptured = true;
                            
                            // Extraer recorte de la pieza desde el canvas
                            const cw = Math.max(16, x2_canvas - x1_canvas);
                            const ch = Math.max(16, y2_canvas - y1_canvas);
                            const offscreen = document.createElement("canvas");
                            offscreen.width = cw;
                            offscreen.height = ch;
                            const octx = offscreen.getContext("2d");
                            octx.drawImage(canvas, x1_canvas, y1_canvas, cw, ch, 0, 0, cw, ch);
                            const cropUrl = offscreen.toDataURL("image/png");

                            // Añadir a la lista de imágenes secuenciales
                            setSequentialImages.unshift({
                                dataUrl: cropUrl,
                                name: d.name,
                                ref: d.ref,
                                conf: d.similarity,
                                color_hex: d.color_hex,
                                time: new Date().toLocaleTimeString()
                            });
                            
                            // Limitar a las últimas 15 capturas para no sobrecargar el DOM
                            if (setSequentialImages.length > 15) {
                                setSequentialImages.pop();
                            }
                            updateSequentialCameraList();
                        }
                    }
                }
            });
        }

        // Pintar overlay de bounding boxes
        drawBboxOverlay();
        
        // Actualizar estadísticas de piezas detectadas en frame en vivo
        document.getElementById("live-frame-pieces").innerText = activeBboxes.length;

    } else {
        // --- MODO ANTERIOR: SIMULACIÓN DE PARTÍCULAS LEGO ---
        // Calcular velocidad de la cinta en px/frame
        const vy = sessionActive ? (beltSpeed * 1000 / 60 / 30 * PX_PER_MM) : 0;

        if (sessionActive) {
            const totalParts = currentSetParts.reduce((sum, p) => sum + p.qty, 0);
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
                    
                    for (let k = 0; k < 15; k++) {
                        const margin = 20 + Math.max(nextPiece.w, nextPiece.h) / 2;
                        let cx = margin + Math.random() * (CANVAS_W - 2 * margin);
                        let cy = -Math.max(nextPiece.w, nextPiece.h) - 50 - Math.random() * 40;
                        
                        let tooClose = false;
                        for (let i = 0; i < beltPieces.length; i++) {
                            let bp = beltPieces[i];
                            let dx = bp.x - cx;
                            let dy = bp.y - cy;
                            let dist = Math.sqrt(dx*dx + dy*dy);
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

            // Repulsión física suave
            for (let i = 0; i < beltPieces.length; i++) {
                const pi = beltPieces[i];
                if (pi.z > 5) continue;
                for (let j = i + 1; j < beltPieces.length; j++) {
                    const pj = beltPieces[j];
                    if (pj.z > 5) continue;
                    const dx = pi.x - pj.x;
                    const dy = pi.y - pj.y;
                    const dist = Math.hypot(dx, dy) || 0.001;
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

            if (exitedCount >= totalParts && beltPieces.length === 0) {
                const btnToggle = document.getElementById("btn-toggle-session");
                if (btnToggle && btnToggle.innerText.includes("Detener")) {
                    btnToggle.click();
                }
            }
        } else {
            beltPieces.forEach(p => {
                updatePiecePhysics(p, vy);
                if (p.y > CANVAS_H + 60) {
                    const sp = spawnPiece();
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

        activeBboxes.forEach(bb => {
            bb.y1 += vy;
            bb.y2 += vy;
        });

        if (sessionActive && timestamp - lastInferenceTime > 500) {
            lastInferenceTime = timestamp;
            captureAndDetect();
        }

        drawBboxOverlay();
    }
}

async function triggerSetSimulation() {
    if (typeof pywebview === "undefined" || !pywebview.api) {
        alert("Esta función requiere ejecutar la aplicación con Python (gui/app.py)");
        return;
    }
    const btnSimulateSet3D = document.getElementById("btn-simulate-set-3d");
    const originalText = btnSimulateSet3D.innerText;
    btnSimulateSet3D.innerText = "⏳ Generando físicas en Blender...";
    btnSimulateSet3D.disabled = true;
    btnSimulateSet3D.style.opacity = "0.7";

    try {
        console.log("Iniciando simulación física de set completo en Blender para set:", currentSetId);
        const res = await pywebview.api.simulate_set_physics_scatter(currentSetId);
        
        btnSimulateSet3D.innerText = originalText;
        btnSimulateSet3D.disabled = false;
        btnSimulateSet3D.style.opacity = "1";

        if (res.status === "success") {
            console.log("Simulación completada con éxito:", res);
            setSimulationMeta = res.metadata;
            
            // Cargar imagen de la simulación
            setSimulationImage = new Image();
            setSimulationImage.onload = () => {
                setSimulationActive = true;
                setScrollY = 0;
                setSequentialImages = [];
                updateSequentialCameraList();
                
                // Mostrar indicador de simulación activa
                const statusWrap = document.getElementById("set-simulation-status-wrap");
                if (statusWrap) statusWrap.style.display = "block";
                
                resetSessionIdentifiedCounts();
                console.log("Imagen de simulación cargada y lista. Activando modo Set Simulation.");
            };
            setSimulationImage.src = res.image_url;
            
            alert("¡Simulación física 3D en Blender completada!\nLas piezas se han estabilizado sin solapamientos a escala real y se han cargado en la Vista en Vivo.");
        } else {
            alert("Error en simulación: " + res.message + "\nDetalles: " + (res.details || ""));
        }
    } catch (e) {
        btnSimulateSet3D.innerText = originalText;
        btnSimulateSet3D.disabled = false;
        btnSimulateSet3D.style.opacity = "1";
        console.error("Error al simular set en Blender:", e);
        alert("Ocurrió un error ejecutando la simulación en Blender: " + e.message);
    }
}

function updateSequentialCameraList() {
    const listEl = document.getElementById("sequential-camera-list");
    if (!listEl) return;

    if (setSequentialImages.length === 0) {
        listEl.innerHTML = `<div style="color: var(--text-secondary); font-size: 0.8rem; text-align: center; padding: 12px;">Ninguna captura secuencial registrada.</div>`;
        return;
    }

    listEl.innerHTML = setSequentialImages.map(img => {
        const pct = Math.round(img.conf * 100);
        return `
            <div class="recent-item" style="display: flex; align-items: center; gap: 10px; background: rgba(30, 41, 59, 0.4); border: 1px solid var(--border); padding: 8px; border-radius: 8px; margin-bottom: 2px;">
                <img src="${img.dataUrl}" style="width: 48px; height: 48px; object-fit: contain; background: #000; border-radius: 4px; border: 1px solid var(--border);" alt="crop">
                <div class="piece-info" style="flex: 1; display: flex; flex-direction: column;">
                    <span class="piece-class" style="font-size: 0.82rem; font-weight: bold; color: var(--accent);">${img.ref}</span>
                    <span class="piece-name" style="font-size: 0.72rem; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;">${img.name}</span>
                    <span style="font-size: 0.65rem; color: var(--text-secondary);">${img.time}</span>
                </div>
                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                    <span class="piece-conf high" style="font-size: 0.78rem; padding: 2px 6px;">${pct}%</span>
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: ${img.color_hex || '#fff'}; border: 1px solid #475569;" title="Color original"></span>
                </div>
            </div>
        `;
    }).join("");
}


function startBeltSimulation(skipInit) {
    if (simulationRafId) cancelAnimationFrame(simulationRafId);
    if (!skipInit) {
        // IDLE mode: pre-populate belt with 6 pieces for visual feedback
        initBeltPieces();
    } else {
        // SESSION mode: belt is already cleared; pieces will spawn from top
        refillSimulationSpawnPool();
    }
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
// 6c. Slider de Piezas en Campo
// =============================================================
function initLivePiecesSlider() {
    const slider  = document.getElementById("live-pieces-slider");
    const display = document.getElementById("live-pieces-value");
    if (!slider) return;
    maxPiecesInField = parseInt(slider.value);
    display.innerText = `${maxPiecesInField}`;
    slider.addEventListener("input", () => {
        maxPiecesInField = parseInt(slider.value);
        display.innerText = `${maxPiecesInField}`;
        if (!sessionActive) {
            // Si la sesión no está activa, refrescamos la cinta para mostrar la nueva densidad de piezas
            initBeltPieces();
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    // ... logic would be initialized, but we add initLivePiecesSlider manually later or in DOM ready...
    initLivePiecesSlider(); // Se asegura de que escuche. (Omitir si está en initLiveView, pero lo ponemos para mayor seguridad)
});

// =============================================================
// 7. Inferencia YOLO → captura el frame y lo envía al API
// =============================================================
async function captureAndDetect() {
    return new Promise(resolve => {
        canvas.toBlob(async blob => {
            const formData = new FormData();
            formData.append("file", blob, "live_frame.png");
            if (sessionActive && typeof currentSessionId !== "undefined") {
                formData.append("session_id", currentSessionId);
            }

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

                const bboxCountEl = document.getElementById("live-bbox-count"); if (bboxCountEl) bboxCountEl.innerText = activeBboxes.length;
                canvas.classList.toggle("has-detections", activeBboxes.length > 0);

            } catch (e) {
                // API offline: usar piezas simuladas como bboxes de demo
                activeBboxes = beltPieces.map(p => {
                    const hw = p.w/2, hh = p.h/2;
                    const R = Math.sqrt(hw*hw + hh*hh);
                    // Garantizar bbox mínima de 20px para piezas muy pequeñas
                    const minR = Math.max(R, 14);
                    return {
                        x1: p.x - minR, y1: p.y - minR,
                        x2: p.x + minR, y2: p.y + minR,
                        label: p.ref,
                        conf:  0.85 + Math.random() * 0.13,
                    };
                });

                // En modo offline marcamos las piezas de la simulación como detectadas
                beltPieces.forEach(p => {
                    p.isDetected = true;
                });

                canvas.classList.toggle("has-detections", activeBboxes.length > 0);
                const bboxCountEl = document.getElementById("live-bbox-count"); if (bboxCountEl) bboxCountEl.innerText = activeBboxes.length;
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

            // Iniciar inferencia sobre el render ya cargado (setSimulationImage/setSimulationMeta)
            btnToggle.innerText = "⏳ Iniciando inferencia...";
            btnToggle.disabled = true;
            badge.innerText = "Iniciando...";
            badge.className = "piece-conf high";

            try {
                // Usar el render ya cargado en setSimulationImage (cargado por btn-generate-render o selector)
                let inferenceRenderResult = null;
                if (setSimulationImage && setSimulationMeta) {
                    inferenceRenderResult = { status: "success", metadata: setSimulationMeta, image_url: setSimulationImage.src };
                }

                // Registrar sesion en el backend de inferencia (opcional, puede fallar)
                let sessionId_local = null;
                try {
                    const res = await fetch(`${API_BASE}/session/start`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            model_version:   "yolov8n_synthetic",
                            belt_speed_mm_s: beltSpeed * 1000 / 60,
                            set_id:          currentSetId || "75078-1",
                        }),
                    });
                    const data = await res.json();
                    if (data.status === "success") {
                        sessionId_local = data.session_id;
                        liveSessId.innerText = sessionId_local;
                    }
                } catch (apiErr) {
                    console.warn("[InferenceTest] Backend API offline, modo demo.");
                    liveSessId.innerText = "demo-local";
                }

                sessionActive = true;
                sessionId = sessionId_local;
                btnToggle.innerText = "Detener Inferencia";
                btnToggle.style.background = "var(--accent-red)";
                btnToggle.disabled = false;
                badge.innerText = "Sesión Activa";
                badge.className = "piece-conf high";

                // Si se genero render Blender con exito, activar modo set-simulation
                if (inferenceRenderResult && inferenceRenderResult.status === "success") {
                    setSimulationMeta = inferenceRenderResult.metadata;
                    setSimulationImage = new Image();
                    setSimulationImage.onload = () => {
                        setSimulationActive = true;
                        setScrollY = 0;
                        setSequentialImages = [];
                        updateSequentialCameraList();
                        const statusWrap = document.getElementById("set-simulation-status-wrap");
                        if (statusWrap) {
                            statusWrap.style.display = "block";
                            const badge2 = document.getElementById("set-sim-badge");
                            if (badge2) badge2.textContent = "Render Blender · stable_poses";
                        }
                        resetSessionIdentifiedCounts();
                        startBeltSimulation(true);
                        startHistoryPolling();
                        console.log("[InferenceTest] Render cargado y simulacion activada.");
                    };
                    setSimulationImage.onerror = () => {
                        console.warn("[InferenceTest] Error cargando imagen del render. Fallback a physBelt.");
                        startBeltSimulation(true);
                        startHistoryPolling();
                    };
                    setSimulationImage.src = inferenceRenderResult.image_url;
                } else {
                    // Fallback: physBelt de particulas 2D
                    console.warn("[InferenceTest] Sin render Blender, usando simulacion de particulas.");
                    startBeltSimulation(true);
                    startHistoryPolling();
                }
            } catch (e) {
                console.warn("[InferenceTest] Error general, fallback a simulacion demo:", e);
                sessionActive = true;
                btnToggle.innerText = "Detener Simulación";
                btnToggle.style.background = "var(--accent-red)";
                btnToggle.disabled = false;
                badge.innerText = "Demo (sin API)";
                badge.className = "piece-conf high";
                startBeltSimulation(true);
                startHistoryPolling();
            }
        } else {
            // Detener
            fetch(`${API_BASE}/session/stop`, { method: "POST" }).catch(e => {
                console.warn("Failed to stop session on backend:", e);
            });

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

    if (btnCapture) {
        btnCapture.addEventListener("click", () => captureAndDetect());
    }
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
        const epochs  = parseInt(document.getElementById("train-epochs").value)  || 50;
        const batch   = parseInt(document.getElementById("train-batch").value)   || 16;
        const dsize   = parseInt(document.getElementById("train-dataset-size").value) || 500;
        const piecesPerImage = parseInt(document.getElementById("train-pieces-per-image")?.value) || 25;
        const emptyRatio = parseFloat(document.getElementById("train-empty-ratio")?.value) || 5.0;

        setStepStatus("step-1", "running", "▶ Generando dataset...");
        // Show YOLO mini progress bar
        const yoloWrap = document.getElementById("yolo-progress-wrap");
        if (yoloWrap) {
            yoloWrap.style.display = "block";
            document.getElementById("yolo-mini-bar").style.width = "0%";
            document.getElementById("yolo-progress-label").textContent = "Generando dataset...";
            document.getElementById("yolo-progress-metrics").textContent = "Loss: — · mAP50: —";
        }
        appendLog(`\n> Iniciando pipeline: ${dsize} imágenes · ${piecesPerImage}±5 piezas/img · ${epochs} épocas · batch=${batch}`);
        appendLog(`> Set: 75078-1 (Imperial Troop Transport) · Domain Randomization activado`);

        try {
            if (typeof pywebview !== "undefined" && pywebview.api) {
                const res = await pywebview.api.start_training(epochs, dsize, batch, piecesPerImage, emptyRatio);
                appendLog(`> ${res.message}`);
                // Mostrar boton Stop YOLO
                const btnStop = document.getElementById("btn-stop-training");
                if (btnStop) btnStop.style.display = "inline-block";
            } else {
                appendLog("> [DEMO] API de PyWebView no disponible en modo navegador.");
                setStepStatus("step-1", "error", "Sin API");
                return;
            }

            // Iniciar polling de progreso
            if (trainingPollInterval) clearInterval(trainingPollInterval);
            trainingPollInterval = setInterval(pollTrainingStatus, 30000);
        } catch (e) {
            appendLog(`> ERROR: ${e.message}`);
            setStepStatus("step-1", "error", "Error");
        }
    });

    document.getElementById("btn-start-indexing").addEventListener("click", async () => {
        setStepStatus("step-2", "running", "▶ Generando refs + Indexando...");
        appendLog("\n> Iniciando indexación DINOv2: genera refs multi-ángulo (física Blender) + embeddings ViT-S/14...");
        appendLog("> Set: 75078-1 · 20 caídas físicas por pieza · Cámara ortogonal 640×640px");
        // Show DINOv2 progress bar
        const dinoWrap = document.getElementById("dino-progress-wrap");
        if (dinoWrap) {
            dinoWrap.style.display = "block";
            document.getElementById("dino-mini-bar").style.width = "0%";
            document.getElementById("dino-progress-label").textContent = "Preparando...";
            document.getElementById("dino-progress-pct").textContent = "0%";
        }
        // Start polling indexing progress
        if (window._dinoProgressInterval) clearInterval(window._dinoProgressInterval);
        window._dinoProgressInterval = setInterval(pollIndexingProgress, 1500);

        try {
            if (typeof pywebview !== "undefined" && pywebview.api) {
                const res = await pywebview.api.start_indexing("75078-1");
                appendLog(`> ${res.message}`);
                // Mostrar boton Stop DINOv2
                const btnStopIdx = document.getElementById("btn-stop-indexing");
                if (btnStopIdx) btnStopIdx.style.display = "inline-block";
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

    document.getElementById("btn-refresh-embeddings").addEventListener("click", openDinoFovModal);

    // ── Botones del Modal DINO-FOV ─────────────────────────
    document.getElementById("btn-close-dino-modal").addEventListener("click", closeDinoFovModal);
    document.getElementById("btn-cancel-dino-sim").addEventListener("click", closeDinoFovModal);
    document.getElementById("btn-submit-dino-sim").addEventListener("click", startDinoFovSimulation);
    document.getElementById("btn-stop-dino-fov").addEventListener("click", stopDinoFovSimulation);

    // ── Botones STOP entrenamiento YOLO y DINOv2 ─────────────────────────
    const btnStopTraining = document.getElementById("btn-stop-training");
    if (btnStopTraining) {
        btnStopTraining.addEventListener("click", async () => {
            if (!confirm("¿Detener el entrenamiento YOLO en curso? El progreso parcial se perderá.")) return;
            btnStopTraining.disabled = true;
            btnStopTraining.textContent = "⏳ Deteniendo...";
            try {
                if (typeof pywebview !== "undefined" && pywebview.api) {
                    const res = await pywebview.api.stop_training();
                    appendLog(`> ⏹ STOP YOLO: ${res.message}`);
                    setStepStatus("step-1", "error", "⏹ Cancelado");
                }
            } catch(e) {
                appendLog(`> Error deteniendo YOLO: ${e.message}`);
            } finally {
                btnStopTraining.style.display = "none";
                btnStopTraining.disabled = false;
                btnStopTraining.textContent = "⏹ Stop";
            }
        });
    }

    const btnStopIndexing = document.getElementById("btn-stop-indexing");
    if (btnStopIndexing) {
        btnStopIndexing.addEventListener("click", async () => {
            if (!confirm("¿Detener la indexación DINOv2 en curso? Los embeddings parciales permanecerán en BD.")) return;
            btnStopIndexing.disabled = true;
            btnStopIndexing.textContent = "⏳ Deteniendo...";
            try {
                if (typeof pywebview !== "undefined" && pywebview.api) {
                    const res = await pywebview.api.stop_indexing();
                    appendLog(`> ⏹ STOP DINOv2: ${res.message}`);
                    setStepStatus("step-2", "error", "⏹ Cancelado");
                    if (window._dinoProgressInterval) { clearInterval(window._dinoProgressInterval); window._dinoProgressInterval = null; }
                }
            } catch(e) {
                appendLog(`> Error deteniendo DINOv2: ${e.message}`);
            } finally {
                btnStopIndexing.style.display = "none";
                btnStopIndexing.disabled = false;
                btnStopIndexing.textContent = "⏹ Stop";
            }
        });
    }
    document.getElementById("btn-clear-logs").addEventListener("click", () => {
        document.getElementById("training-logs").textContent = "> LegoVision Training Terminal\n> Logs limpios.\n";
    });

    const btnRefreshEval = document.getElementById("btn-refresh-eval");
    if (btnRefreshEval) btnRefreshEval.addEventListener("click", loadEvalResults);

    // ── Botones VALIDACIÓN DE ESTABILIDAD (Fase 4) ─────────────────────────
    const btnStartVal = document.getElementById("btn-start-validation");
    if (btnStartVal) btnStartVal.addEventListener("click", startValidation);

    const btnStopVal = document.getElementById("btn-stop-validation");
    if (btnStopVal) btnStopVal.addEventListener("click", stopValidation);

    // ── Botones EXPORTAR EXCEL (Fase 4 - post validacion) ─────────────────
    const btnExportExcel = document.getElementById("btn-export-excel");
    if (btnExportExcel) btnExportExcel.addEventListener("click", exportValidationExcel);

    const btnOpenExcel = document.getElementById("btn-open-excel");
    if (btnOpenExcel) btnOpenExcel.addEventListener("click", openValidationExcel);

    // Intentar cargar estado inicial
    window.addEventListener("pywebviewready", () => {
        pollTrainingStatus();
        refreshEmbeddingCount();
        loadScopeInfo();
        loadEvalResults();
        pollValidationProgress();
        loadPreviousValidationResults();
        pollDinoFovProgressOnLoad();
    });
    
}

let validationPollInterval = null;

async function startValidation() {
    const runs = parseInt(document.getElementById("validation-runs").value) || 20;

    setStepStatus("step-4", "running", "▶ Validando...");
    appendLog("\n> Iniciando validación de posiciones estables para set 75078-1...");
    appendLog(`> ${runs} lanzamientos (simulación física en Blender) por cada geometría única...`);
    
    // Mostrar progreso de validación
    const progressWrap = document.getElementById("validation-progress-wrap");
    if (progressWrap) {
        progressWrap.style.display = "block";
        document.getElementById("validation-mini-bar").style.width = "0%";
        document.getElementById("validation-progress-label").textContent = "Preparando...";
        document.getElementById("validation-progress-file").textContent = "";
    }
    
    const btnStop = document.getElementById("btn-stop-validation");
    if (btnStop) btnStop.style.display = "inline-block";

    try {
        if (typeof pywebview !== "undefined" && pywebview.api) {
            const res = await pywebview.api.start_validation(runs);
            appendLog(`> ${res.message}`);
            
            // Iniciar polling
            if (validationPollInterval) clearInterval(validationPollInterval);
            validationPollInterval = setInterval(pollValidationProgress, 1500);
        } else {
            appendLog("> [DEMO] API de PyWebView no disponible en modo navegador.");
            setStepStatus("step-4", "error", "Sin API");
        }
    } catch (e) {
        appendLog(`> ERROR: ${e.message}`);
        setStepStatus("step-4", "error", "Error");
    }
}

async function pollValidationProgress() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const prog = await pywebview.api.get_validation_progress();
        const bar = document.getElementById("validation-mini-bar");
        const label = document.getElementById("validation-progress-label");
        const pctEl = document.getElementById("validation-progress-pct");
        const fileEl = document.getElementById("validation-progress-file");
        const wrap = document.getElementById("validation-progress-wrap");
        const btnStop = document.getElementById("btn-stop-validation");
        
        if (prog.active) {
            setStepStatus("step-4", "running", `Pieza ${prog.current}/${prog.total}`);
            if (bar) bar.style.width = `${prog.pct}%`;
            if (label) label.textContent = `Pieza ${prog.current} / ${prog.total}`;
            if (pctEl) pctEl.textContent = `${prog.pct}%`;
            if (fileEl && prog.current_piece) fileEl.textContent = prog.current_piece;
            if (wrap) wrap.style.display = "block";
            if (btnStop) btnStop.style.display = "inline-block";
            
            if (!validationPollInterval) {
                validationPollInterval = setInterval(pollValidationProgress, 1500);
            }
        }
        
        if (prog.done && prog.active === false) {
            if (bar) bar.style.width = "100%";
            if (pctEl) pctEl.textContent = "100% ✓";
            if (fileEl) fileEl.textContent = "Validación completada";
            setStepStatus("step-4", "done", "Validado ✓");
            if (btnStop) btnStop.style.display = "none";
            
            if (validationPollInterval) {
                clearInterval(validationPollInterval);
                validationPollInterval = null;
            }
            
            // Renderizar los resultados
            if (prog.results) {
                renderValidationResults(prog.results);
            }
            // Mostrar boton de exportar Excel al completar
            const btnExcel = document.getElementById("btn-export-excel");
            if (btnExcel) btnExcel.style.display = "";
        }
        
        if (prog.error) {
            setStepStatus("step-4", "error", "Error");
            appendLog(`> Error validación: ${prog.error}`);
            if (btnStop) btnStop.style.display = "none";
            if (validationPollInterval) {
                clearInterval(validationPollInterval);
                validationPollInterval = null;
            }
        }
    } catch(e) {
        console.warn("pollValidationProgress:", e);
    }
}

async function stopValidation() {
    if (!confirm("¿Detener la validación en curso? El progreso parcial se perderá.")) return;
    const btnStop = document.getElementById("btn-stop-validation");
    if (btnStop) {
        btnStop.disabled = true;
        btnStop.textContent = "⏳ Deteniendo...";
    }
    try {
        if (typeof pywebview !== "undefined" && pywebview.api) {
            const res = await pywebview.api.stop_validation();
            appendLog(`> ⏹ STOP VALIDACIÓN: ${res.message}`);
            setStepStatus("step-4", "error", "⏹ Cancelado");
            if (validationPollInterval) {
                clearInterval(validationPollInterval);
                validationPollInterval = null;
            }
        }
    } catch (e) {
        appendLog(`> Error deteniendo validación: ${e.message}`);
    } finally {
        if (btnStop) {
            btnStop.style.display = "none";
            btnStop.disabled = false;
            btnStop.textContent = "⏹ Stop";
        }
    }
}

function renderValidationResults(results) {
    const container = document.getElementById("stability-validation-panel");
    const tbody = document.getElementById("stability-validation-table-body");
    const badge = document.getElementById("validation-summary-badge");
    
    if (!container || !tbody) return;
    
    tbody.innerHTML = "";
    let totalDiscrepancies = 0;
    
    if (results && results.report) {
        results.report.forEach(item => {
            const row = document.createElement("tr");
            row.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
            row.style.background = item.discrepancy ? "rgba(239, 68, 68, 0.05)" : "transparent";
            
            if (item.discrepancy) {
                totalDiscrepancies++;
            }
            
            const mapFace = f => {
                if (f === 0) return "Cara 0 (Base)";
                if (f === 1) return "Cara 1 (Lateral)";
                if (f === 2) return "Cara 2 (Top)";
                return `Cara ${f}`;
            };
            
            const expFacesStr = item.experimental_faces.length > 0 
                ? item.experimental_faces.map(mapFace).join(", ") 
                : "Ninguna";
                
            const dbFacesStr = item.database_faces.length > 0 
                ? item.database_faces.map(mapFace).join(", ") 
                : "Ninguna";
                
            const missingStr = item.missing_in_db.length > 0 
                ? item.missing_in_db.map(mapFace).join(", ") 
                : "—";
                
            const extraStr = item.extra_in_db.length > 0 
                ? item.extra_in_db.map(mapFace).join(", ") 
                : "—";
                
            const statusHtml = item.discrepancy 
                ? `<span class="badge-status more">Discrepancia</span>` 
                : `<span class="badge-status ok">Correcto</span>`;
                
            row.innerHTML = `
                <td style="padding:10px 8px; font-weight:600; font-family:monospace; color:var(--accent);">${item.part_ref}</td>
                <td style="padding:10px 8px; color:#fff;">${item.name}</td>
                <td style="padding:10px 8px; text-align:center;">${item.experimental_poses_count}</td>
                <td style="padding:10px 8px; font-size:0.78rem; color:var(--text-secondary);">${expFacesStr}</td>
                <td style="padding:10px 8px; font-size:0.78rem; color:var(--text-secondary);">${dbFacesStr}</td>
                <td style="padding:10px 8px; font-size:0.78rem; color:#ef4444; font-weight:500;">${missingStr}</td>
                <td style="padding:10px 8px; font-size:0.78rem; color:#f59e0b; font-weight:500;">${extraStr}</td>
                <td style="padding:10px 8px; text-align:center;">${statusHtml}</td>
            `;
            tbody.appendChild(row);
        });
        
        if (badge) {
            badge.innerText = `${totalDiscrepancies} Discrepancias`;
            badge.className = totalDiscrepancies > 0 ? "piece-conf low" : "piece-conf high";
        }
        
        container.style.display = "block";
    }
}

async function loadPreviousValidationResults() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const prog = await pywebview.api.get_validation_progress();
        if (prog.results) {
            renderValidationResults(prog.results);
            setStepStatus("step-4", "done", "Validado ✓");
        }
    } catch(e) {
        console.warn("loadPreviousValidationResults error:", e);
    }
}


async function loadScopeInfo() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const info = await pywebview.api.get_training_scope_info();
        if (info.status === "ok") {
            const s = info.single_set;
            const a = info.all_sets;
            const elSingle = document.getElementById("scope-stats-single");
            const elAll = document.getElementById("scope-stats-all");
            if (elSingle) elSingle.textContent = `${s.yolo_geometries} geom · ${s.dino_pairs} pares (ref,color) · ~${Math.round(s.total_eta_min/60)}h`;
            if (elAll) elAll.textContent = `${a.yolo_geometries} geom · ${a.dino_pairs} pares · ~${Math.round(a.total_eta_min/60)}h`;
        }
    } catch(e) { console.warn("loadScopeInfo:", e); }
}

async function pollIndexingProgress() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const prog = await pywebview.api.get_indexing_progress();
        const bar = document.getElementById("dino-mini-bar");
        const label = document.getElementById("dino-progress-label");
        const pct = document.getElementById("dino-progress-pct");
        const fileEl = document.getElementById("dino-progress-file");
        const wrap = document.getElementById("dino-progress-wrap");
        if (bar) bar.style.width = `${prog.pct}%`;
        if (label) label.textContent = `${prog.current} / ${prog.total} embeddings`;
        if (pct) pct.textContent = `${prog.pct}%`;
        if (fileEl && prog.current_file) fileEl.textContent = prog.current_file;
        if (wrap) wrap.style.display = "block";
        if (prog.done) {
            if (bar) bar.style.width = "100%";
            if (pct) pct.textContent = "100% ✓";
            if (fileEl) fileEl.textContent = "Indexación completada";
            setStepStatus("step-2", "done", "Indexado ✓");
            clearInterval(window._dinoProgressInterval);
            window._dinoProgressInterval = null;
            refreshEmbeddingCount();
        }
        if (prog.error) {
            setStepStatus("step-2", "error", "Error");
            appendLog(`> Error indexacion: ${prog.error}`);
            clearInterval(window._dinoProgressInterval);
            window._dinoProgressInterval = null;
        }
    } catch(e) { console.warn("pollIndexingProgress:", e); }
}

async function pollTrainingStatus() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;

    try {
        const res = await pywebview.api.get_training_status();
        if (res.status !== "ok" || !res.run) return;

        const run = res.run;

        // Badge del header
        const badge = document.getElementById("train-status-badge");
        badge.innerText = run.status === "generating" ? "Generando..." :
                          run.status === "running" ? "Entrenando..." :
                          run.status === "completed" ? "Completado ✓" :
                          run.status === "failed"    ? "Error ✗" : run.status;
        badge.className = (run.status === "generating" || run.status === "running" || run.status === "completed")
                          ? "piece-conf high" : "piece-conf low";

        // Paso 1 status
        if (run.status === "generating") {
            setStepStatus("step-1", "running", `Generando dataset... (${run.generation_current}/${run.generation_total})`);
            const yBar = document.getElementById("yolo-mini-bar");
            const yLabel = document.getElementById("yolo-progress-label");
            const yMetrics = document.getElementById("yolo-progress-metrics");
            const yWrap = document.getElementById("yolo-progress-wrap");
            if (yBar) yBar.style.width = `${run.generation_pct}%`;
            if (yLabel) yLabel.textContent = `Generando dataset: ${run.generation_pct}%`;
            if (yMetrics) yMetrics.textContent = `${run.generation_current} / ${run.generation_total} imágenes`;
            if (yWrap) yWrap.style.display = "block";
            if (!trainingPollInterval)
                trainingPollInterval = setInterval(pollTrainingStatus, 30000);
        } else if (run.status === "running") {
            setStepStatus("step-1", "running", `Época ${run.current_epoch}/${run.epochs}`);
            // Update YOLO mini progress bar
            const yoloPct = run.epochs > 0 ? (run.current_epoch / run.epochs) * 100 : 0;
            const yBar = document.getElementById("yolo-mini-bar");
            const yLabel = document.getElementById("yolo-progress-label");
            const yMetrics = document.getElementById("yolo-progress-metrics");
            const yWrap = document.getElementById("yolo-progress-wrap");
            if (yBar) yBar.style.width = `${yoloPct.toFixed(1)}%`;
            if (yLabel) yLabel.textContent = `Época ${run.current_epoch} / ${run.epochs}`;
            if (yMetrics && run.loss !== undefined) {
                const loss = run.loss ? run.loss.toFixed(4) : "—";
                const map50 = run.map50 ? run.map50.toFixed(3) : "—";
                yMetrics.textContent = `Loss: ${loss} · mAP50: ${map50}`;
            }
            if (yWrap) yWrap.style.display = "block";
            if (!trainingPollInterval)
                trainingPollInterval = setInterval(pollTrainingStatus, 30000);
        } else if (run.status === "completed") {
            setStepStatus("step-1", "done", "Completado ✓");
            if (trainingPollInterval) { clearInterval(trainingPollInterval); trainingPollInterval = null; }
                const yBarFinal = document.getElementById("yolo-mini-bar");
                if (yBarFinal) yBarFinal.style.width = "100%";
                const yLabelFinal = document.getElementById("yolo-progress-label");
                if (yLabelFinal) yLabelFinal.textContent = "Completado ✓";
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
        // Actualizamos también el step-3 a done si hay embeddings indexados
        setStepStatus("step-3", count > 0 ? "done" : "pending",
            count > 0 ? "Actualizado ✓" : "Pendiente");
    } catch (e) { console.error("Embedding count error:", e); }
}

// ── Funciones de Control del Modal y Simulación DINO-FOV ───────────────────
async function openDinoFovModal() {
    const select = document.getElementById("dino-part-ref-select");
    select.innerHTML = '<option value="">Cargando piezas...</option>';
    
    // Mostrar modal y overlay
    document.getElementById("dino-fov-overlay").style.display = "block";
    document.getElementById("dino-fov-modal").style.display = "block";
    
    // Refrescar recuento en segundo plano
    refreshEmbeddingCount();
    
    try {
        const set_id = currentSetId || "75078-1";
        let parts = [];
        if (typeof pywebview !== "undefined" && pywebview.api) {
            const res = await pywebview.api.get_set_inventory(set_id);
            if (res.status === "success") {
                parts = res.parts;
            }
        }
        
        if (parts.length === 0) {
            select.innerHTML = '<option value="">Error cargando inventario</option>';
            return;
        }
        
        // Poblar el dropdown con las piezas del set actual
        select.innerHTML = '<option value="">Seleccione una pieza...</option>' + 
            parts.map(p => `<option value="${p.ref}" data-color="${p.color_hex || 'A0A5A9'}">Pieza ${p.ref} (${p.color_name || 'Gray'})</option>`).join("");
            
    } catch (e) {
        console.error("Error al abrir modal DINO-FOV:", e);
        select.innerHTML = '<option value="">Error cargando inventario</option>';
    }
}

function closeDinoFovModal() {
    document.getElementById("dino-fov-overlay").style.display = "none";
    document.getElementById("dino-fov-modal").style.display = "none";
}

async function startDinoFovSimulation() {
    const select = document.getElementById("dino-part-ref-select");
    const part_ref = select.value;
    if (!part_ref) {
        alert("Por favor, seleccione una pieza de referencia.");
        return;
    }
    const option = select.options[select.selectedIndex];
    const color_hex = option.getAttribute("data-color") || "A0A5A9";
    const num_rotations = parseInt(document.getElementById("dino-rotations").value) || 12;
    const num_pieces = parseInt(document.getElementById("dino-pieces-count").value) || 30;
    
    closeDinoFovModal();
    
    // Actualizar UX del Step 3
    setStepStatus("step-3", "running", "▶ Simulando FOV...");
    appendLog(`\n> Iniciando simulación DINO-FOV para pieza ${part_ref} (Color #${color_hex})...`);
    appendLog(`> Configuración: ${num_rotations} rotaciones, ${num_pieces} instancias en total.`);
    
    // Mostrar barra de progreso
    const wrap = document.getElementById("dino-fov-progress-wrap");
    if (wrap) {
        wrap.style.display = "block";
        document.getElementById("dino-fov-mini-bar").style.width = "0%";
        document.getElementById("dino-fov-progress-label").textContent = "Fase: Iniciando...";
        document.getElementById("dino-fov-progress-pct").textContent = "0%";
        document.getElementById("dino-fov-progress-file").textContent = "Preparando escena en Blender...";
    }
    
    const btnStop = document.getElementById("btn-stop-dino-fov");
    if (btnStop) btnStop.style.display = "inline-block";
    
    try {
        if (typeof pywebview !== "undefined" && pywebview.api) {
            const res = await pywebview.api.start_dinov2_fov_simulation(part_ref, num_rotations, num_pieces, color_hex, true);
            appendLog(`> ${res.message}`);
            
            // Iniciar polling
            if (window._dinoFovProgressInterval) clearInterval(window._dinoFovProgressInterval);
            window._dinoFovProgressInterval = setInterval(pollDinoFovProgress, 1500);
        } else {
            appendLog("> [DEMO] API no disponible en navegador.");
            setStepStatus("step-3", "error", "Sin API");
        }
    } catch (e) {
        appendLog(`> ERROR: ${e.message}`);
        setStepStatus("step-3", "error", "Error");
    }
}

async function pollDinoFovProgress() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const prog = await pywebview.api.get_dinov2_fov_progress();
        const bar = document.getElementById("dino-fov-mini-bar");
        const label = document.getElementById("dino-fov-progress-label");
        const pctEl = document.getElementById("dino-fov-progress-pct");
        const fileEl = document.getElementById("dino-fov-progress-file");
        const btnStop = document.getElementById("btn-stop-dino-fov");
        
        if (prog.active) {
            let pct = 0;
            if (prog.total > 0) {
                pct = Math.round((prog.current / prog.total) * 100);
            } else if (prog.phase === "indexing") {
                pct = prog.current;
            }
            pct = Math.min(100, Math.max(0, pct));
            
            let phaseText = "Renderizando...";
            if (prog.phase === "indexing") {
                phaseText = "Vectorizando DINOv2...";
            }
            
            setStepStatus("step-3", "running", `${phaseText} ${pct}%`);
            if (bar) bar.style.width = `${pct}%`;
            if (label) label.textContent = `Fase: ${phaseText}`;
            if (pctEl) pctEl.textContent = `${pct}%`;
            if (fileEl && prog.current_file) fileEl.textContent = prog.current_file;
            if (btnStop) btnStop.style.display = "inline-block";
        }
        
        if (prog.done && !prog.active) {
            if (bar) bar.style.width = "100%";
            if (pctEl) pctEl.textContent = "100% ✓";
            if (fileEl) fileEl.textContent = "Vectorización y guardado completados en BD.";
            setStepStatus("step-3", "done", "Actualizado ✓");
            if (btnStop) btnStop.style.display = "none";
            
            if (window._dinoFovProgressInterval) {
                clearInterval(window._dinoFovProgressInterval);
                window._dinoFovProgressInterval = null;
            }
            appendLog(`> Embeddings dinámicos generados correctamente.`);
            refreshEmbeddingCount();
        }
        
        if (prog.error) {
            setStepStatus("step-3", "error", "Error");
            appendLog(`> ERROR en simulación FOV DINOv2: ${prog.error}`);
            if (btnStop) btnStop.style.display = "none";
            if (window._dinoFovProgressInterval) {
                clearInterval(window._dinoFovProgressInterval);
                window._dinoFovProgressInterval = null;
            }
        }
    } catch(e) {
        console.warn("pollDinoFovProgress:", e);
    }
}

async function stopDinoFovSimulation() {
    if (!confirm("¿Detener la simulación DINO-FOV en curso? El progreso parcial se perderá.")) return;
    const btnStop = document.getElementById("btn-stop-dino-fov");
    if (btnStop) {
        btnStop.disabled = true;
        btnStop.textContent = "⏳ Deteniendo...";
    }
    try {
        if (typeof pywebview !== "undefined" && pywebview.api) {
            const res = await pywebview.api.stop_dinov2_fov_simulation();
            appendLog(`> ⏹ STOP SIMULACIÓN: ${res.message}`);
            setStepStatus("step-3", "error", "⏹ Cancelado");
            if (window._dinoFovProgressInterval) {
                clearInterval(window._dinoFovProgressInterval);
                window._dinoFovProgressInterval = null;
            }
        }
    } catch (e) {
        appendLog(`> Error deteniendo simulación DINO-FOV: ${e.message}`);
    } finally {
        if (btnStop) {
            btnStop.style.display = "none";
            btnStop.disabled = false;
            btnStop.textContent = "⏹ Stop";
        }
    }
}

async function pollDinoFovProgressOnLoad() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const prog = await pywebview.api.get_dinov2_fov_progress();
        if (prog.active) {
            const wrap = document.getElementById("dino-fov-progress-wrap");
            if (wrap) wrap.style.display = "block";
            const btnStop = document.getElementById("btn-stop-dino-fov");
            if (btnStop) btnStop.style.display = "inline-block";
            
            if (window._dinoFovProgressInterval) clearInterval(window._dinoFovProgressInterval);
            window._dinoFovProgressInterval = setInterval(pollDinoFovProgress, 1500);
        }
    } catch(e) {
        console.error("Error polling DINO-FOV progress on load:", e);
    }
}

// =============================================================
// 12b. Evaluación Post-Entrenamiento (Opción C)
// =============================================================

async function loadEvalResults() {
    if (typeof pywebview === "undefined" || !pywebview.api) return;
    try {
        const res = await pywebview.api.get_eval_results();
        const noData   = document.getElementById("eval-no-data");
        const metrics  = document.getElementById("eval-metrics-wrap");

        if (res.status !== "ok") {
            if (noData)  noData.style.display  = "block";
            if (metrics) metrics.style.display = "none";
            if (noData && res.message) {
                noData.innerHTML = `<span style="font-size:2rem; display:block; margin-bottom:8px;">📊</span>${res.message}`;
            }
            return;
        }

        const e = res.eval;
        if (noData)  noData.style.display  = "none";
        if (metrics) metrics.style.display = "block";

        // Métricas principales
        const fmt = v => (v * 100).toFixed(1) + "%";
        const el = id => document.getElementById(id);
        if (el("eval-precision")) el("eval-precision").innerText = fmt(e.precision);
        if (el("eval-recall"))    el("eval-recall").innerText    = fmt(e.recall);
        if (el("eval-f1"))        el("eval-f1").innerText        = fmt(e.f1);
        if (el("eval-map50"))     el("eval-map50").innerText     = fmt(e.map50);
        if (el("eval-tp"))        el("eval-tp").innerText        = e.tp;
        if (el("eval-fp"))        el("eval-fp").innerText        = e.fp;
        if (el("eval-fn"))        el("eval-fn").innerText        = e.fn;
        if (el("eval-gt"))        el("eval-gt").innerText        = e.gt_total;
        if (el("eval-dets"))      el("eval-dets").innerText      = e.detections_total;

        // Barra mAP50
        const map50pct = Math.round(e.map50 * 100);
        if (el("eval-map50-pct")) el("eval-map50-pct").innerText = map50pct + "%";
        requestAnimationFrame(() => {
            if (el("eval-map50-bar")) el("eval-map50-bar").style.width = map50pct + "%";
        });

        // Tabla per-piece
        const table = el("eval-per-piece-table");
        if (table && e.per_piece_stats && e.per_piece_stats.length > 0) {
            const sorted = [...e.per_piece_stats].sort((a, b) => b.total_gt - a.total_gt).slice(0, 15);
            table.innerHTML = sorted.map(p => {
                const recPct = Math.round(p.recall * 100);
                const color = recPct >= 70 ? "#10b981" : recPct >= 40 ? "#f59e0b" : "#ef4444";
                return `
                    <div style="display:flex; align-items:center; gap:8px; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="font-family:monospace; color:var(--accent); min-width:80px; font-size:0.78rem;">${p.ref}</span>
                        <div style="flex:1; background:rgba(255,255,255,0.06); border-radius:10px; height:6px; overflow:hidden;">
                            <div style="width:${recPct}%; height:100%; background:${color}; border-radius:10px; transition:width 0.5s;"></div>
                        </div>
                        <span style="color:${color}; font-weight:bold; min-width:38px; text-align:right; font-size:0.78rem;">${recPct}%</span>
                        <span style="color:var(--text-secondary); font-size:0.7rem; min-width:50px;">TP:${p.tp} FN:${p.fn}</span>
                    </div>`;
            }).join("");
        }
    } catch (e) {
        console.error("loadEvalResults error:", e);
    }
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
let currentPieceIndex = -1; // -1 means show the full frame, 0+ means show specific piece
let currentFrameImg = null; // To hold the loaded image for cropping
let historyCanvas, historyCtx;
let historyPollingTimer = null;

function initHistoryView() {
    historyCanvas = document.getElementById("history-canvas");
    if (!historyCanvas) return;
    historyCtx = historyCanvas.getContext("2d");
    historyCanvas.width = CANVAS_W;
    historyCanvas.height = CANVAS_H;

    clearHistoryCanvas();

    document.getElementById("btn-history-prev").addEventListener("click", () => navigateHistoryPiece(-1));
    document.getElementById("btn-history-next").addEventListener("click", () => navigateHistoryPiece(1));
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
    grad.addColorStop(0,   "#1b303f");
    grad.addColorStop(0.5, "#254154");
    grad.addColorStop(1,   "#1b303f");
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
            currentFrameImg = img;
            currentPieceIndex = detectData.detections && detectData.detections.length > 0 ? 0 : -1;
            
            historyBboxes = (detectData.detections || []).map(d => {
                const [xc, yc, wn, hn] = d.bbox;
                const x1 = (xc - wn/2) * CANVAS_W;
                const y1 = (yc - hn/2) * CANVAS_H;
                const x2 = (xc + wn/2) * CANVAS_W;
                const y2 = (yc + hn/2) * CANVAS_H;
                return {
                    x1:    x1,
                    y1:    y1,
                    x2:    x2,
                    y2:    y2,
                    label: d.name || "lego_piece",
                    conf:  d.confidence || 0,
                    bbox:  d.bbox
                };
            });
            renderCurrentPiece();
        };
    } catch (e) {
        console.error("Error al cargar imagen del historial:", e);
        clearHistoryCanvas();
        historyCtx.fillStyle = "#ef4444";
        historyCtx.fillText("Error al procesar el frame del historial.", CANVAS_W / 2, CANVAS_H / 2);
    }
}

async function renderCurrentPiece() {
    if (!currentFrameImg) return;
    
    updateHistoryNavigationControls();
    clearHistoryCanvas();

    if (currentPieceIndex === -1 || historyBboxes.length === 0) {
        // Draw full frame
        historyCtx.drawImage(currentFrameImg, 0, 0, CANVAS_W, CANVAS_H);
        const pageNumEl = document.getElementById("history-page-num");
        if (pageNumEl) pageNumEl.innerText = "Sin piezas detectadas";
        return;
    }

    const bb = historyBboxes[currentPieceIndex];
    const pageNumEl = document.getElementById("history-page-num");
    if (pageNumEl) {
        pageNumEl.innerText = `Pieza ${currentPieceIndex + 1} de ${historyBboxes.length}`;
    }
    
    const cw = bb.x2 - bb.x1;
    const ch = bb.y2 - bb.y1;
    
    if (cw < 4 || ch < 4) {
        historyCtx.drawImage(currentFrameImg, 0, 0, CANVAS_W, CANVAS_H);
        return;
    }

    // Zoom piece to fit canvas
    const scale = Math.min(CANVAS_W / cw, CANVAS_H / ch) * 0.9;
    const scaledW = cw * scale;
    const scaledH = ch * scale;
    const dx = (CANVAS_W - scaledW) / 2;
    const dy = (CANVAS_H - scaledH) / 2;

    historyCtx.imageSmoothingEnabled = false;
    historyCtx.drawImage(currentFrameImg, bb.x1, bb.y1, cw, ch, dx, dy, scaledW, scaledH);
    
    historyCtx.strokeStyle = "var(--accent, #3b82f6)";
    historyCtx.lineWidth = 4;
    historyCtx.strokeRect(dx, dy, scaledW, scaledH);

    historyCtx.fillStyle = "rgba(255,255,255,0.8)";
    historyCtx.font = "14px Outfit, sans-serif";
    historyCtx.textAlign = "left";
    historyCtx.fillText("Clasificando...", dx + 5, dy + 20);

    // Call DINOv2
    const offscreen = document.createElement("canvas");
    offscreen.width  = cw;
    offscreen.height = ch;
    const octx = offscreen.getContext("2d");
    octx.drawImage(currentFrameImg, bb.x1, bb.y1, cw, ch, 0, 0, cw, ch);
    const cropDataUrl = offscreen.toDataURL("image/png");

    const frameDataUrl = currentFrameImg.src; // Wait, we might need b64, but classifyWithDINOv2 uses frameB64
    // We can recreate it from canvas
    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = CANVAS_W;
    frameCanvas.height = CANVAS_H;
    frameCanvas.getContext("2d").drawImage(currentFrameImg, 0, 0, CANVAS_W, CANVAS_H);
    const frameB64 = frameCanvas.toDataURL("image/png").replace(/^data:image\/\w+;base64,/, "");
    const filename = historyImages[currentHistoryIndex] || null;

    try {
        const data = await classifyWithDINOv2(frameB64, bb, filename);
        // Redraw
        clearHistoryCanvas();
        historyCtx.drawImage(currentFrameImg, bb.x1, bb.y1, cw, ch, dx, dy, scaledW, scaledH);
        historyCtx.strokeRect(dx, dy, scaledW, scaledH);
        
        let overlayText = "Sin coincidencia";
        let colorText = "#ef4444";
        
        if (data && data.best_match) {
            const best = data.best_match;
            const pct = Math.round(best.score * 100);
            overlayText = `${pct}% - ${best.part_name || best.part_ref}`;
            colorText = "#10b981"; // green
        }

        historyCtx.fillStyle = "rgba(0,0,0,0.7)";
        historyCtx.fillRect(dx, dy, scaledW, 30);
        historyCtx.fillStyle = colorText;
        historyCtx.font = "bold 16px Outfit, sans-serif";
        historyCtx.fillText(overlayText, dx + 5, dy + 20);
    } catch (err) {
        console.error("DINOv2 error in history:", err);
    }
}

function updateHistoryNavigationControls() {
    const btnPrev = document.getElementById("btn-history-prev");
    const btnNext = document.getElementById("btn-history-next");
    if (btnPrev) btnPrev.disabled = (currentPieceIndex <= 0);
    if (btnNext) btnNext.disabled = (historyBboxes.length === 0 || currentPieceIndex >= historyBboxes.length - 1);
}

function navigateHistoryPiece(direction) {
    if (historyBboxes.length === 0) return;
    const nextIndex = currentPieceIndex + direction;
    if (nextIndex >= 0 && nextIndex < historyBboxes.length) {
        currentPieceIndex = nextIndex;
        renderCurrentPiece();
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
    
    // Scale factor: standard unit stud is 8mm. 8mm * 3.2 px/mm = ~25.6 px.
    const STUD_PX = 26;
    
    // Check if it's a minifigure (approx 1.6 x 4 cm -> 2 studs x 5 studs)
    if (ref.startsWith("sw") || ref.startsWith("fig") || n.includes("minifig")) {
        return { w: 2 * STUD_PX, h: 5 * STUD_PX };
    }
    
    // Check for standard Brick sizes in name (e.g. "Brick 2 x 4", "Plate 1 x 2")
    const match = n.match(/(?:brick|plate|tile|slope|wedge)\s+(\d+)\s*x\s*(\d+)/i);
    if (match) {
        const x = parseInt(match[1]);
        const y = parseInt(match[2]);
        return { w: Math.max(STUD_PX, x * STUD_PX), h: Math.max(STUD_PX, y * STUD_PX) };
    }
    
    // Fallbacks based on common LDraw reference names if description is simple
    if (ref === "3001") return { w: 4 * STUD_PX, h: 2 * STUD_PX }; // Brick 2x4
    if (ref === "3002") return { w: 3 * STUD_PX, h: 2 * STUD_PX }; // Brick 2x3
    if (ref === "3003") return { w: 2 * STUD_PX, h: 2 * STUD_PX }; // Brick 2x2
    if (ref === "3004") return { w: 2 * STUD_PX, h: 1 * STUD_PX }; // Brick 1x2
    if (ref === "3005") return { w: 1 * STUD_PX, h: 1 * STUD_PX }; // Brick 1x1
    if (ref === "3010") return { w: 4 * STUD_PX, h: 1 * STUD_PX }; // Brick 1x4
    if (ref === "3020") return { w: 4 * STUD_PX, h: 2 * STUD_PX }; // Plate 2x4
    if (ref === "3021") return { w: 3 * STUD_PX, h: 2 * STUD_PX }; // Plate 2x3
    if (ref === "3022") return { w: 2 * STUD_PX, h: 2 * STUD_PX }; // Plate 2x2
    if (ref === "3023") return { w: 2 * STUD_PX, h: 1 * STUD_PX }; // Plate 1x2
    if (ref === "3024") return { w: 1 * STUD_PX, h: 1 * STUD_PX }; // Plate 1x1
    if (ref === "3070" || ref === "3070b") return { w: 1 * STUD_PX, h: 1 * STUD_PX }; // Tile 1x1
    if (ref === "3069" || ref === "3069b") return { w: 2 * STUD_PX, h: 1 * STUD_PX }; // Tile 1x2
    if (ref === "3068" || ref === "3068b") return { w: 2 * STUD_PX, h: 2 * STUD_PX }; // Tile 2x2
    if (ref === "2420") return { w: 2 * STUD_PX, h: 2 * STUD_PX }; // Plate 2x2 Corner
    if (ref === "3710") return { w: 3 * STUD_PX, h: 1 * STUD_PX }; // Plate 1x3
    if (ref === "3666") return { w: 6 * STUD_PX, h: 1 * STUD_PX }; // Plate 1x6
    if (ref === "3795") return { w: 6 * STUD_PX, h: 2 * STUD_PX }; // Plate 2x6
    if (ref === "4073") return { w: 1 * STUD_PX, h: 1 * STUD_PX }; // Plate Round 1x1
    if (ref === "6141") return { w: 1 * STUD_PX, h: 1 * STUD_PX }; // Plate Round 1x1
    if (ref === "15573") return { w: 2 * STUD_PX, h: 1 * STUD_PX }; // Plate Modified Jumper 1x2
    if (ref === "18674") return { w: 2 * STUD_PX, h: 2 * STUD_PX }; // Plate Round 2x2
    if (ref === "32000") return { w: 2 * STUD_PX, h: 1 * STUD_PX }; // Technic Brick 1x2
    if (ref === "2780") return { w: 20, h: 8 };   // Technic Pin
    if (ref === "3673") return { w: 20, h: 8 };   // Technic Pin
    if (ref === "4274") return { w: 14, h: 8 };   // Technic Pin
    
    // Bar / rod piezas largas y finas
    if (ref === "30374" || n.includes("bar 4l") || n.includes("lightsaber blade") || n.includes("wand")) {
        return { w: 10, h: 56 };
    }
    // Weapon hilt / cilíndrico pequeño
    if (ref === "64567" || n.includes("lightsaber hilt") || n.includes("hilt")) {
        return { w: 22, h: 22 };
    }
    // Minifig weapon genérico
    if (n.includes("weapon") || n.includes("blaster") || n.includes("gun")) {
        return { w: 18, h: 14 };
    }

    // Default fallback
    return { w: 45, h: 45 };
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
                        colorCode: "15",
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
                        colorCode: p.color_code || p.colorCode || "15",
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
                setSimulationActive = false;
                setSimulationImage = null;
                setSimulationMeta = null;
                setSequentialImages = [];
                updateSequentialCameraList();
                const statusWrap = document.getElementById("set-simulation-status-wrap");
                if (statusWrap) statusWrap.style.display = "none";

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
        
        // Reset set simulation variables
        setSimulationActive = false;
        setSimulationImage = null;
        setSimulationMeta = null;
        setSequentialImages = [];
        updateSequentialCameraList();
        const statusWrap = document.getElementById("set-simulation-status-wrap");
        if (statusWrap) statusWrap.style.display = "none";

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




// ══════════════════════════════════════════════════════════════════════════════
// EXPORTAR EXCEL DE VALIDACION DE POSICIONES ESTABLES
// ══════════════════════════════════════════════════════════════════════════════

async function exportValidationExcel() {
    const btn = document.getElementById("btn-export-excel");
    const msgEl = document.getElementById("excel-status-msg");
    if (btn) {
        btn.disabled = true;
        btn.textContent = "⏳ Generando Excel...";
    }
    if (msgEl) {
        msgEl.style.display = "";
        msgEl.textContent = "Renderizando poses con Blender y generando Excel...";
        msgEl.style.color = "#f39c12";
    }
    appendLog("> Generando Excel de validacion de posiciones estables...");
    try {
        if (typeof pywebview === "undefined" || !pywebview.api) {
            throw new Error("pywebview API no disponible");
        }
        const result = await pywebview.api.generate_validation_excel("75078-1");
        if (result.status === "success") {
            appendLog("> Excel generado: " + result.filename + " (" + result.size_kb + " KB)");
            if (msgEl) {
                msgEl.textContent = "Excel generado: " + result.filename + " (" + result.size_kb + " KB)";
                msgEl.style.color = "#7DF9AA";
            }
            // Mostrar boton de abrir
            const btnOpen = document.getElementById("btn-open-excel");
            if (btnOpen) btnOpen.style.display = "";
        } else {
            appendLog("> Error generando Excel: " + result.message);
            if (msgEl) {
                msgEl.textContent = "Error: " + result.message;
                msgEl.style.color = "#e74c3c";
            }
        }
    } catch(e) {
        appendLog("> Error exportando Excel: " + e.message);
        if (msgEl) {
            msgEl.textContent = "Error: " + e.message;
            msgEl.style.color = "#e74c3c";
        }
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = "📊 Exportar Excel";
        }
    }
}

async function openValidationExcel() {
    try {
        if (typeof pywebview === "undefined" || !pywebview.api) return;
        const result = await pywebview.api.open_validation_excel("75078-1");
        if (result.status === "success") {
            appendLog("> Abriendo " + result.message);
        } else {
            appendLog("> " + result.message);
        }
    } catch(e) {
        appendLog("> Error abriendo Excel: " + e.message);
    }
}

// =============================================================
// INFERENCIA-TEST: Generar Render + Selector de Renders
// =============================================================
function initInferenceTestPanel() {
    const btnGenerateRender = document.getElementById("btn-generate-render");
    const btnRefreshRenders = document.getElementById("btn-refresh-renders");
    const selectRender = document.getElementById("select-inference-render");
    const previewImg = document.getElementById("render-preview-img");
    const previewPlaceholder = document.getElementById("render-preview-placeholder");
    const metaJson = document.getElementById("render-meta-json");
    const metaSummary = document.getElementById("render-meta-summary");
    const statusDiv = document.getElementById("render-gen-status");

    // Renders disponibles (cargados del backend)
    let availableRenders = [];

    // Cargar lista de renders al inicio
    loadAvailableRenders();

    // Boton: Generar nuevo render con Blender
    if (btnGenerateRender) {
        btnGenerateRender.addEventListener("click", async () => {
            btnGenerateRender.disabled = true;
            btnGenerateRender.innerText = "⏳ Generando render en Blender...";
            showRenderStatus("info", "⏳ Generando render en Blender... esto puede tardar 1-2 min.");
            try {
                const pf = maxPiecesInField || 30;
                const sid = currentSetId || "75078-1";
                const isRollingEl = document.getElementById("live-is-rolling");
                const isRolling = isRollingEl ? isRollingEl.checked : true;
                
                let res;
                if (typeof pywebview === "undefined" || !pywebview.api) {
                    console.log("[InferenceTest] Usando fallback REST API para generar render: pieces_in_field=" + pf + " set_id=" + sid + " is_rolling=" + isRolling);
                    const response = await fetch(`${API_BASE}/generate_inference_render?pieces_in_field=${pf}&set_id=${sid}&is_rolling=${isRolling}`, {
                        method: "POST"
                    });
                    res = await response.json();
                    if (!response.ok) {
                        throw new Error(res.detail || "Error en el servidor API");
                    }
                } else {
                    console.log("[InferenceTest] Generando render: pieces_in_field=" + pf + " set_id=" + sid + " is_rolling=" + isRolling);
                    res = await pywebview.api.generate_inference_test_render(pf, sid, isRolling);
                }
                
                if (res.status === "success") {
                    showRenderStatus("ok", "✅ " + res.message);
                    await loadAvailableRenders();
                    // Auto-seleccionar el render recien generado
                    const filename = res.image_url.split("/").pop();
                    if (selectRender) {
                        for (let opt of selectRender.options) {
                            if (opt.value === filename) { selectRender.value = filename; break; }
                        }
                    }
                    displaySelectedRender(res.image_url, res.metadata);
                } else {
                    showRenderStatus("error", "❌ Error: " + res.message);
                }
            } catch(e) {
                showRenderStatus("error", "❌ Excepcion: " + e.message);
            }
            btnGenerateRender.disabled = false;
            btnGenerateRender.innerText = "🎬 Generar Render 2D (Blender)";
        });
    }

    // Boton: Actualizar lista de renders
    if (btnRefreshRenders) {
        btnRefreshRenders.addEventListener("click", () => loadAvailableRenders());
    }

    const btnDeleteRender = document.getElementById("btn-delete-render");

    // Selector: cambio de render
    if (selectRender) {
        selectRender.addEventListener("change", () => {
            const selectedFilename = selectRender.value;
            if (!selectedFilename) {
                if (btnDeleteRender) btnDeleteRender.style.display = "none";
                clearRenderPreview();
                return;
            }
            if (btnDeleteRender) btnDeleteRender.style.display = "inline-flex";
            const renderData = availableRenders.find(r => r.filename === selectedFilename);
            if (renderData) {
                displaySelectedRender(renderData.image_url, renderData.metadata);
                // Pre-cargar en setSimulationImage para cuando se ejecute inferencia
                const img = new Image();
                img.onload = () => {
                    setSimulationImage = img;
                    setSimulationMeta = renderData.metadata;
                    console.log("[InferenceTest] Render pre-cargado:", selectedFilename);
                };
                img.src = renderData.image_url;
            }
        });
    }

    // Boton Borrar Render
    if (btnDeleteRender) {
        btnDeleteRender.addEventListener("click", async () => {
            const selectedFilename = selectRender ? selectRender.value : null;
            if (!selectedFilename) return;

            const confirmDelete = confirm("¿Estás seguro de que deseas borrar este render de prueba permanentemente?");
            if (!confirmDelete) return;

            btnDeleteRender.disabled = true;
            showRenderStatus("info", "⏳ Eliminando render...");

            try {
                let res;
                if (typeof pywebview === "undefined" || !pywebview.api) {
                    const response = await fetch(`${API_BASE}/inference-render/${selectedFilename}`, {
                        method: "DELETE"
                    });
                    res = await response.json();
                    if (!response.ok) {
                        throw new Error(res.detail || "Error en el servidor API");
                    }
                } else {
                    res = await pywebview.api.delete_inference_render(selectedFilename);
                }

                if (res.status === "success") {
                    showRenderStatus("ok", "✅ Render eliminado con éxito.");
                    
                    // Reset variables de simulación de render
                    setSimulationImage = null;
                    setSimulationMeta = null;
                    
                    // Limpiar preview y recargar dropdown
                    clearRenderPreview();
                    if (selectRender) selectRender.value = "";
                    if (btnDeleteRender) btnDeleteRender.style.display = "none";
                    await loadAvailableRenders();
                } else {
                    showRenderStatus("error", "❌ Error: " + res.message);
                }
            } catch (e) {
                showRenderStatus("error", "❌ Excepción al eliminar: " + e.message);
            } finally {
                btnDeleteRender.disabled = false;
            }
        });
    }

    // Click en imagen: abrir a pantalla completa
    if (previewImg) {
        previewImg.addEventListener("click", () => {
            if (previewImg.src) window.open(previewImg.src, "_blank");
        });
    }

    async function loadAvailableRenders() {
        try {
            const sid = currentSetId || "75078-1";
            let renders = [];
            
            if (typeof pywebview === "undefined" || !pywebview.api) {
                // Si no hay pywebview (modo navegador), podemos consultar el backend REST API si tuviera un list_renders (no lo tiene implementado en api.py, pero list_inference_renders de app.py se ejecuta en modo pywebview)
                // Para mantener compatibilidad, llamamos a list_inference_renders si está el puente de pywebview
                console.warn("[InferenceTest] Modo navegador - la carga de renders depende de la API local.");
            } else {
                const res = await pywebview.api.list_inference_renders(sid);
                if (res.status === "success") {
                    renders = res.renders;
                }
            }
            
            availableRenders = renders;
            populateRenderSelector(renders);
            
            // Auto-seleccionar el primero si hay renders y nada seleccionado
            if (renders.length > 0 && selectRender && !selectRender.value) {
                selectRender.value = renders[0].filename;
                selectRender.dispatchEvent(new Event("change"));
            }
        } catch(e) {
            console.warn("[InferenceTest] Error cargando renders:", e);
        }
    }

    function populateRenderSelector(renders) {
        if (!selectRender) return;
        const current = selectRender.value;
        selectRender.innerHTML = '<option value="">&mdash; Sin renders generados &mdash;</option>';
        renders.forEach(r => {
            const opt = document.createElement("option");
            opt.value = r.filename;
            opt.textContent = r.label;
            selectRender.appendChild(opt);
        });
        
        // Mantener la selección anterior si es posible y sigue existiendo
        if (current && renders.some(r => r.filename === current)) {
            selectRender.value = current;
            if (btnDeleteRender) btnDeleteRender.style.display = "inline-flex";
        } else {
            selectRender.value = "";
            if (btnDeleteRender) btnDeleteRender.style.display = "none";
        }
    }

    function displaySelectedRender(imageUrl, metadata) {
        if (!previewImg || !previewPlaceholder) return;
        previewImg.src = imageUrl;
        previewImg.style.display = "block";
        previewPlaceholder.style.display = "none";

        // Mostrar metadata JSON
        if (metaJson) {
            // Mostrar version resumida (sin la lista completa de detections)
            const summary = Object.assign({}, metadata);
            if (summary.detections && summary.detections.length > 3) {
                summary.detections_preview = summary.detections.slice(0, 3);
                summary.detections = "[" + summary.detections.length + " piezas - ver JSON completo]";
            }
            metaJson.textContent = JSON.stringify(summary, null, 2);
        }
        if (metaSummary && metadata) {
            const pp = metadata.pieces_placed || 0;
            const st = metadata.set_id || "";
            const pf = metadata.pieces_in_field || 0;
            metaSummary.textContent = st + " | " + pp + " piezas colocadas | campo=" + pf;
        }
    }

    function clearRenderPreview() {
        if (previewImg) { previewImg.style.display = "none"; previewImg.src = ""; }
        if (previewPlaceholder) previewPlaceholder.style.display = "flex";
        if (metaJson) metaJson.textContent = "Sin render seleccionado.";
        if (metaSummary) metaSummary.textContent = "";
        if (btnDeleteRender) btnDeleteRender.style.display = "none";
    }

    function showRenderStatus(type, msg) {
        if (!statusDiv) return;
        statusDiv.style.display = "block";
        statusDiv.style.color = type === "error" ? "#ef4444" : type === "ok" ? "#7DF9AA" : "#38bdf8";
        statusDiv.textContent = msg;
    }
}

// =============================================================
// INFERENCIA-TEST: 1 pieza, 3 imágenes (Multicámara)
// =============================================================
function initMulticamPanel() {
    const selectSet = document.getElementById("select-multicam-set");
    const btnGenerate = document.getElementById("btn-multicam-generate");
    const btnInfer = document.getElementById("btn-multicam-infer");
    const btnPrev = document.getElementById("btn-multicam-prev");
    const btnNext = document.getElementById("btn-multicam-next");
    
    const indicator = document.getElementById("multicam-piece-indicator");
    const gtRef = document.getElementById("multicam-gt-ref");
    const gtName = document.getElementById("multicam-gt-name");
    const gtColorBadge = document.getElementById("multicam-gt-color-badge");
    const gtColorText = document.getElementById("multicam-gt-color-text");
    
    const consensusRef = document.getElementById("multicam-consensus-ref");
    const consensusName = document.getElementById("multicam-consensus-name");
    const consensusScore = document.getElementById("multicam-consensus-score");
    
    // Feeds
    const imgCenital = document.getElementById("multicam-img-cenital");
    const placeCenital = document.getElementById("multicam-placeholder-cenital");
    const infCenital = document.getElementById("multicam-inf-cenital");
    const blCenital = document.getElementById("multicam-bl-cenital");
    const refCenital = document.getElementById("multicam-ref-cenital");
    const scoreCenital = document.getElementById("multicam-score-cenital");
    
    const imgLatL = document.getElementById("multicam-img-lateral-l");
    const placeLatL = document.getElementById("multicam-placeholder-lateral-l");
    const infLatL = document.getElementById("multicam-inf-lateral-l");
    const blLatL = document.getElementById("multicam-bl-lateral-l");
    const refLatL = document.getElementById("multicam-ref-lateral-l");
    const scoreLatL = document.getElementById("multicam-score-lateral-l");
    
    const imgLatR = document.getElementById("multicam-img-lateral-r");
    const placeLatR = document.getElementById("multicam-placeholder-lateral-r");
    const infLatR = document.getElementById("multicam-inf-lateral-r");
    const blLatR = document.getElementById("multicam-bl-lateral-r");
    const refLatR = document.getElementById("multicam-ref-lateral-r");
    const scoreLatR = document.getElementById("multicam-score-lateral-r");
    
    // Footer / Tabla
    const meanAccuracy = document.getElementById("multicam-mean-accuracy");
    const worstPieces = document.getElementById("multicam-worst-pieces");
    const tbody = document.getElementById("multicam-inventory-tbody");
    
    let metadata = null;
    let inferenceResults = null;
    let currentIndex = 0;
    
    // Al cargar o cambiar de set, intentar cargar metadatos existentes
    if (selectSet) {
        selectSet.addEventListener("change", () => {
            loadExistingMetadata();
        });
    }
    
    async function loadExistingMetadata() {
        if (!selectSet) return;
        const setId = selectSet.value;
        // Reset states
        metadata = null;
        inferenceResults = null;
        currentIndex = 0;
        updateUI();
        
        try {
            // Intentamos cargar el archivo multicam_metadata.json desde el backend
            const response = await fetch(`${API_BASE}/renders/multicam/multicam_metadata.json?t=${Date.now()}`);
            if (response.ok) {
                const data = await response.json();
                // Verificar que corresponda al set seleccionado
                if (data.set_id === setId) {
                    metadata = data;
                    currentIndex = 0;
                    console.log("[Multicam] Cargados renders existentes para el set", setId);
                }
            }
        } catch (e) {
            console.log("[Multicam] No hay renders existentes para el set", setId, e);
        }
        
        updateUI();
    }
    
    function updateUI() {
        const hasRenders = metadata && metadata.renders && metadata.renders.length > 0;
        
        // Botones de navegacion
        if (btnPrev) btnPrev.disabled = !hasRenders || currentIndex === 0;
        if (btnNext) btnNext.disabled = !hasRenders || currentIndex >= (metadata ? metadata.renders.length - 1 : 0);
        if (btnInfer) btnInfer.disabled = !hasRenders;
        
        if (!hasRenders) {
            if (indicator) indicator.textContent = "Sin renders";
            if (gtRef) gtRef.textContent = "—";
            if (gtName) gtName.textContent = "—";
            if (gtColorBadge) gtColorBadge.style.backgroundColor = "transparent";
            if (gtColorText) gtColorText.textContent = "—";
            
            if (consensusRef) consensusRef.textContent = "—";
            if (consensusName) consensusName.textContent = "—";
            if (consensusScore) consensusScore.textContent = "—";
            
            // Ocultar imagenes y mostrar placeholders
            [imgCenital, imgLatL, imgLatR].forEach(img => { if (img) img.style.display = "none"; });
            [placeCenital, placeLatL, placeLatR].forEach(p => { if (p) p.style.display = "flex"; });
            [infCenital, infLatL, infLatR].forEach(inf => { if (inf) inf.style.display = "none"; });
            
            if (meanAccuracy) meanAccuracy.textContent = "0.0%";
            if (worstPieces) worstPieces.textContent = "—";
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" style="text-align:center; padding:20px; color:var(--text-secondary);">
                            Sin datos de inferencia. Genera renders y haz click en "Identificar Piezas".
                        </td>
                    </tr>
                `;
            }
            return;
        }
        
        // Hay renders, mostrar la pieza actual
        const item = metadata.renders[currentIndex];
        if (indicator) indicator.textContent = `Pieza ${currentIndex + 1} de ${metadata.renders.length}`;
        if (gtRef) gtRef.textContent = item.ref;
        if (gtName) gtName.textContent = item.name;
        if (gtColorBadge) gtColorBadge.style.backgroundColor = item.color_hex;
        if (gtColorText) gtColorText.textContent = `LDraw #${item.color_code}`;
        
        // Cargar imagenes
        const cenCam = item.cameras.cenital;
        const latLCam = item.cameras.lateral_l;
        const latRCam = item.cameras.lateral_r;
        
        if (imgCenital && cenCam) {
            imgCenital.src = `${API_BASE}${cenCam.image_url}?t=${Date.now()}`;
            imgCenital.style.display = "block";
        }
        if (placeCenital) placeCenital.style.display = "none";
        
        if (imgLatL && latLCam) {
            imgLatL.src = `${API_BASE}${latLCam.image_url}?t=${Date.now()}`;
            imgLatL.style.display = "block";
        }
        if (placeLatL) placeLatL.style.display = "none";
        
        if (imgLatR && latRCam) {
            imgLatR.src = `${API_BASE}${latRCam.image_url}?t=${Date.now()}`;
            imgLatR.style.display = "block";
        }
        if (placeLatR) placeLatR.style.display = "none";
        
        // Si hay resultados de inferencia, mostrar la prediccion
        if (inferenceResults && inferenceResults.results && inferenceResults.results[currentIndex]) {
            const inf = inferenceResults.results[currentIndex];
            
            if (consensusRef) {
                consensusRef.textContent = inf.consensus_ref;
                if (inf.is_consensus_correct) {
                    consensusRef.style.color = "#00ff88";
                } else {
                    consensusRef.style.color = "#ef4444";
                }
            }
            if (consensusName) consensusName.textContent = inf.consensus_name;
            if (consensusScore) consensusScore.textContent = `Confianza: ${(inf.consensus_score * 100).toFixed(1)}%`;
            
            // Camara cenital prediccion
            const pCen = inf.cameras.cenital;
            if (pCen && infCenital) {
                infCenital.style.display = "flex";
                if (blCenital) blCenital.src = pCen.bricklink_url || "";
                if (refCenital) {
                    refCenital.textContent = pCen.predicted_ref;
                    refCenital.style.color = pCen.is_correct ? "#00ff88" : "#ef4444";
                }
                if (scoreCenital) scoreCenital.textContent = `Conf: ${(pCen.score * 100).toFixed(0)}%`;
            } else if (infCenital) {
                infCenital.style.display = "none";
            }
            
            // Camara lateral L prediccion
            const pLatL = inf.cameras.lateral_l;
            if (pLatL && infLatL) {
                infLatL.style.display = "flex";
                if (blLatL) blLatL.src = pLatL.bricklink_url || "";
                if (refLatL) {
                    refLatL.textContent = pLatL.predicted_ref;
                    refLatL.style.color = pLatL.is_correct ? "#00ff88" : "#ef4444";
                }
                if (scoreLatL) scoreLatL.textContent = `Conf: ${(pLatL.score * 100).toFixed(0)}%`;
            } else if (infLatL) {
                infLatL.style.display = "none";
            }
            
            // Camara lateral R prediccion
            const pLatR = inf.cameras.lateral_r;
            if (pLatR && infLatR) {
                infLatR.style.display = "flex";
                if (blLatR) blLatR.src = pLatR.bricklink_url || "";
                if (refLatR) {
                    refLatR.textContent = pLatR.predicted_ref;
                    refLatR.style.color = pLatR.is_correct ? "#00ff88" : "#ef4444";
                }
                if (scoreLatR) scoreLatR.textContent = `Conf: ${(pLatR.score * 100).toFixed(0)}%`;
            } else if (infLatR) {
                infLatR.style.display = "none";
            }
        } else {
            // Ocultar info de inferencia
            if (consensusRef) {
                consensusRef.textContent = "—";
                consensusRef.style.color = "var(--accent)";
            }
            if (consensusName) consensusName.textContent = "—";
            if (consensusScore) consensusScore.textContent = "—";
            
            [infCenital, infLatL, infLatR].forEach(inf => { if (inf) inf.style.display = "none"; });
        }
    }
    
    // Eventos de botones de navegacion
    if (btnPrev) {
        btnPrev.addEventListener("click", () => {
            if (currentIndex > 0) {
                currentIndex--;
                updateUI();
            }
        });
    }
    
    if (btnNext) {
        btnNext.addEventListener("click", () => {
            if (metadata && currentIndex < metadata.renders.length - 1) {
                currentIndex++;
                updateUI();
            }
        });
    }
    
    // Boton: Generar Renders
    if (btnGenerate) {
        btnGenerate.addEventListener("click", async () => {
            if (!selectSet) return;
            const setId = selectSet.value;
            btnGenerate.disabled = true;
            btnGenerate.innerText = "⏳ Generando en Blender...";
            
            // Deshabilitar el boton de inferencia temporalmente
            if (btnInfer) btnInfer.disabled = true;
            
            try {
                const response = await fetch(`${API_BASE}/generate_single_piece_renders?set_id=${setId}`, {
                    method: "POST"
                });
                const res = await response.json();
                if (response.ok && res.status === "success") {
                    metadata = res.metadata;
                    currentIndex = 0;
                    inferenceResults = null; // Limpiar inferencia anterior
                    alert("✅ Renders generados correctamente con Cycles GPU (Metal).");
                } else {
                    alert("❌ Error al generar renders: " + (res.detail || res.message || "Error desconocido"));
                }
            } catch (e) {
                console.error("Excepcion generando renders:", e);
                alert("❌ Excepción al generar renders: " + e.message);
            } finally {
                btnGenerate.disabled = false;
                btnGenerate.innerText = "⚡ Generar Renders";
                updateUI();
            }
        });
    }
    
    // Boton: Identificar Piezas (Inferencia)
    if (btnInfer) {
        btnInfer.addEventListener("click", async () => {
            if (!selectSet) return;
            const setId = selectSet.value;
            btnInfer.disabled = true;
            btnInfer.innerText = "⏳ Identificando...";
            
            try {
                const response = await fetch(`${API_BASE}/inference_multicam_set?set_id=${setId}`, {
                    method: "POST"
                });
                const res = await response.json();
                if (response.ok && res.status === "success") {
                    inferenceResults = res;
                    
                    // Actualizar footer de estadisticas
                    if (meanAccuracy) meanAccuracy.textContent = `${res.mean_accuracy}%`;
                    if (worstPieces) worstPieces.textContent = res.worst_3_pieces;
                    
                    // Llenar tabla de inventario
                    if (tbody) {
                        tbody.innerHTML = "";
                        res.inventory.forEach(item => {
                            const tr = document.createElement("tr");
                            tr.style.borderBottom = "1px solid var(--border)";
                            
                            // Color del texto de accuracy: verde si es alto, rojo si es bajo
                            let accColor = "var(--text-secondary)";
                            if (item.accuracy_pct >= 90) accColor = "#00ff88";
                            else if (item.accuracy_pct < 50) accColor = "#ef4444";
                            
                            // Imagen de catalogo de BrickLink
                            const blImgUrl = `https://img.bricklink.com/ItemImage/PN/15/${item.ref}.png`;
                            
                            tr.innerHTML = `
                                <td style="padding:10px; font-weight:bold; color:var(--accent);">${item.ref}</td>
                                <td style="padding:10px; text-align:center;">
                                    <img src="${blImgUrl}" style="height:32px; object-fit:contain; background:#fff; padding:2px; border-radius:4px;" onerror="this.src='https://img.bricklink.com/ItemImage/PN/0/${item.ref}.png'; this.onerror=null;">
                                </td>
                                <td style="padding:10px;">${item.name}</td>
                                <td style="padding:10px; text-align:center;">${item.qty_original}</td>
                                <td style="padding:10px; text-align:center; font-weight:bold; color:${item.qty_detected === item.qty_original ? '#00ff88' : '#eab308'};">${item.qty_detected}</td>
                                <td style="padding:10px; text-align:center; font-weight:bold; color:${accColor};">${item.accuracy_pct}%</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    }
                    
                    alert("✅ Identificación completada. Desplázate por el carrusel para ver las predicciones individuales y por consenso.");
                } else if (res.status === "not_ready") {
                    alert("⚠️ " + res.message);
                } else {
                    alert("❌ Error en la inferencia: " + (res.detail || res.message || "Error desconocido"));
                }
            } catch (e) {
                console.error("Excepcion en inferencia multicam:", e);
                alert("❌ Excepción en inferencia multicam: " + e.message);
            } finally {
                btnInfer.disabled = false;
                btnInfer.innerText = "🔍 Identificar Piezas";
                updateUI();
            }
        });
    }
    
    // Inicializar al arrancar
    loadExistingMetadata();
}

// =============================================================
// ENTRENAMIENTO: Pipelines de datos Paso 5
// =============================================================
function initStep5Pipelines() {
    const pipelines = [
        { id: "yolo-cenital-render", title: "YOLO Cenital Render" },
        { id: "yolo-cenital-train", title: "YOLO Cenital Train" },
        { id: "yolo-lateral-render", title: "YOLO Lateral Render" },
        { id: "yolo-lateral-train", title: "YOLO Lateral Train" },
        { id: "dinov2-cenital", title: "DINOv2 Cenital" },
        { id: "dinov2-lateral", title: "DINOv2 Lateral" }
    ];

    pipelines.forEach(p => {
        const storedStatus = localStorage.getItem(`status-${p.id}`) || "Pendiente";
        const storedTime = localStorage.getItem(`time-${p.id}`) || "—";
        updatePipelineUI(p.id, storedStatus, storedTime);

        const btn = document.getElementById(`btn-render-${p.id}`);
        if (btn) {
            btn.addEventListener("click", () => runPipelineRender(p.id));
        }
    });

    const specLinks = document.querySelectorAll(".open-specs-link");
    specLinks.forEach(link => {
        link.addEventListener("click", async (e) => {
            e.preventDefault();
            try {
                if (typeof pywebview !== "undefined" && pywebview.api && pywebview.api.open_training_specs) {
                    await pywebview.api.open_training_specs();
                } else {
                    console.log("pywebview ready link fallback - docs/training_parameters_detail.md");
                    alert("Abre el archivo docs/training_parameters_detail.md para ver las especificaciones.");
                }
            } catch (err) {
                console.error("Error abriendo especificaciones:", err);
            }
        });
    });
}

function updatePipelineUI(id, status, timestamp) {
    const statusEl = document.getElementById(`status-${id}`);
    const timeEl = document.getElementById(`time-${id}`);
    const btn = document.getElementById(`btn-render-${id}`);

    if (statusEl) {
        statusEl.textContent = status;
        statusEl.className = "piece-conf"; // base class
        if (status === "Trained" || status === "Indexado" || status === "Listo") {
            statusEl.classList.add("high");
            statusEl.style.background = "rgba(0, 255, 136, 0.15)";
            statusEl.style.color = "#00ff88";
            statusEl.style.border = "1px solid rgba(0, 255, 136, 0.3)";
        } else if (status === "⏳ Generando...") {
            statusEl.classList.add("low");
            statusEl.style.background = "rgba(56, 189, 248, 0.15)";
            statusEl.style.color = "#38bdf8";
            statusEl.style.border = "1px solid rgba(56, 189, 248, 0.3)";
        } else {
            statusEl.classList.add("low");
            statusEl.style.background = "rgba(239, 68, 68, 0.15)";
            statusEl.style.color = "#ef4444";
            statusEl.style.border = "1px solid rgba(239, 68, 68, 0.3)";
        }
    }
    if (timeEl) timeEl.textContent = timestamp;
    if (btn) {
        if (status === "⏳ Generando...") {
            btn.disabled = true;
            btn.innerText = "⏳ Generando...";
        } else {
            btn.disabled = false;
            if (id.endsWith("-train")) {
                btn.innerText = "Entrenar modelo";
            } else {
                btn.innerText = "Generar renders";
            }
        }
    }
}

async function runPipelineRender(id) {
    if (typeof pywebview === "undefined" || !pywebview.api || !pywebview.api.start_step5_pipeline) {
        // Fallback mockup
        updatePipelineUI(id, "⏳ Generando...", "En proceso");
        setTimeout(() => {
            const now = new Date();
            const dateStr = now.toLocaleDateString() + " " + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const finalStatus = id.startsWith("yolo") ? "Trained" : "Indexado";
            localStorage.setItem(`status-${id}`, finalStatus);
            localStorage.setItem(`time-${id}`, dateStr);
            updatePipelineUI(id, finalStatus, dateStr);
            alert(`[Mock] ✅ Renders y entrenamiento para ${id.toUpperCase()} completados.`);
        }, 2500);
        return;
    }

    try {
        const res = await pywebview.api.start_step5_pipeline(id);
        if (res.status === "error") {
            alert(`Error: ${res.message}`);
            return;
        }

        updatePipelineUI(id, "⏳ Generando...", "En proceso");

        // Interval to poll status
        const intervalId = setInterval(async () => {
            try {
                const statusData = await pywebview.api.get_step5_pipeline_status(id);
                if (statusData.done) {
                    clearInterval(intervalId);
                    
                    const now = new Date();
                    const dateStr = now.toLocaleDateString() + " " + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    const finalStatus = statusData.status;

                    if (finalStatus !== "Error") {
                        localStorage.setItem(`status-${id}`, finalStatus);
                        localStorage.setItem(`time-${id}`, dateStr);
                        updatePipelineUI(id, finalStatus, dateStr);
                        setTimeout(() => {
                            alert(`✅ Renders y entrenamiento para ${id.toUpperCase()} completados con éxito.`);
                        }, 100);
                    } else {
                        updatePipelineUI(id, "Pendiente", "—");
                        setTimeout(() => {
                            alert(`❌ El entrenamiento de ${id.toUpperCase()} falló: ${statusData.error}`);
                        }, 100);
                    }
                }
            } catch (pollErr) {
                console.error("Error consultando estado del pipeline:", pollErr);
            }
        }, 2000);

    } catch (err) {
        console.error("Error iniciando pipeline:", err);
        setTimeout(() => {
            alert(`Error al iniciar: ${err.message}`);
        }, 100);
    }
}

