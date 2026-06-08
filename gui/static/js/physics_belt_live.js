// LegoVision Physics Belt Live
var physBeltActive = false;
var physBeltPieces = [];
var physBeltPool = [];
var physBeltInstanceCounter = 0;
var physBeltHistory = [];
var physBeltHistoryIdx = 0;
var physBeltImgCache = new Map();

function initPhysBelt() {
  physBeltActive = false;
  physBeltPieces = [];
  physBeltPool = [];
  physBeltHistory = [];
  physBeltHistoryIdx = 0;
  physBeltInstanceCounter = 0;
  window.physBeltLastCaptureTime = 0;
  updatePhysBeltHistoryUI();
}

function activatePhysBelt() {
  initPhysBelt();
  physBeltPool = buildPhysBeltPool(currentSetParts);
  physBeltActive = true;

  // Pre-poblar la cinta de forma inmediata según la densidad del UX
  var targetCount = Math.min(maxPiecesInField, physBeltPool.length);
  for (var i = 0; i < targetCount; i++) {
    var randomY = Math.random() * (CANVAS_H - 120) + 40;
    var piece = spawnPhysBeltPiece(randomY);
    if (!piece) break;

    // Resolver solapamientos en la colocación inicial
    var ok = true, att = 0;
    while (att < 25) {
      ok = true;
      for (var bi = 0; bi < physBeltPieces.length; bi++) {
        var bp = physBeltPieces[bi];
        var distY = Math.abs(bp.y - piece.y);
        var distX = Math.abs(bp.x - piece.x);
        if (distY < (bp.h_px / 2 + piece.h_px / 2) * 1.15 && 
            distX < (bp.w_px / 2 + piece.w_px / 2) * 1.15) {
          ok = false;
          break;
        }
      }
      if (ok) break;
      piece.x = 25 + Math.random() * (CANVAS_W - 50);
      piece.y = Math.random() * (CANVAS_H - 120) + 40;
      att++;
    }

    if (ok) {
      // Las piezas pre-pobladas ya están en reposo en la cinta (z = 0)
      piece.z = 0;
      piece.vz = 0;
      physBeltPieces.push(piece);
    }
  }
}

function deactivatePhysBelt() {
  physBeltActive = false;
  physBeltPieces = [];
  physBeltPool = [];
}

function buildPhysBeltPool(parts) {
  var pool = [];
  for (var i = 0; i < parts.length; i++) {
    var pt = parts[i];
    for (var q = 0; q < pt.qty; q++) {
      var ci = Math.floor(Math.random() * 15);
      var ch = (pt.color || '#A0A5A9').replace('#', '').toUpperCase();
      var fallbackUrl = API_BASE + '/renders/render_' + pt.ref + '_' + ch + '.png';
      
      // Pre-cargar la imagen de referencia limpia para el HUD de DINOv2
      var refImg = new Image();
      refImg.src = fallbackUrl;

      pool.push({
        ref: pt.ref,
        name: pt.name,
        colorHex: ch,
        colorName: pt.colorName || "",
        color: pt.color || "#A0A5A9",
        cropUrl: API_BASE + '/renders/physics_scatter_' + pt.ref + '_' + ch + '_crop_' + ci + '.png',
        fallbackUrl: fallbackUrl,
        refImg: refImg,
        w: pt.w,
        h: pt.h,
        cropIdx: ci
      });
    }
  }
  // Barajar (Fisher-Yates)
  for (var s = pool.length - 1; s > 0; s--) {
    var r = Math.floor(Math.random() * (s + 1));
    var t = pool[s];
    pool[s] = pool[r];
    pool[r] = t;
  }
  return pool;
}

function getPieceThickness(name, ref) {
  var n = name.toLowerCase();
  if (ref.startsWith("sw") || n.includes("minifig")) return 15.0; // Minifigura
  if (n.includes("brick")) return 9.6; // Ladrillo standard
  if (n.includes("plate") || n.includes("tile")) return 3.2; // Placa / Loseta
  if (n.includes("slope")) return 6.4; // Pendiente
  return 5.0; // Espesor general
}

function spawnPhysBeltPiece(yStart) {
  if (!physBeltPool.length) {
    physBeltPool = buildPhysBeltPool(currentSetParts);
    if (!physBeltPool.length) return null;
  }
  var pt = physBeltPool.pop();
  if (!pt) return null;
  var iid = physBeltInstanceCounter++;

  // Obtener dimensiones de referencia del catálogo estático si no vienen en la BD
  var refW = pt.w;
  var refH = pt.h;
  if (!refW || !refH) {
    var staticPart = (typeof SET_75078_1_PARTS !== "undefined") ? 
      SET_75078_1_PARTS.find(function(sp) { return sp.ref === pt.ref; }) : null;
    if (staticPart) {
      refW = staticPart.w;
      refH = staticPart.h;
    } else {
      refW = 28; // Fallback por defecto (equivalente a pieza 2x2)
      refH = 28;
    }
  }

  // Conversión a dimensiones reales (las piezas de los sets están a escala 1.75)
  // 640 px = 200 mm (20 cm) -> PX_PER_MM = 3.2
  var w_px = (refW / 1.75) * 3.2;
  var h_px = (refH / 1.75) * 3.2;

  var mg = 20 + Math.max(w_px, h_px) / 2;
  var x = mg + Math.random() * (CANVAS_W - 2 * mg);
  var y = (yStart !== undefined) ? yStart : (-h_px / 2 - 30 - Math.random() * 50);

  var img = physBeltImgCache.get(pt.cropUrl);
  if (!img) {
    img = new Image();
    (function(k, fb) {
      img.onerror = function() {
        var f = new Image();
        f.src = fb;
        physBeltImgCache.set(k, f);
      };
    })(pt.cropUrl, pt.fallbackUrl);
    img.src = pt.cropUrl;
    physBeltImgCache.set(pt.cropUrl, img);
  }

  var t_mm = getPieceThickness(pt.name, pt.ref);

  return {
    instanceId: iid,
    ref: pt.ref,
    name: pt.name,
    colorHex: pt.colorHex,
    colorName: pt.colorName,
    color: pt.color,
    w: pt.w,
    h: pt.h,
    w_px: w_px,
    h_px: h_px,
    t_mm: t_mm,
    x: x,
    y: y,
    z: (yStart !== undefined) ? 0 : (80 + Math.random() * 60), // z inicial si cae desde arriba
    vz: 0,
    restitution: 0.2 + Math.random() * 0.15,
    gravity: 0.45,
    img: img,
    refImg: pt.refImg,
    cropUrl: pt.cropUrl,
    angle: (Math.random() - 0.5) * 0.5,
    spin: (Math.random() - 0.5) * 0.05,
    counted: false,
    yoloConf: Math.floor(90 + Math.random() * 9),
    dinoConf: Math.floor(85 + Math.random() * 12)
  };
}

function fillPhysBeltPieces() {
  while (physBeltPieces.length < maxPiecesInField && physBeltPool.length > 0) {
    var topY = 0;
    if (physBeltPieces.length > 0) {
      topY = physBeltPieces[0].y - physBeltPieces[0].h_px / 2;
      for (var k = 1; k < physBeltPieces.length; k++) {
        var cv = physBeltPieces[k].y - physBeltPieces[k].h_px / 2;
        if (cv < topY) topY = cv;
      }
    }
    
    // Spacing dinámico basado en la densidad seleccionada (maxPiecesInField)
    // Permite que las piezas aparezcan mucho más juntas y en paralelo si el slider está alto
    var targetDensity = Math.max(5, maxPiecesInField);
    var verticalSpacing = 90.0 / (targetDensity / 9.0);
    var sy = Math.min(-55, topY - verticalSpacing - Math.random() * 25);
    
    var piece = spawnPhysBeltPiece(sy);
    if (!piece) break;

    // Repulsión física inicial para evitar solapamientos
    var ok = true, att = 0;
    while (att < 25) {
      ok = true;
      for (var bi = 0; bi < physBeltPieces.length; bi++) {
        var bp = physBeltPieces[bi];
        // Si no se solapan verticalmente, no hay conflicto
        if (Math.abs(bp.y - piece.y) > (bp.h_px / 2 + piece.h_px / 2) * 1.15) continue;
        // Si se solapan verticalmente, evitar que estén muy cerca en horizontal
        if (Math.abs(bp.x - piece.x) < (bp.w_px / 2 + piece.w_px / 2) * 1.15) {
          ok = false;
          break;
        }
      }
      if (ok) break;
      var margin = 20 + Math.max(piece.w_px, piece.h_px) / 2;
      piece.x = margin + Math.random() * (CANVAS_W - 2 * margin);
      att++;
    }

    if (ok) {
      physBeltPieces.push(piece);
    } else {
      break;
    }
  }
}

function drawPhysBeltPiece(p) {
  var Z_cam = 250.0; // Distancia física cámara -> cinta en mm (25 cm)
  
  // Factor de escala de perspectiva basado en la distancia a la cámara
  var scale = Z_cam / (Z_cam - (p.z + p.t_mm));
  var max_dim = Math.max(p.w_px, p.h_px);
  var draw_w = max_dim * scale;
  var draw_h = max_dim * scale;

  var parallaxY = -(p.z + p.t_mm) * 0.18;

  // Dibujar la pieza con paralaje vertical y sombra nativa
  ctx.save();
  ctx.translate(p.x, p.y + parallaxY);
  ctx.rotate(p.angle);
  
  var img = physBeltImgCache.get(p.cropUrl);
  if (img && img.complete && img.naturalWidth > 0) {
    // Configurar sombra nativa del canvas para proyectar la forma alfa de la pieza
    var shadowOpacity = Math.max(0.08, 0.45 - (p.z + p.t_mm) * 0.0025);
    var shadowShiftX = (p.z + p.t_mm) * 0.18;
    var shadowShiftY = 3 + (p.z + p.t_mm) * 0.26;
    var shadowBlur = 4 + (p.z + p.t_mm) * 0.14;

    ctx.shadowColor = "rgba(0, 0, 0, " + shadowOpacity + ")";
    ctx.shadowBlur = shadowBlur;
    ctx.shadowOffsetX = shadowShiftX;
    ctx.shadowOffsetY = shadowShiftY;

    ctx.drawImage(img, -draw_w / 2, -draw_h / 2, draw_w, draw_h);
  } else {
    // Sombra simplificada para el fallback
    var shadowOpacity = Math.max(0.08, 0.45 - (p.z + p.t_mm) * 0.0025);
    ctx.shadowColor = "rgba(0, 0, 0, " + shadowOpacity + ")";
    ctx.shadowBlur = 4;
    ctx.shadowOffsetX = 2;
    ctx.shadowOffsetY = 4;

    ctx.beginPath();
    ctx.roundRect(-p.w_px / 2, -p.h_px / 2, p.w_px, p.h_px, 4);
    ctx.fillStyle = p.color || '#A0A5A9';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
}

function animatePhysBelt(timestamp, dt) {
  if (!physBeltActive) return;

  // vy_mm_s = velocidad física en mm/s
  // vy = velocidad en píxeles por frame
  var vy_mm_s = sessionActive ? ((beltSpeed * 1000.0) / 60.0) : 0;
  var vy = vy_mm_s * 3.2 * dt;

  fillPhysBeltPieces();

  var toRemove = [];
  for (var pi = 0; pi < physBeltPieces.length; pi++) {
    var p = physBeltPieces[pi];
    
    if (p.z > 0) {
      // Física en el eje Z (gravedad y rebotes)
      p.vz -= p.gravity;
      p.z += p.vz;
      p.y += vy * 0.35; // Resistencia de avance en el aire
      p.angle += p.spin;
      
      if (p.z <= 0) {
        p.z = 0;
        p.vz = -p.vz * p.restitution;
        p.spin = (Math.random() - 0.5) * 0.08;
        p.angle += (Math.random() - 0.5) * 0.15;
        p.x += (Math.random() - 0.5) * 6;
        p.y += (Math.random() - 0.5) * 4;
        
        if (Math.abs(p.vz) < 0.6) {
          p.z = 0;
          p.vz = 0;
          p.spin = (Math.random() - 0.5) * 0.002;
        }
      }
    } else {
      // Asentado en la cinta
      p.y += vy;
      p.angle += p.spin;
      // Vibración de motor
      p.x += (Math.random() - 0.5) * 0.35;
      p.y += (Math.random() - 0.5) * 0.35;
      p.angle += (Math.random() - 0.5) * 0.001;
    }

    var margin = 20 + Math.max(p.w_px, p.h_px) / 2;
    p.x = Math.max(margin, Math.min(CANVAS_W - margin, p.x));

    // Conteo de inventario único al cruzar el centro del FOV (Y = 320)
    if (p.y >= (CANVAS_H / 2) && !p.counted) {
      p.counted = true;
      sessionIdentifiedCounts[p.ref] = (sessionIdentifiedCounts[p.ref] || 0) + 1;
      updateSessionInventoryTable();
      
      try {
        var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        var osc = audioCtx.createOscillator();
        var gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.type = "sine";
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.03, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.15);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.15);
      } catch (e) {}
    }

    if (p.y > CANVAS_H + Math.max(p.w_px, p.h_px) + 20) {
      toRemove.push(p.instanceId);
    }
  }

  if (toRemove.length) {
    physBeltPieces = physBeltPieces.filter(function(pp) {
      return toRemove.indexOf(pp.instanceId) === -1;
    });
  }

  // Redibujar la cinta
  drawBeltBackground();
  
  // Dibujar las piezas
  for (var di = 0; di < physBeltPieces.length; di++) {
    drawPhysBeltPiece(physBeltPieces[di]);
  }

  // Actualizar bounding boxes activas para el render en vivo en pantalla (YOLO)
  activeBboxes = [];
  for (var b2 = 0; b2 < physBeltPieces.length; b2++) {
    var p2 = physBeltPieces[b2];
    if (p2.y < -p2.h_px/2 || p2.y > CANVAS_H + p2.h_px/2) continue;
    
    var scale = 250.0 / (250.0 - (p2.z + p2.t_mm));
    var dw = p2.w_px * scale;
    var dh = p2.h_px * scale;
    
    activeBboxes.push({
      x1: p2.x - dw / 2,
      y1: p2.y - dh / 2,
      x2: p2.x + dw / 2,
      y2: p2.y + dh / 2,
      label: p2.ref,
      conf: p2.dinoConf / 100,
      name: p2.name
    });
  }
  drawBboxOverlay();

  var elP = document.getElementById('live-frame-pieces');
  if (elP) elP.innerText = activeBboxes.length;

  // LÓGICA DE CAPTURA SECUENCIAL MATEMÁTICA DEL CAMPO COMPLETO
  if (sessionActive && vy_mm_s > 0) {
    var vy_px_s = vy_mm_s * 3.2; // px/s
    var T_pass = CANVAS_H / vy_px_s; // tiempo de tránsito total por el FOV en segundos
    var capture_interval_s = T_pass / 5; // capturar 5 veces en ese tiempo
    
    if (!window.physBeltLastCaptureTime) window.physBeltLastCaptureTime = timestamp;
    var elapsed_s = (timestamp - window.physBeltLastCaptureTime) / 1000.0;
    
    if (elapsed_s >= capture_interval_s) {
      window.physBeltLastCaptureTime = timestamp;
      captureFullFrameSnapshot();
    }
  }
}

function captureFullFrameSnapshot() {
  if (!sessionActive) return;

  // Encontrar la pieza más cercana a la línea central de detección para el HUD de DINOv2
  var triggeringPiece = null;
  var minDistanceToCenter = Infinity;
  var centerY = CANVAS_H / 2;

  for (var i = 0; i < physBeltPieces.length; i++) {
    var p = physBeltPieces[i];
    if (p.y >= 0 && p.y <= CANVAS_H) {
      var dist = Math.abs(p.y - centerY);
      if (dist < minDistanceToCenter) {
        minDistanceToCenter = dist;
        triggeringPiece = p;
      }
    }
  }

  // Crear canvas offscreen para el snapshot con overlays
  var snap = document.createElement("canvas");
  snap.width = CANVAS_W;
  snap.height = CANVAS_H;
  var sctx = snap.getContext("2d");

  // 1. Dibujar el fotograma actual de la cinta
  sctx.drawImage(canvas, 0, 0);

  // 2. Sobreponer cajas de detección de YOLO para TODAS las piezas visibles
  physBeltPieces.forEach(function(pp) {
    if (pp.y < -pp.h_px / 2 || pp.y > CANVAS_H + pp.h_px / 2) return;

    var Z_cam = 250.0;
    var scale = Z_cam / (Z_cam - (pp.z + pp.t_mm));
    var dw = pp.w_px * scale;
    var dh = pp.h_px * scale;

    var x1 = pp.x - dw / 2;
    var y1 = pp.y - dh / 2;

    var isTrigger = triggeringPiece && (pp.instanceId === triggeringPiece.instanceId);

    // Caja YOLO (verde) o resaltada (cian)
    sctx.strokeStyle = isTrigger ? "#38bdf8" : "#00ff88";
    sctx.lineWidth = isTrigger ? 3 : 2;
    sctx.shadowColor = isTrigger ? "rgba(56, 189, 248, 0.4)" : "rgba(0, 255, 136, 0.4)";
    sctx.shadowBlur = 6;
    sctx.strokeRect(x1, y1, dw, dh);
    sctx.shadowBlur = 0;

    // Etiqueta
    var label = pp.ref + " " + pp.yoloConf + "%";
    sctx.font = "bold 10px monospace";
    var tw = sctx.measureText(label).width + 6;
    sctx.fillStyle = isTrigger ? "#38bdf8" : "#00ff88";
    sctx.fillRect(x1, y1 - 14, tw, 13);
    sctx.fillStyle = "#000";
    sctx.fillText(label, x1 + 3, y1 - 4);
  });

  // 3. Sobreponer tarjeta de HUD DINOv2 para la pieza central seleccionada
  if (triggeringPiece) {
    var pt = triggeringPiece;
    var hudW = 175;
    var hudH = 75;
    var hudX = CANVAS_W - hudW - 15;
    var hudY = 15;

    // Panel HUD translúcido cian
    sctx.fillStyle = "rgba(15, 23, 42, 0.85)";
    sctx.strokeStyle = "#38bdf8";
    sctx.lineWidth = 1.5;
    sctx.beginPath();
    sctx.roundRect(hudX, hudY, hudW, hudH, 8);
    sctx.fill();
    sctx.stroke();

    // Título HUD
    sctx.fillStyle = "#38bdf8";
    sctx.font = "bold 9px Outfit, sans-serif";
    sctx.fillText("DINOv2 CLASSIFICATION", hudX + 10, hudY + 16);

    // Render de referencia de BrickLink / LDraw
    if (pt.refImg && pt.refImg.complete && pt.refImg.naturalWidth > 0) {
      sctx.drawImage(pt.refImg, hudX + 10, hudY + 24, 40, 40);
    } else {
      sctx.fillStyle = "rgba(255, 255, 255, 0.1)";
      sctx.fillRect(hudX + 10, hudY + 24, 40, 40);
      sctx.fillStyle = "#fff";
      sctx.font = "8px sans-serif";
      sctx.fillText("No img", hudX + 15, hudY + 47);
    }

    // Texto descriptivo de pieza y similitud
    sctx.fillStyle = "#fff";
    sctx.font = "bold 10px Outfit, sans-serif";
    var dispName = pt.name;
    if (dispName.length > 17) dispName = dispName.substring(0, 15) + "...";
    sctx.fillText(dispName, hudX + 58, hudY + 34);

    sctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    sctx.font = "9px monospace";
    sctx.fillText("Ref: " + pt.ref, hudX + 58, hudY + 46);

    sctx.fillStyle = "#7df9aa";
    sctx.font = "bold 9px Outfit, sans-serif";
    sctx.fillText("Sim: " + pt.dinoConf + "%", hudX + 58, hudY + 58);
  }

  var dataUrl = snap.toDataURL("image/png");

  // Añadir al historial
  var dispRef = triggeringPiece ? triggeringPiece.ref : "N/A";
  var dispName = triggeringPiece ? triggeringPiece.name : "Vacio";
  var dispConf = triggeringPiece ? (triggeringPiece.dinoConf / 100) : 0.0;

  physBeltHistory.push({
    processedDataUrl: dataUrl,
    ref: dispRef,
    name: dispName,
    avgConf: dispConf,
    captureCount: 5,
    ts: new Date().toLocaleTimeString()
  });

  // Limitar historial a 30 elementos
  if (physBeltHistory.length > 30) {
    physBeltHistory.shift();
  }

  physBeltHistoryIdx = physBeltHistory.length - 1;
  updatePhysBeltHistoryUI();
}

function updatePhysBeltHistoryUI() {
  var hc = document.getElementById("history-canvas");
  var prevBtn = document.getElementById("btn-history-prev");
  var nextBtn = document.getElementById("btn-history-next");
  var pageNum = document.getElementById("history-page-num");
  var thumbGrid = document.getElementById("history-thumbnails");
  
  if (!hc) return;

  if (!physBeltHistory.length) {
    pageNum.innerText = "Sin fotos capturadas";
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    var hctx = hc.getContext("2d");
    hc.width = 640;
    hc.height = 640;
    hctx.fillStyle = "#0f172a";
    hctx.fillRect(0, 0, 640, 640);
    hctx.fillStyle = "rgba(255, 255, 255, 0.2)";
    hctx.font = "14px monospace";
    hctx.fillText("Inicia sesion para ver historial", 180, 320);
    if (thumbGrid) thumbGrid.innerHTML = "";
    return;
  }

  var idx = physBeltHistoryIdx;
  var total = physBeltHistory.length;
  pageNum.innerText = "Frame " + (idx + 1) + " / " + total;
  prevBtn.disabled = (idx <= 0);
  nextBtn.disabled = (idx >= total - 1);

  var entry = physBeltHistory[idx];
  var img = new Image();
  img.onload = function() {
    var hctx = hc.getContext("2d");
    hc.width = img.width;
    hc.height = img.height;
    hctx.drawImage(img, 0, 0, hc.width, hc.height);
  };
  img.src = entry.processedDataUrl;

  if (thumbGrid) {
    thumbGrid.innerHTML = "";
    for (var t = 0; t < physBeltHistory.length; t++) {
      var e = physBeltHistory[t];
      var div = document.createElement("div");
      div.style.cssText = "cursor:pointer;border:2px solid " + (t === idx ? "#38bdf8" : "transparent") + ";border-radius:6px;overflow:hidden;position:relative;background:rgba(0,0,0,0.3);";
      
      var timg = new Image();
      timg.src = e.processedDataUrl;
      timg.style.cssText = "width:100%;height:65px;object-fit:cover;display:block;";
      
      var lbl = document.createElement("div");
      lbl.innerText = e.ref + " " + Math.round(e.avgConf * 100) + "%";
      lbl.style.cssText = "font-size:8px;color:#38bdf8;text-align:center;padding:2px;background:rgba(0,0,0,0.8);font-family:monospace;";
      
      div.appendChild(timg);
      div.appendChild(lbl);
      
      (function(idx2) {
        div.onclick = function() {
          physBeltHistoryIdx = idx2;
          updatePhysBeltHistoryUI();
        };
      })(t);
      thumbGrid.appendChild(div);
    }
  }
}

function initPhysBeltHistoryNav() {
  var prev = document.getElementById("btn-history-prev");
  var next = document.getElementById("btn-history-next");
  if (prev) {
    prev.onclick = function() {
      if (physBeltHistoryIdx > 0) {
        physBeltHistoryIdx--;
        updatePhysBeltHistoryUI();
      }
    };
  }
  if (next) {
    next.onclick = function() {
      if (physBeltHistoryIdx < physBeltHistory.length - 1) {
        physBeltHistoryIdx++;
        updatePhysBeltHistoryUI();
      }
    };
  }
}
