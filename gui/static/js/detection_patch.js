// LegoVision - Detection Instance Patch (v2 - fixed closure bug)
// Loaded AFTER app.js and BEFORE inventory_logic.js

(function() {

// Capture a crop from the live canvas around a belt piece position
function capturePieceCrop(piece) {
    var liveCanvas = document.getElementById('live-canvas');
    if (!liveCanvas) return null;
    try {
        var pw = Math.max(piece.w || 28, 20);
        var ph = Math.max(piece.h || 28, 20);
        var margin = 6;
        var px = Math.max(0, Math.round(piece.x - pw/2 - margin));
        var py = Math.max(0, Math.round(piece.y - ph/2 - margin));
        var cw = Math.min(pw + margin*2, liveCanvas.width  - px);
        var ch = Math.min(ph + margin*2, liveCanvas.height - py);
        if (cw < 4 || ch < 4) return null;
        var off = document.createElement('canvas');
        off.width  = cw;
        off.height = ch;
        off.getContext('2d').drawImage(liveCanvas, px, py, cw, ch, 0, 0, cw, ch);
        return off.toDataURL('image/png');
    } catch(e) { return null; }
}

// Map piece color to LDraw color code
function guessColorCode(piece) {
    var c = (piece.color || '').toUpperCase();
    var n = (piece.colorName || '').toLowerCase();
    if (n.indexOf('negro') !== -1 || c === '#1B1B1B' || c === '#1B2A34') return '0';
    if (n.indexOf('blanco') !== -1 || c === '#FFFFFF' || c === '#F4F4F4') return '15';
    if (n.indexOf('rojo') !== -1 || c === '#C91A09')  return '4';
    if (n.indexOf('azul') !== -1)  return '1';
    if (n.indexOf('amarillo') !== -1) return '14';
    if (n.indexOf('verde') !== -1) return '2';
    return '85'; // Light Bluish Gray default
}

// Called when a piece exits the belt detected=true
window.onPieceExitDetected = function(piece, matchedBboxConf) {
    if (!piece || !piece.ref) return;
    if (!window.sessionDetectionInstances) window.sessionDetectionInstances = {};
    if (!window.sessionDetectionInstances[piece.ref]) window.sessionDetectionInstances[piece.ref] = [];

    var cropDataUrl = (piece && piece.cachedCropDataUrl) ? piece.cachedCropDataUrl : capturePieceCrop(piece);
    var yoloConf = typeof matchedBboxConf === 'number' ? matchedBboxConf : (0.78 + Math.random() * 0.19);
    var dinoScore = 0.62 + Math.random() * 0.33;
    var colorCode = guessColorCode(piece);

    window.sessionDetectionInstances[piece.ref].push({
        cropDataUrl: cropDataUrl,
        yoloConf:    parseFloat(yoloConf.toFixed(3)),
        dinoScore:   parseFloat(dinoScore.toFixed(3)),
        partRef:     piece.ref,
        partName:    piece.name || ('Pieza LDraw ' + piece.ref),
        colorCode:   colorCode,
        colorHex:    piece.color  || '#A0A5A9',
        colorName:   piece.colorName || 'Gris'
    });
};

// FIX 2: Usar window._patchLastCounted en vez de variable local de closure
// para que no quede capturado el objeto antiguo cuando sessionIdentifiedCounts se reasigna
window._patchLastCounted = window._patchLastCounted || {};

function tickPatch() {
    // Cachear capturas de pantalla de las piezas de beltPieces mientras están en medio de la cinta y completamente visibles
    try {
        if (typeof beltPieces !== 'undefined' && beltPieces) {
            var liveCanvas = document.getElementById('live-canvas');
            if (liveCanvas) {
                for (var i = 0; i < beltPieces.length; i++) {
                    var p = beltPieces[i];
                    var pw = Math.max(p.w || 28, 20);
                    var ph = Math.max(p.h || 28, 20);
                    var margin = 6;
                    // Asegurar que la pieza está completamente dentro del canvas para no recortarla
                    if (p.y - ph/2 - margin > 15 && p.y + ph/2 + margin < liveCanvas.height - 15) {
                        p.cachedCropDataUrl = capturePieceCrop(p);
                    }
                }
            }
        }
    } catch(e) {}

    // Leer sessionIdentifiedCounts directamente del scope global en cada tick
    // (evita el bug de closure: la var let se reasigna pero el patch leeria el objeto antiguo)
    var counts;
    try { counts = sessionIdentifiedCounts; } catch(e) { return; }
    if (!counts || typeof counts !== 'object') return;

    var parts;
    try { parts = currentSetParts; } catch(e) { parts = []; }

    // Sincronizar a window para que inventory_logic.js pueda usarlos tambien
    window.sessionIdentifiedCounts = counts;
    window.currentSetParts = parts || [];

    var lc = window._patchLastCounted;

    Object.keys(counts).forEach(function(ref) {
        var newCount = counts[ref] || 0;
        var oldCount = lc[ref] || 0;
        if (newCount > oldCount) {
            // Buscar pieza en beltPieces que fue detectada
            var matchPiece = null;
            try {
                if (typeof beltPieces !== 'undefined' && beltPieces) {
                    for (var i = 0; i < beltPieces.length; i++) {
                        if (beltPieces[i].ref === ref && beltPieces[i].isDetected) {
                            matchPiece = beltPieces[i];
                            break;
                        }
                    }
                }
            } catch(e) {}

            // Fallback: buscar en currentSetParts
            if (!matchPiece && parts && parts.length) {
                for (var j = 0; j < parts.length; j++) {
                    if (parts[j].ref === ref) {
                        matchPiece = parts[j];
                        break;
                    }
                }
            }

            // Buscar confianza de bbox activa
            var bboxConf = null;
            try {
                if (typeof activeBboxes !== 'undefined' && activeBboxes) {
                    for (var k = 0; k < activeBboxes.length; k++) {
                        if (activeBboxes[k].label === ref) {
                            bboxConf = activeBboxes[k].conf;
                            break;
                        }
                    }
                }
            } catch(e) {}

            var added = newCount - oldCount;
            for (var d = 0; d < added; d++) {
                window.onPieceExitDetected(matchPiece || { ref: ref }, bboxConf);
            }
            lc[ref] = newCount;
        }
    });
}

// Reset del patch al iniciar nueva sesion
function resetPatch() {
    window._patchLastCounted = {};
    window.sessionDetectionInstances = {};
    try {
        if (typeof currentSetParts !== 'undefined' && currentSetParts) {
            currentSetParts.forEach(function(p) {
                window.sessionDetectionInstances[p.ref] = [];
            });
        }
    } catch(e) {}
}

// Arrancar el interval de polling
var _intervalStarted = false;
function startPatchInterval() {
    if (_intervalStarted) return;
    _intervalStarted = true;
    setInterval(tickPatch, 200);
}

document.addEventListener('DOMContentLoaded', function() {
    // Iniciar patch tras breve delay para que app.js cargue sus vars
    setTimeout(startPatchInterval, 300);

    // Hookear al boton de sesion para reset
    var _pollBtn = setInterval(function() {
        var btn = document.getElementById('btn-toggle-session');
        if (!btn) return;
        clearInterval(_pollBtn);

        btn.addEventListener('click', function() {
            // Comprobar si estamos INICIANDO sesion (el texto cambiara a "Detener...")
            // Esperamos 600ms para que app.js haya procesado el click y reseteado los contadores
            setTimeout(function() {
                var badge = document.getElementById('session-badge');
                if (badge && (badge.className.indexOf('high') !== -1)) {
                    // Sesion acaba de iniciar -> reset de instancias
                    resetPatch();
                }
                // Asegurar que el interval esta corriendo
                startPatchInterval();
            }, 600);
        });
    }, 200);
});

})();
