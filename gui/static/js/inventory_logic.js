// LegoVision - Inventory Table with Detection Instance Cards
// Loaded AFTER app.js - overrides updateSessionInventoryTable
window.sessionDetectionInstances = window.sessionDetectionInstances || {};

function buildBLImageUrl(partRef, colorCode) {
    var isMinifig = /^(sw|fig|cty|cas|pi|tor|wr|hp|arc|jw|ind|idea|tlm|dim|sh|njo|tnt|gam|utl|bat|sp|rac)/.test(String(partRef));
    if (isMinifig) return 'https://img.bricklink.com/ItemImage/MN/0/' + partRef + '.png';
    var m = {'0':'11','1':'7','4':'5','14':'3','15':'1','84':'85','85':'86','36':'17','2':'6','10':'10','25':'8','26':'26','27':'34','17':'40','73':'42'};
    return 'https://img.bricklink.com/ItemImage/PN/' + (m[String(colorCode)] || '86') + '/' + partRef + '.png';
}

function buildBLPageUrl(partRef) {
    var isMinifig = /^(sw|fig|cty|cas|pi|tor|wr|hp|arc|jw|ind)/.test(String(partRef));
    return isMinifig
        ? 'https://www.bricklink.com/v2/catalog/catalogitem.page?M=' + partRef
        : 'https://www.bricklink.com/v2/catalog/catalogitem.page?P=' + partRef;
}

function confClass(v) {
    if (v >= 0.75) return 'conf-high';
    if (v >= 0.50) return 'conf-mid';
    return 'conf-low';
}

function buildDetectionCardHtml(inst, index) {
    var blImgUrl  = buildBLImageUrl(inst.partRef, inst.colorCode);
    var blPageUrl = buildBLPageUrl(inst.partRef);
    var yoloPct = Math.round((inst.yoloConf  || 0) * 100);
    var dinoPct = Math.round((inst.dinoScore || 0) * 100);
    var yoloCls = confClass(inst.yoloConf  || 0);
    var dinoCls = confClass(inst.dinoScore || 0);
    var cropHtml = inst.cropDataUrl
        ? '<img src="' + inst.cropDataUrl + '" alt="Crop" class="det-card-img">'
        : '<div class="det-card-img-placeholder">Sin imagen</div>';
    var blImgHtml = '<img src="' + blImgUrl + '" alt="BL" class="det-card-img" onerror="this.outerHTML=\'<div class=\\\"det-card-img-placeholder\\\">Sin img BL</div>\'">';
    return '<div class="detection-instance-card">' +
        '<div class="det-card-header">Deteccion #' + (index+1) + '</div>' +
        '<div class="det-card-images">' +
            '<div class="det-card-img-box">' +
                '<span class="det-card-img-label">Cinta</span>' + cropHtml +
            '</div>' +
            '<div class="det-card-img-box">' +
                '<span class="det-card-img-label">BrickLink</span>' + blImgHtml +
            '</div>' +
        '</div>' +
        '<div class="det-card-ref">' +
            '<a href="' + blPageUrl + '" target="_blank" class="det-card-ref-link">' + inst.partRef + '</a>' +
            '<span class="det-card-ref-name">' + (inst.partName || '') + '</span>' +
        '</div>' +
        '<div class="det-card-scores">' +
            '<div class="det-score-item">' +
                '<span class="det-score-label">Confianza YOLO</span>' +
                '<div class="det-score-bar-wrap">' +
                    '<div class="det-score-bar-track"><div class="det-score-bar-fill ' + yoloCls + '" style="width:' + yoloPct + '%"></div></div>' +
                    '<span class="det-score-value ' + yoloCls + '">' + yoloPct + '%</span>' +
                '</div>' +
            '</div>' +
            '<div class="det-score-item">' +
                '<span class="det-score-label">Similitud DINOv2</span>' +
                '<div class="det-score-bar-wrap">' +
                    '<div class="det-score-bar-track"><div class="det-score-bar-fill ' + dinoCls + '" style="width:' + dinoPct + '%"></div></div>' +
                    '<span class="det-score-value ' + dinoCls + '">' + dinoPct + '%</span>' +
                '</div>' +
            '</div>' +
        '</div>' +
    '</div>';
}

function updateSessionInventoryTable() {
    var tbody = document.getElementById('session-inventory-tbody');
    if (!tbody) return;

    // FIX 1: Leer directamente del scope global (var en app.js es accesible globalmente)
    // sin depender del patch que tarda ~500ms en sincronizar a window.*
    var parts  = (typeof currentSetParts !== 'undefined' && currentSetParts && currentSetParts.length > 0)
                 ? currentSetParts
                 : (window.currentSetParts || []);
    var counts = (typeof sessionIdentifiedCounts !== 'undefined' && sessionIdentifiedCounts)
                 ? sessionIdentifiedCounts
                 : (window.sessionIdentifiedCounts || {});
    var insts  = window.sessionDetectionInstances || {};

    var totalSet = parts.reduce(function(s,p){ return s + (p.qty || 0); }, 0);
    var totalDet = parts.reduce(function(s,p){ return s + (counts[p.ref] || 0); }, 0);

    var el1 = document.getElementById('inv-total-set');
    var el2 = document.getElementById('inv-total-detected');
    if (el1) el1.textContent = totalSet;
    if (el2) el2.textContent = totalDet;

    var tableHtml = '';
    parts.forEach(function(p) {
        var count = counts[p.ref] || 0;
        var badge = '';
        if (count === p.qty) {
            badge = '<span class="badge-status ok">&#10003; Completo</span>';
        } else if (count < p.qty) {
            badge = '<span class="badge-status less">&#9888; Falta ' + (p.qty - count) + ' (De menos)</span>';
        } else {
            badge = '<span class="badge-status more">&#128680; Exceso +' + (count - p.qty) + ' (De mas)</span>';
        }
        tableHtml += '<tr>' +
            '<td style="padding:10px;font-weight:bold;color:var(--accent);">' +
            '<a href="#" onclick="openPartInspector(\'' + p.ref + '\'); return false;" style="color:var(--accent); text-decoration:underline;">' + p.ref + '</a>' +
            '</td>' +
            '<td style="padding:10px;color:#fff;">' + (p.name || ('Pieza ' + p.ref)) + ' (' + (p.colorName || '') + ')</td>' +
            '<td style="padding:10px;text-align:center;color:var(--text-secondary);font-weight:bold;">' + p.qty + '</td>' +
            '<td style="padding:10px;text-align:center;color:#fff;font-size:1rem;font-weight:bold;">' + count + '</td>' +
            '<td style="padding:10px;text-align:center;">' + badge + '</td>' +
            '</tr>';
    });
    tbody.innerHTML = tableHtml;

    var detSection = document.getElementById('session-detection-details');
    if (!detSection) return;
    var anyDet = parts.some(function(p){ return (insts[p.ref] || []).length > 0; });
    if (!anyDet) {
        detSection.innerHTML = '<div class="det-section-empty">Sin detecciones registradas aun. Inicia una sesion de inferencia.</div>';
        return;
    }
    var detHtml = '';
    parts.forEach(function(p) {
        var instances = insts[p.ref] || [];
        if (instances.length === 0) return;
        var count = counts[p.ref] || 0;
        var bc = count === p.qty ? 'ok' : (count > p.qty ? 'more' : 'less');
        var bt = count === p.qty ? 'Completo' : (count > p.qty ? 'Exceso +' + (count - p.qty) : 'Falta ' + (p.qty - count));
        detHtml += '<div class="det-part-group">';
        detHtml += '<div class="det-part-group-header">';
        detHtml += '<div class="det-part-group-title">';
        detHtml += '<span class="det-part-ref-badge">' + p.ref + '</span>';
        detHtml += '<span class="det-part-group-name">' + (p.name || ('Pieza ' + p.ref)) + '</span>';
        detHtml += '</div>';
        detHtml += '<div class="det-part-group-counts">';
        detHtml += '<span class="det-count-label">Set: <strong>' + p.qty + '</strong></span>';
        detHtml += '<span class="det-count-label">Detectadas: <strong>' + instances.length + '</strong></span>';
        detHtml += '<span class="badge-status ' + bc + '">' + bt + '</span>';
        detHtml += '</div></div>';
        detHtml += '<div class="det-cards-grid">';
        instances.forEach(function(inst, idx) { detHtml += buildDetectionCardHtml(inst, idx); });
        detHtml += '</div></div>';
    });
    detSection.innerHTML = detHtml || '<div class="det-section-empty">Sin detecciones.</div>';
}

function openPartInspector(partRef) {
    var modal = document.getElementById('part-inspector-modal');
    var overlay = document.getElementById('part-inspector-overlay');
    if (!modal || !overlay) return;

    var parts = (typeof currentSetParts !== 'undefined' && currentSetParts) ? currentSetParts : (window.currentSetParts || []);
    var p = parts.find(function(item) { return item.ref === partRef; });
    var partName = p ? p.name : 'Pieza ' + partRef;
    var colorCode = p ? p.colorCode || '15' : '15';
    var colorName = p ? p.colorName || 'Desconocido' : 'Desconocido';
    var colorHex = p ? p.color || '#FFFFFF' : '#FFFFFF';

    document.getElementById('inspector-part-name').textContent = partName;
    document.getElementById('inspector-part-ref').textContent = 'Referencia: ' + partRef;
    
    var colorBadge = document.getElementById('inspector-color-badge');
    if (colorBadge) colorBadge.style.backgroundColor = colorHex;
    
    var colorText = document.getElementById('inspector-color-text');
    if (colorText) colorText.textContent = colorName + ' (Código: ' + colorCode + ')';

    var blPageUrl = buildBLPageUrl(partRef);
    var blUrlEl = document.getElementById('inspector-bricklink-url');
    if (blUrlEl) blUrlEl.href = blPageUrl;

    var blImgUrl = buildBLImageUrl(partRef, colorCode);
    var wrap = document.getElementById('inspector-bricklink-wrap');
    if (wrap) {
        wrap.innerHTML = '<img src="' + blImgUrl + '" alt="BrickLink Referencia" style="max-width:100%; max-height:120px; object-fit:contain;" onerror="this.outerHTML=\'<div class=\\\'drawer-img-placeholder\\\'>Sin imagen de BrickLink</div>\'">';
    }

    var grid = document.getElementById('inspector-crops-grid');
    if (grid) {
        var instances = (window.sessionDetectionInstances || {})[partRef] || [];
        if (instances.length === 0) {
            grid.innerHTML = '<div style="color: var(--text-secondary); font-size: 0.85rem; padding: 12px; text-align: center; grid-column: 1 / -1;">No se han detectado instancias en esta sesión.</div>';
        } else {
            var gridHtml = '';
            instances.forEach(function(inst, idx) {
                var scoreVal = Math.round((inst.dinoScore || 0) * 100);
                var imgHtml = inst.cropDataUrl
                    ? '<img src="' + inst.cropDataUrl + '" alt="Crop">'
                    : '<div class="drawer-img-placeholder" style="width:100px; height:100px;">Sin img</div>';
                
                gridHtml += '<div class="inspector-crop-card">' +
                    imgHtml +
                    '<div class="inspector-crop-score">Sim: ' + scoreVal + '%</div>' +
                    '</div>';
            });
            grid.innerHTML = gridHtml;
        }
    }

    overlay.style.display = 'block';
    modal.style.display = 'flex';
    setTimeout(function() {
        overlay.classList.add('visible');
        modal.classList.add('visible');
    }, 10);
}

function closePartInspector() {
    var modal = document.getElementById('part-inspector-modal');
    var overlay = document.getElementById('part-inspector-overlay');
    if (!modal || !overlay) return;

    modal.classList.remove('visible');
    overlay.classList.remove('visible');
    setTimeout(function() {
        modal.style.display = 'none';
        overlay.style.display = 'none';
    }, 250);
}

// Configurar los event listeners cuando se cargue el DOM
document.addEventListener('DOMContentLoaded', function() {
    var closeBtn = document.getElementById('btn-close-inspector');
    var overlay = document.getElementById('part-inspector-overlay');
    if (closeBtn) closeBtn.addEventListener('click', closePartInspector);
    if (overlay) overlay.addEventListener('click', closePartInspector);
});

