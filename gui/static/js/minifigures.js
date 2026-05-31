/**
 * LegoVision Minifigures Module - Three.js 3D viewer
 */

let mfScene=null, mfCamera=null, mfRenderer=null, mfControls=null;
let mfAnimFrame=null, currentMinifigRef=null;

document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("nav-minifigures").addEventListener("click", loadMinifigList);
    document.getElementById("btn-load-minifig").addEventListener("click", function() {
        const ref = document.getElementById("minifig-select").value;
        if (!ref) return;
        currentMinifigRef = ref;
        loadMinifigComponents(ref);
    });
    document.getElementById("btn-assemble-minifig").addEventListener("click", function() {
        if (currentMinifigRef) triggerAssembly(currentMinifigRef);
    });
});

async function loadMinifigList() {
    const sel = document.getElementById("minifig-select");
    try {
        const r = await pywebview.api.get_minifigures_for_sets();
        if (r.status !== "success" || !r.minifigures.length) {
            sel.innerHTML = "<option value=''>Sin minifiguras disponibles</option>";
            return;
        }
        sel.innerHTML = r.minifigures.map(m =>
            "<option value='" + m.ref + "'>" + m.ref + " - " + m.name + " (Set " + m.set_id + ")</option>"
        ).join("");
        currentMinifigRef = r.minifigures[0].ref;
        // Auto-load components and check if 3D model exists
        loadMinifigComponents(currentMinifigRef);
    } catch(e) {
        console.error("[minifigures]", e);
        sel.innerHTML = "<option value=''>Error al cargar</option>";
    }
}

async function loadMinifigComponents(ref) {
    const panel = document.getElementById("minifig-components-panel");
    const grid  = document.getElementById("minifig-components-grid");
    const hdr   = document.getElementById("minifig-name-header");
    const badge = document.getElementById("minifig-status-badge");
    grid.innerHTML = "<div style='color:var(--text-secondary);padding:12px;'>Cargando...</div>";
    panel.style.display = "block";
    try {
        const r = await pywebview.api.get_minifig_components(ref);
        if (r.status !== "success") {
            grid.innerHTML = "<div style='color:#f87171;'>" + r.message + "</div>";
            return;
        }
        hdr.textContent = "Componentes - " + r.name;
        badge.textContent = r.name;
        badge.className = "piece-conf mid";
        grid.innerHTML = r.components.map(function(c) {
            var img = c.image
                ? "<img src='" + c.image + "' style='width:100%;height:80px;object-fit:contain;background:#1e293b;border-radius:6px;border:1px solid var(--border);'>"
                : "<div style='width:100%;height:80px;background:" + c.color_hex + "33;border-radius:6px;border:2px solid " + c.color_hex + ";display:flex;align-items:center;justify-content:center;font-size:0.68rem;color:var(--text-secondary);'>" + c.part_file + "</div>";
            return "<div style='background:rgba(15,23,42,0.6);border:1px solid var(--border);border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:8px;align-items:center;'>" +
                img +
                "<div style='text-align:center;'>" +
                "<div style='font-size:0.78rem;font-weight:600;color:#fff;'>" + c.label + "</div>" +
                "<div style='font-size:0.68rem;color:var(--text-secondary);'>" + c.part_file + "</div>" +
                "<div style='display:flex;align-items:center;gap:5px;justify-content:center;margin-top:4px;'>" +
                "<span style='width:11px;height:11px;border-radius:50%;background:" + c.color_hex + ";border:1px solid rgba(255,255,255,0.25);display:inline-block;'></span>" +
                "<span style='font-size:0.68rem;color:var(--text-secondary);'>" + c.color_name + "</span>" +
                "</div></div></div>";
        }).join("");
        document.getElementById("btn-assemble-minifig").disabled = false;
        checkAssemblyStatus(ref);
    } catch(e) {
        console.error("[minifigures]", e);
        grid.innerHTML = "<div style='color:#f87171;'>Error de conexion</div>";
    }
}

async function checkAssemblyStatus(ref) {
    try {
        const r = await pywebview.api.get_minifig_assembly_status(ref);
        if (r.status !== "success") return;
        var vp = document.getElementById("minifig-viewer-panel");
        var db = document.getElementById("minifig-db-badge");
        var btn = document.getElementById("btn-assemble-minifig");
        if (r.assembled && r.glb_url) {
            vp.style.display = "flex";
            if (r.in_db) db.style.display = "block";
            btn.textContent = "Reensamblar";
            init3DViewer(r.glb_url);
        } else {
            vp.style.display = "none";
            btn.textContent = "Ensamblar 3D";
        }
    } catch(e) { console.error("[minifigures]", e); }
}

async function triggerAssembly(ref) {
    var sd = document.getElementById("minifig-assemble-status");
    var ms = document.getElementById("minifig-assemble-msg");
    var btn = document.getElementById("btn-assemble-minifig");
    sd.style.display = "block";
    btn.disabled = true;
    ms.textContent = "Ensamblando con Blender... Puede tomar 1-3 minutos.";
    try {
        await pywebview.api.assemble_minifig(ref);
        var poll = setInterval(async function() {
            var s = await pywebview.api.get_minifig_assembly_status(ref);
            if (s.assembled) {
                clearInterval(poll);
                sd.style.display = "none";
                btn.disabled = false;
                btn.textContent = "Reensamblar";
                document.getElementById("minifig-viewer-panel").style.display = "flex";
                if (s.in_db) document.getElementById("minifig-db-badge").style.display = "block";
                init3DViewer(s.glb_url);
            }
        }, 5000);
        setTimeout(function() { clearInterval(poll); sd.style.display="none"; btn.disabled=false; }, 300000);
    } catch(e) {
        console.error("[minifigures]", e);
        sd.style.display = "none";
        btn.disabled = false;
    }
}

function init3DViewer(glbUrl) {
    var container = document.getElementById("minifig-3d-container");
    var canvas    = document.getElementById("minifig-3d-canvas");
    var loading   = document.getElementById("minifig-3d-loading");
    if (!container || !canvas) return;
    if (typeof THREE === "undefined") {
        loading.innerHTML = "<span style='color:#f87171;'>Three.js no disponible.</span>";
        return;
    }
    loading.style.display = "flex";
    loading.innerHTML = "<span>Cargando modelo 3D...</span>";
    if (mfAnimFrame) { cancelAnimationFrame(mfAnimFrame); mfAnimFrame=null; }
    if (mfRenderer)  { mfRenderer.dispose(); mfRenderer=null; }

    var w = container.clientWidth  || 800;
    var h = container.clientHeight || 480;

    mfScene = new THREE.Scene();
    mfScene.background = new THREE.Color(0x0f172a);

    mfCamera = new THREE.PerspectiveCamera(45, w/h, 0.1, 1000);
    mfCamera.position.set(0, 2, 6);

    mfRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    mfRenderer.setSize(w, h);
    mfRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mfRenderer.shadowMap.enabled = true;

    mfScene.add(new THREE.AmbientLight(0xffffff, 0.6));
    var dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(5, 10, 5);
    dir.castShadow = true;
    mfScene.add(dir);
    var fill = new THREE.DirectionalLight(0x87ceeb, 0.4);
    fill.position.set(-5, 3, -5);
    mfScene.add(fill);
    mfScene.add(new THREE.GridHelper(10, 20, 0x1e3a5f, 0x0f172a));

    if (typeof THREE.OrbitControls !== "undefined") {
        mfControls = new THREE.OrbitControls(mfCamera, mfRenderer.domElement);
        mfControls.enableDamping = true;
        mfControls.dampingFactor = 0.08;
        mfControls.minDistance = 1;
        mfControls.maxDistance = 20;
        mfControls.target.set(0, 1, 0);
        mfControls.update();
    }

    if (typeof THREE.GLTFLoader !== "undefined") {
        var loader = new THREE.GLTFLoader();
        loader.load(
            glbUrl,
            function(gltf) {
                loading.style.display = "none";
                var model = gltf.scene;
                var box = new THREE.Box3().setFromObject(model);
                var center = box.getCenter(new THREE.Vector3());
                var size   = box.getSize(new THREE.Vector3());
                model.position.sub(center);
                model.position.y += size.y / 2;
                var maxDim = Math.max(size.x, size.y, size.z);
                if (maxDim > 0) model.scale.setScalar(3 / maxDim);
                model.traverse(function(child) {
                    if (child.isMesh) { child.castShadow=true; child.receiveShadow=true; }
                });
                mfScene.add(model);
                mfCamera.position.set(0, 2.5, 5);
                if (mfControls) { mfControls.target.set(0,1,0); mfControls.update(); }
            },
            function(xhr) {
                var pct = xhr.total > 0 ? Math.round((xhr.loaded/xhr.total)*100) + "%" : "...";
                loading.innerHTML = "<span>Cargando: " + pct + "</span>";
            },
            function(err) {
                loading.innerHTML = "<span style='color:#f87171;'>Error: " + (err.message||err) + "</span>";
                console.error("[minifigures GLB]", err);
            }
        );
    } else {
        loading.style.display = "none";
        var mesh = new THREE.Mesh(
            new THREE.BoxGeometry(1,2,0.5),
            new THREE.MeshPhongMaterial({color:0xffffff})
        );
        mesh.position.y = 1;
        mfScene.add(mesh);
    }

    function animate() {
        mfAnimFrame = requestAnimationFrame(animate);
        if (mfControls) mfControls.update();
        mfRenderer.render(mfScene, mfCamera);
    }
    animate();

    window.addEventListener("resize", function() {
        if (!mfRenderer || !container) return;
        var nw = container.clientWidth;
        var nh = container.clientHeight;
        mfCamera.aspect = nw/nh;
        mfCamera.updateProjectionMatrix();
        mfRenderer.setSize(nw, nh);
    });
}
