// =============================================================
// LegoVision — Seccion de Fisica de Piezas y Caida en Cinta
// =============================================================

document.addEventListener("DOMContentLoaded", () => {
    initPhysicsView();
});

// Store crops array globally for slider access
let _physicsCrops = [];
let _physicsTimestamp = null;

function initPhysicsView() {
    const partSelect = document.getElementById("physics-part-select");
    const colorSelect = document.getElementById("physics-color-select");
    const btnSimulate = document.getElementById("btn-simulate-physics");
    if (!btnSimulate) return;

    // Poblar dinámicamente las piezas del set 75078-1 si están disponibles
    if (partSelect && typeof SET_75078_1_PARTS !== "undefined") {
        partSelect.innerHTML = "";
        
        // Evitar duplicados agrupando por referencia
        const seenRefs = new Set();
        SET_75078_1_PARTS.forEach(part => {
            if (!seenRefs.has(part.ref)) {
                seenRefs.add(part.ref);
                const opt = document.createElement("option");
                opt.value = part.ref;
                opt.dataset.color = part.color || "#A0A5A9";
                opt.dataset.colorName = part.colorName || "Gris";
                opt.innerText = `${part.ref} - ${part.name} (${part.colorName || "Gris"})`;
                partSelect.appendChild(opt);
            }
        });

        // Actualizar y deshabilitar el color asociado al cambiar de pieza
        const updateSelectedColor = () => {
            const selectedOpt = partSelect.options[partSelect.selectedIndex];
            if (selectedOpt && colorSelect) {
                const colorHex = selectedOpt.dataset.color;
                const colorName = selectedOpt.dataset.colorName;
                colorSelect.innerHTML = `<option value="${colorHex}" selected>${colorName} (${colorHex})</option>`;
                colorSelect.value = colorHex;
                colorSelect.disabled = true;
            }
        };

        partSelect.addEventListener("change", updateSelectedColor);
        updateSelectedColor(); // Ejecutar inicialmente
    }

    // Slider vertical — actualizar imagen activa al mover
    const rotationSlider = document.getElementById("physics-rotation-slider");
    if (rotationSlider) {
        rotationSlider.addEventListener("input", () => {
            const idx = parseInt(rotationSlider.value);
            showPhysicsCropAtIndex(idx);
        });
    }

    btnSimulate.addEventListener("click", async () => {
        const partRef = partSelect ? partSelect.value : "";
        const colorHex = colorSelect ? colorSelect.value : "#A0A5A9";
        const simCountInput = document.getElementById("physics-simulations");
        
        let numSimulations = 15;
        if (simCountInput) {
            const val = parseInt(simCountInput.value);
            if (!isNaN(val) && val >= 1 && val <= 100) {
                numSimulations = val;
            } else {
                alert("Por favor, ingresa un número de simulaciones válido entre 1 y 100.");
                return;
            }
        }
        
        if (!partRef) {
            alert("Por favor, selecciona una referencia de pieza LEGO.");
            return;
        }

        // 1. Configurar interfaz para estado de carga
        setPhysicsLoadingState(true);
        updatePhysicsStatusBadge("Simulando...", "low");
        
        // Ocultar estadísticas de tiempo previas
        const timeStatsPanel = document.getElementById("physics-time-stats");
        if (timeStatsPanel) timeStatsPanel.style.display = "none";

        // 2. Mostrar estado de carga en el área del visor
        const placeholder = document.getElementById("physics-carousel-placeholder");
        const activeBox = document.getElementById("physics-carousel-active");
        const sliderWrap = document.getElementById("physics-slider-wrap");

        if (placeholder) {
            placeholder.style.display = "flex";
            placeholder.innerHTML = `
                <div class="drawer-spinner" style="width:40px; height:40px; margin-bottom:12px;"></div>
                <span style="font-weight:500; color:#fff;">Simulando caídas y renderizando planos a Z=10 cm...</span>
                <span style="font-size:0.85rem; color:var(--text-secondary); margin-top:4px;">${numSimulations} caídas 3D sobre la cinta de goma</span>
            `;
        }
        if (activeBox) activeBox.style.display = "none";
        if (sliderWrap) sliderWrap.style.display = "none";

        try {
            // Comprobar que pywebview esté listo
            if (typeof pywebview === "undefined" || !pywebview.api) {
                throw new Error("El puente API de Python (PyWebView) no está disponible.");
            }

            console.log(`[Physics UX] Enviando simulación física para ${partRef} con color ${colorHex} (${numSimulations} runs)`);
            
            // 3. Ejecutar simulación a través del puente Python
            const result = await pywebview.api.simulate_physics_scatter(partRef, colorHex, numSimulations);
            
            console.log("[Physics UX] Respuesta recibida:", result);

            if (result.status === "success") {
                _physicsTimestamp = new Date().getTime();
                _physicsCrops = result.crops || [];

                // Mostrar estadísticas de tiempo de cómputo
                if (timeStatsPanel && typeof result.total_physics_time !== "undefined") {
                    document.getElementById("physics-time-total").innerText = `${result.total_physics_time.toFixed(2)}s`;
                    document.getElementById("physics-time-per-piece").innerText = `${result.physics_time_per_piece.toFixed(2)}s`;
                    document.getElementById("render-time-total").innerText = `${result.total_render_time.toFixed(2)}s`;
                    document.getElementById("render-time-per-piece").innerText = `${result.render_time_per_piece.toFixed(2)}s`;
                    timeStatsPanel.style.display = "flex";
                }

                // 4. Configurar slider y mostrar primera imagen
                if (_physicsCrops.length > 0) {
                    // Ocultar placeholder
                    if (placeholder) placeholder.style.display = "none";

                    // Configurar slider vertical
                    if (rotationSlider && sliderWrap) {
                        rotationSlider.min = 0;
                        rotationSlider.max = _physicsCrops.length - 1;
                        rotationSlider.value = 0;
                        sliderWrap.style.display = "flex";
                    }

                    // Mostrar primera imagen centrada
                    showPhysicsCropAtIndex(0);
                    if (activeBox) activeBox.style.display = "flex";

                    // Actualizar también el track de miniaturas (oculto, para compatibilidad)
                    const carouselTrack = document.getElementById("physics-carousel-track");
                    if (carouselTrack) {
                        carouselTrack.innerHTML = "";
                        _physicsCrops.forEach((cropUrl, idx) => {
                            const img = document.createElement("img");
                            img.src = `${cropUrl}?t=${_physicsTimestamp}`;
                            img.alt = `Rot ${idx + 1}`;
                            img.style.cssText = "width:64px; height:64px; object-fit:contain; border-radius:8px; border:1px solid var(--border); background:rgba(0,0,0,0.2); cursor:pointer;";
                            img.addEventListener("click", () => {
                                if (rotationSlider) rotationSlider.value = idx;
                                showPhysicsCropAtIndex(idx);
                            });
                            carouselTrack.appendChild(img);
                        });
                        // Mostrar miniaturas si hay múltiples
                        if (_physicsCrops.length > 1) {
                            carouselTrack.style.display = "flex";
                        }
                    }
                }
                
                updatePhysicsStatusBadge("Completado", "high");
            } else {
                updatePhysicsStatusBadge("Error", "low");
                showPhysicsError(result.message || "Error desconocido en simulación.");
            }
        } catch (error) {
            console.error("[Physics UX Error]", error);
            updatePhysicsStatusBadge("Error", "low");
            showPhysicsError(error.message || "Fallo de comunicación con el backend.");
        } finally {
            setPhysicsLoadingState(false);
        }
    });
}

/**
 * Muestra la imagen del crop en el índice dado en el visor central.
 */
function showPhysicsCropAtIndex(idx) {
    if (!_physicsCrops || _physicsCrops.length === 0) return;
    const clampedIdx = Math.max(0, Math.min(idx, _physicsCrops.length - 1));
    
    const activeImg = document.getElementById("physics-carousel-active-img");
    const activeLabel = document.getElementById("physics-carousel-active-label");
    const sliderLabel = document.getElementById("physics-slider-label");
    const rotationSlider = document.getElementById("physics-rotation-slider");

    if (activeImg) {
        activeImg.src = `${_physicsCrops[clampedIdx]}?t=${_physicsTimestamp}`;
        activeImg.alt = `Rotación ${clampedIdx + 1} de ${_physicsCrops.length}`;
    }
    if (activeLabel) {
        activeLabel.textContent = `Caída estabilizada ${clampedIdx + 1} de ${_physicsCrops.length}`;
    }
    if (sliderLabel) {
        sliderLabel.textContent = `${clampedIdx + 1}/${_physicsCrops.length}`;
    }
    if (rotationSlider) {
        rotationSlider.value = clampedIdx;
    }
}

/**
 * Muestra un mensaje de error en el área del visor.
 */
function showPhysicsError(message) {
    const placeholder = document.getElementById("physics-carousel-placeholder");
    const activeBox = document.getElementById("physics-carousel-active");
    const sliderWrap = document.getElementById("physics-slider-wrap");

    if (activeBox) activeBox.style.display = "none";
    if (sliderWrap) sliderWrap.style.display = "none";
    if (placeholder) {
        placeholder.style.display = "flex";
        placeholder.innerHTML = `
            <span style="font-size:2.5rem; display:block; margin-bottom:12px;">⚠️</span>
            <strong style="display:block; margin-bottom:6px; color:var(--accent-red);">Error en Simulación</strong>
            <span style="font-size:0.85rem; max-width:400px; text-align:center; color:var(--text-secondary);">${message}</span>
        `;
    }
}

/**
 * Activa o desactiva la barra de progreso y bloquea el botón durante la simulación.
 */
function setPhysicsLoadingState(isLoading) {
    const btnSimulate = document.getElementById("btn-simulate-physics");
    const progressPanel = document.getElementById("physics-progress-panel");
    
    if (isLoading) {
        btnSimulate.disabled = true;
        btnSimulate.innerHTML = "⏳ Simulando...";
        progressPanel.style.display = "block";
    } else {
        btnSimulate.disabled = false;
        btnSimulate.innerHTML = "⚡ Simular Caída y Renderizar";
        progressPanel.style.display = "none";
    }
}

/**
 * Actualiza el indicador visual de estado.
 * @param {string} text Texto del estado.
 * @param {string} type Tipo de clase de confianza ("high" = verde, "low" = rojo).
 */
function updatePhysicsStatusBadge(text, type) {
    const badge = document.getElementById("physics-status-badge");
    if (!badge) return;
    badge.innerText = text;
    badge.className = `piece-conf ${type}`;
}