// =============================================================
// LegoVision — Seccion de Fisica de Piezas y Caida en Cinta
// =============================================================

document.addEventListener("DOMContentLoaded", () => {
    initPhysicsView();
});

function initPhysicsView() {
    const btnSimulate = document.getElementById("btn-simulate-physics");
    if (!btnSimulate) return;

    btnSimulate.addEventListener("click", async () => {
        const partSelect = document.getElementById("physics-part-select").value;
        const colorSelect = document.getElementById("physics-color-select").value;
        const partCustom = document.getElementById("physics-part-custom").value.trim();
        
        // Determinar referencia final (dar prioridad a la entrada manual si existe)
        const partRef = partCustom ? partCustom : partSelect;
        const colorHex = colorSelect;
        
        if (!partRef) {
            alert("Por favor, ingresa o selecciona una referencia de pieza LEGO.");
            return;
        }

        // 1. Configurar interfaz para estado de carga
        setPhysicsLoadingState(true);
        updatePhysicsStatusBadge("Simulando...", "low");

        // 2. Ocultar panel general y colocar placeholder de carga en el carrusel
        const carouselPanel = document.getElementById("physics-carousel-panel");
        const carouselTrack = document.getElementById("physics-carousel-track");
        if (carouselTrack) {
            carouselTrack.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; padding:40px; color:var(--text-secondary);">
                    <div class="drawer-spinner" style="width:40px; height:40px; margin-bottom:12px;"></div>
                    <span style="font-weight: 500; color: #fff;">Simulando caídas y renderizando planos a Z=10 cm...</span>
                    <span style="font-size:0.85rem; color:var(--text-secondary); margin-top:4px;">Se simulan 15 inclinaciones 3D de la pieza sobre cinta de goma</span>
                </div>
            `;
        }

        try {
            // Comprobar que pywebview esté listo
            if (typeof pywebview === "undefined" || !pywebview.api) {
                throw new Error("El puente API de Python (PyWebView) no está disponible.");
            }

            console.log(`[Physics UX] Enviando simulación física para ${partRef} con color ${colorHex}`);
            
            // 3. Ejecutar simulación a través del puente Python
            const result = await pywebview.api.simulate_physics_scatter(partRef, colorHex);
            
            console.log("[Physics UX] Respuesta recibida:", result);

            if (result.status === "success") {
                const timestamp = new Date().getTime();
                
                // 4. Cargar y mostrar carrusel de recortes de validación
                if (carouselPanel && carouselTrack) {
                    carouselTrack.innerHTML = "";
                    if (result.crops && result.crops.length > 0) {
                        result.crops.forEach((cropUrl, idx) => {
                            const card = document.createElement("div");
                            card.className = "carousel-crop-card";
                            card.style.cssText = "flex: 0 0 auto; display: flex; flex-direction: column; align-items: center; background: rgba(30, 41, 59, 0.7); border: 1px solid var(--border); border-radius: 12px; padding: 10px; gap: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); transition: transform 0.25s ease, border-color 0.25s ease;";
                            
                            // Contenedor de la imagen con overflow oculto para zoom contenido
                            const imgWrapper = document.createElement("div");
                            imgWrapper.style.cssText = "width: 150px; height: 150px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); overflow: hidden; position: relative; background: rgba(0,0,0,0.2);";

                            const img = document.createElement("img");
                            img.src = `${cropUrl}?t=${timestamp}`;
                            img.alt = `Caída ${idx + 1}`;
                            img.style.cssText = "width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s cubic-bezier(0.25, 1, 0.5, 1); transform-origin: center;";
                            
                            imgWrapper.appendChild(img);

                            // Efecto de interacción al pasar el cursor (sin zoom en imagen)
                            card.addEventListener("mouseenter", () => {
                                card.style.transform = "translateY(-4px)";
                                card.style.borderColor = "var(--accent)";
                            });
                            card.addEventListener("mouseleave", () => {
                                card.style.transform = "translateY(0)";
                                card.style.borderColor = "var(--border)";
                            });

                            const label = document.createElement("span");
                            label.innerText = `Ángulo Caída ${idx + 1}`;
                            label.style.cssText = "font-size: 0.8rem; color: var(--text-secondary); font-family: Outfit, sans-serif; font-weight: 500;";
                            
                            card.appendChild(imgWrapper);
                            card.appendChild(label);
                            carouselTrack.appendChild(card);
                        });
                        carouselPanel.style.display = "flex";
                    }
                }
                
                updatePhysicsStatusBadge("Completado", "high");
            } else {
                updatePhysicsStatusBadge("Error", "low");
                if (carouselTrack) {
                    carouselTrack.innerHTML = `
                        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; padding:40px; color:var(--accent-red);">
                            <span style="font-size: 2.5rem; display: block; margin-bottom: 12px;">⚠️</span>
                            <strong style="display:block; margin-bottom:6px;">Error en Simulación</strong>
                            <span style="font-size:0.85rem; max-width:400px; text-align:center; color:var(--text-secondary);">${result.message}</span>
                        </div>
                    `;
                }
            }
        } catch (error) {
            console.error("[Physics UX Error]", error);
            updatePhysicsStatusBadge("Error", "low");
            if (carouselTrack) {
                carouselTrack.innerHTML = `
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; padding:40px; color:var(--accent-red);">
                        <span style="font-size: 2.5rem; display: block; margin-bottom: 12px;">❌</span>
                        <strong style="display:block; margin-bottom:6px;">Fallo de Comunicación</strong>
                        <span style="font-size:0.85rem; max-width:400px; text-align:center; color:var(--text-secondary);">${error.message}</span>
                    </div>
                `;
            }
        } finally {
            // Terminar estado de carga
            setPhysicsLoadingState(false);
        }
    });
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
