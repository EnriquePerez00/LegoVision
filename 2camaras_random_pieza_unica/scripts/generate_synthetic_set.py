import os
import sys
import re
import random
import math
import json

try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# Rutas estándar de Studio 2.0 en macOS
STUDIO_LDRAW_PATH = "/Applications/Studio 2.0/ldraw"
STUDIO_PARTS_DIR = os.path.join(STUDIO_LDRAW_PATH, "parts")

def get_ldraw_part_path(part_ref):
    """Busca la referencia de la pieza en la biblioteca local de Studio o local del proyecto (incluyendo directorios Unofficial)."""
    # 1. Definir los directorios base a buscar
    ldraw_roots = []
    
    # Directorio local del proyecto (data/ldraw)
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_ldraw = os.path.join(project_root_dir, "data", "ldraw")
    if os.path.exists(project_ldraw):
        ldraw_roots.append(project_ldraw)
        
    # Directorio de Studio 2.0 (macOS)
    if os.path.exists("/Applications/Studio 2.0/ldraw"):
        ldraw_roots.append("/Applications/Studio 2.0/ldraw")
        
    # 2. Definir los subdirectorios relativos dentro de cada raíz de LDraw, priorizando Unofficial
    subdirs = [
        "UnOfficial/parts",
        "UnOfficial/p",
        "Unofficial/parts",
        "Unofficial/p",
        "parts",
        "p",
        "parts/s",
        "parts/s/Unofficial",
        "parts/s/UnOfficial"
    ]
    
    exact_file = f"{part_ref}.dat"
    
    # 3. Buscar en cada raíz y cada subdirectorio
    for root in ldraw_roots:
        for subdir in subdirs:
            parts_dir = os.path.join(root, subdir)
            if not os.path.exists(parts_dir):
                continue
                
            # Coincidencia exacta
            exact_path = os.path.join(parts_dir, exact_file)
            if os.path.exists(exact_path):
                return exact_path
                
            # Mapeos especiales
            if part_ref == "4589b":
                fallback_path = os.path.join(parts_dir, "4589.dat")
                if os.path.exists(fallback_path):
                    return fallback_path
                    
            # Búsqueda insensible a mayúsculas/minúsculas
            try:
                for f in os.listdir(parts_dir):
                    if f.lower() == exact_file.lower():
                        return os.path.join(parts_dir, f)
            except Exception:
                pass
                    
    return None

def setup_physics_world():
    """Configura el entorno de físicas (Rigid Body World) en la escena de Blender."""
    if not IN_BLENDER:
        return
        
    scene = bpy.context.scene
    scene.use_gravity = True
    scene.gravity = (0.0, 0.0, -9.81) # Gravedad terrestre estándar en Z negativo
    
    # Crear o recuperar Rigid Body World
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
        
    rb_world = scene.rigidbody_world
    rb_world.substeps_per_frame = 10
    rb_world.solver_iterations = 10
    rb_world.time_scale = 1.0

def create_conveyor_belt_collider():
    """Crea el plano que actúa como la superficie de la cinta transportadora.
    
    Cross-Polarization (2026-06-13):
      Material de la cinta con Specular=0.0 y Roughness=1.0 (mate perfecto)
      para no reflejar el Dome Light hacia las cámaras.
    """
    if not IN_BLENDER:
        return None
    
    # Importar parámetros de cross-polarization desde scene_config
    try:
        from scene_config import BELT_SPECULAR, BELT_ROUGHNESS, BELT_COLOR_LINEAR
    except ImportError:
        BELT_SPECULAR = 0.0
        BELT_ROUGHNESS = 1.0
        BELT_COLOR_LINEAR = (0.145, 0.255, 0.33, 1.0)
        
    # Eliminar plano anterior si existe
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(True)
        bpy.ops.object.delete()
        
    # Crear plano
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Conveyor_Belt_Plane"
    
    # Configurar cuerpo rígido pasivo
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    plane.rigid_body.type = 'PASSIVE'
    plane.rigid_body.collision_shape = 'BOX'
    plane.rigid_body.friction = 0.6       # Fricción para simular goma/PVC
    plane.rigid_body.restitution = 0.1    # Bajo rebote para estabilidad
    plane.rigid_body.kinematic = True      # Permitir que la animación de la cinta afecte a las físicas
    
    # Configurar el material de la cinta a Azul Petróleo Claro con cross-polarization
    plane.is_shadow_catcher = True
    mat_name = "Light_Petrol_Blue_Belt"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        # Color azul petróleo canónico (NO modificar para CIELAB)
        principled.inputs['Base Color'].default_value = BELT_COLOR_LINEAR
        # Cross-polarization: mate perfecto para no reflejar Dome Light
        principled.inputs['Roughness'].default_value = BELT_ROUGHNESS  # 1.0
        # Specular = 0.0 (sin brillo especular)
        if 'Specular IOR Level' in principled.inputs:
            principled.inputs['Specular IOR Level'].default_value = BELT_SPECULAR  # 0.0
        elif 'Specular' in principled.inputs:
            principled.inputs['Specular'].default_value = BELT_SPECULAR  # 0.0
    plane.data.materials.clear()
    plane.data.materials.append(mat)
    
    # Animación inicial y final para crear un Action y animar los canales
    plane.location = (0.0, 0.0, 0.0)
    plane.rotation_euler = (0.0, 0.0, 0.0)
    plane.keyframe_insert(data_path="location", frame=1)
    plane.keyframe_insert(data_path="rotation_euler", frame=1)
    plane.keyframe_insert(data_path="location", frame=120)
    plane.keyframe_insert(data_path="rotation_euler", frame=120)
    
    # Añadir F-Curve Noise Modifiers para simular vibración/cinta rugosa en movimiento
    if plane.animation_data and plane.animation_data.action:
        action = plane.animation_data.action
        fcurves = []
        
        # Compatibilidad con Blender 5.x (layered slots)
        if hasattr(action, "layers") and len(action.layers) > 0:
            try:
                layer = action.layers[0]
                if len(layer.strips) > 0:
                    strip = layer.strips[0]
                    slot = action.slots[0] if (hasattr(action, "slots") and len(action.slots) > 0) else None
                    bag = strip.channelbag(slot=slot) if slot else strip.channelbag()
                    if hasattr(bag, "fcurves"):
                        fcurves = bag.fcurves
            except Exception as e:
                print(f"Error al obtener fcurves en modo estructurado 5.x: {e}")
                
        # Compatibilidad con Blender 4.x / Legacy
        if not fcurves and hasattr(action, "fcurves"):
            fcurves = action.fcurves
            
        for fcurve in fcurves:
            try:
                noise_mod = fcurve.modifiers.new(type='NOISE')
                if "location" in fcurve.data_path:
                    noise_mod.strength = 0.04  # Amplitud de vibración (aprox. 4cm)
                    noise_mod.scale = 2.0      # Alta frecuencia
                elif "rotation_euler" in fcurve.data_path:
                    noise_mod.strength = 0.06  # Ruido angular sutil (aprox. 3.4 grados)
                    noise_mod.scale = 1.5
            except Exception as e:
                print(f"Error al aplicar Noise a fcurve: {e}")
                
    return plane

def apply_rigid_body_physics(obj, mass=0.008):
    """Convierte el objeto de pieza de Lego en un cuerpo rígido activo."""
    if not IN_BLENDER or not obj:
        return
        
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Añadir física
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    obj.rigid_body.type = 'ACTIVE'
    obj.rigid_body.mass = mass
    obj.rigid_body.friction = 0.4
    obj.rigid_body.restitution = 0.2
    
    # Usar CONVEX_HULL para mejor precisión de colisión
    obj.rigid_body.collision_shape = 'CONVEX_HULL'
    obj.rigid_body.use_deactivation = False
    obj.rigid_body.use_start_deactivated = False

def simulate_drop_to_rest(obj):
    """Ubica el objeto arriba con rotación aleatoria y corre la simulación hasta que repose."""
    if not IN_BLENDER or not obj:
        return
        
    # Posicionar sobre el plano de la cinta
    obj.location = (
        random.uniform(-0.1, 0.1), 
        random.uniform(-0.1, 0.1), 
        0.5 # Altura de caída reducida a 50 cm
    )
    
    # Rotación inicial orientada a caras estables (top o bottom planas, más pequeña inclinación de +-15 grados / ~0.26 rad)
    pitch = random.choice([0.0, math.pi]) + random.uniform(-0.26, 0.26)
    roll = random.choice([0.0, math.pi]) + random.uniform(-0.26, 0.26)
    yaw = random.uniform(0, 2 * math.pi)
    
    obj.rotation_euler = (pitch, roll, yaw)
    
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 120
    
    print(f"Simulando caída física para {obj.name}...")
    
    # Correr la simulación frame a frame hasta que se detenga
    for frame in range(1, 100):
        scene.frame_set(frame)
        # Fuerza la actualización de físicas en Blender background
        bpy.context.view_layer.update()
        
    # Aplicar transformación final de forma visual permanente
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.visual_transform_apply()
    
    # Remover físicas para congelar la pieza y evitar que cambie durante el render
    bpy.ops.rigidbody.object_remove()
    
    # Centrar la cámara en la nueva posición del objeto
    # Ajustar origen al centro de volumen
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    print(f"Objeto asentado en posición: {obj.location}")

    # Reposicionar la cámara y su target Empty para que estén perfectamente alineados
    dx = obj.location.x
    dy = obj.location.y
    dz = obj.location.z
    
    if "Camera_Target" in bpy.data.objects:
        bpy.data.objects["Camera_Target"].location = (dx, dy, dz)
        
    if "Camera" in bpy.data.objects:
        # Encuadre dinámico: calcular la dimensión máxima para ajustar la distancia focal (Z)
        max_dim = max(obj.dimensions) if hasattr(obj, "dimensions") else 1.0
        # Escalado dinámico para que la pieza ocupe el ~70% de la toma
        # Para lentes de 50mm, un factor de 2.2 respecto a la dimensión máxima da un encuadre óptimo
        camera_height = max(0.6, max_dim * 2.2) 
        
        # Colocar la cámara directamente encima del objeto para una vista cenital (top-down)
        bpy.data.objects["Camera"].location = (dx, dy, camera_height + dz)
        print(f"Cámara cenital posicionada a altura: {camera_height:.3f}m para pieza de dimensión máxima: {max_dim:.3f}m")

def setup_studio_lighting():
    """Configura iluminación de estudio de tres puntos de alta calidad."""
    if not IN_BLENDER:
        return
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    
    # Key Light (Luz Principal)
    bpy.ops.object.light_add(type='AREA', location=(3.0, -3.0, 5.0))
    key = bpy.context.active_object
    key.name = "Key_Light"
    key.data.energy = 600.0
    key.data.size = 1.5
    
    # Fill Light (Luz de Relleno)
    bpy.ops.object.light_add(type='AREA', location=(-3.0, -2.0, 3.0))
    fill = bpy.context.active_object
    fill.name = "Fill_Light"
    fill.data.energy = 250.0
    fill.data.size = 2.0
    
    # Rim Light (Luz de Contorno para realzar bordes/studs)
    bpy.ops.object.light_add(type='AREA', location=(0.0, 4.0, 4.0))
    rim = bpy.context.active_object
    rim.name = "Rim_Light"
    rim.data.energy = 400.0
    rim.data.size = 1.2

COLOR_CATALOG = None


# ═════════════════════════════════════════════════════════════════════════
#  Helpers para piezas translúcidas (Trans Red, Trans Brown, Trans Clear, …)
# ═════════════════════════════════════════════════════════════════════════

def _is_eevee_next() -> bool:
    """True si Blender ≥ 4.2 (EEVEE Next con raytracing nativo)."""
    if not IN_BLENDER:
        return False
    try:
        return tuple(bpy.app.version[:2]) >= (4, 2)
    except Exception:
        return False


def configure_eevee_for_translucent(scene=None):
    """Activa SSR + Refraction (o Raytracing en 4.2+) para EEVEE.

    Sólo modifica los flags estrictamente necesarios para que las piezas
    translúcidas refracten correctamente. **NO** altera luces, world, cámara,
    AO, bloom ni samples TAA. Es idempotente y seguro de invocar varias veces.
    """
    if not IN_BLENDER:
        return
    scene = scene or bpy.context.scene
    # Sólo aplica si el motor activo es EEVEE (no Cycles).
    engine = scene.render.engine or ""
    if "EEVEE" not in engine.upper():
        return

    eev = getattr(scene, "eevee", None)
    if eev is None:
        return

    if _is_eevee_next():
        # Blender 4.2+: API unificada bajo "use_raytracing".
        if hasattr(eev, "use_raytracing"):
            eev.use_raytracing = True
        # Algunos builds intermedios mantienen aún "use_ssr_refraction".
        if hasattr(eev, "use_ssr_refraction"):
            eev.use_ssr_refraction = True
    else:
        # Blender 3.x / 4.0 / 4.1: SSR + Refraction clásicos.
        if hasattr(eev, "use_ssr"):
            eev.use_ssr = True
        if hasattr(eev, "use_ssr_refraction"):
            eev.use_ssr_refraction = True


def _set_principled_input(node, names, value):
    """Asigna `value` al primer socket de `node` cuyo nombre esté en `names`.

    Útil para sortear renombrados entre versiones del Principled BSDF
    (e.g. "Transmission" → "Transmission Weight" en 4.x).
    """
    for n in names:
        if n in node.inputs:
            try:
                node.inputs[n].default_value = value
                return True
            except Exception:
                pass
    return False


def _build_translucent_lego_material(mat, rgba):
    """Construye la red de nodos para plástico translúcido LEGO realista.

    Topología (clave para sombras coloreadas en EEVEE/Cycles):

        Principled BSDF ─┐
                         ├─► Mix Shader.Shader[0]
        Transparent BSDF ─► Mix Shader.Shader[1]
        Light Path.Is Shadow Ray ─► Mix Shader.Fac
                         ▼
                  Material Output.Surface

    El Principled gestiona refracción/reflexión para rays primarios,
    y el Transparent BSDF (con el mismo color base) tiñe los rays de
    sombra para que la sombra proyectada herede el tinte de la pieza.
    """
    # rgba viene como [r,g,b,a]; forzamos alpha=1 en Base Color porque la
    # translucidez la maneja Transmission, NO el alpha (evita lavado gris).
    base_color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
    transparent_color = base_color  # mismo tinte para sombras coloreadas

    nt = mat.node_tree
    nt.nodes.clear()
    nodes = nt.nodes
    links = nt.links

    n_principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    n_principled.location = (-300, 200)
    n_principled.inputs["Base Color"].default_value = base_color
    _set_principled_input(n_principled, ("Roughness",), 0.07)
    _set_principled_input(
        n_principled, ("Transmission Weight", "Transmission"), 1.0
    )
    _set_principled_input(n_principled, ("IOR",), 1.58)
    # Metallic 0, Specular PIECE_SPECULAR (cross-polarization), Subsurface Scattering enabled, Alpha=1
    try:
        from scene_config import PIECE_SPECULAR
    except ImportError:
        PIECE_SPECULAR = 0.05
    _set_principled_input(n_principled, ("Specular", "Specular IOR Level"), PIECE_SPECULAR)
    _set_principled_input(n_principled, ("Subsurface", "Subsurface Weight"), 0.2)
    _set_principled_input(n_principled, ("Metallic",), 0.0)
    _set_principled_input(n_principled, ("Alpha",), 1.0)

    n_transparent = nodes.new(type="ShaderNodeBsdfTransparent")
    n_transparent.location = (-300, -100)
    n_transparent.inputs["Color"].default_value = transparent_color

    n_lightpath = nodes.new(type="ShaderNodeLightPath")
    n_lightpath.location = (-300, 500)

    n_mix = nodes.new(type="ShaderNodeMixShader")
    n_mix.location = (50, 200)

    n_output = nodes.new(type="ShaderNodeOutputMaterial")
    n_output.location = (300, 200)

    # Conexiones según especificación.
    links.new(n_principled.outputs["BSDF"], n_mix.inputs[1])
    links.new(n_transparent.outputs["BSDF"], n_mix.inputs[2])
    links.new(n_lightpath.outputs["Is Shadow Ray"], n_mix.inputs["Fac"])
    links.new(n_mix.outputs["Shader"], n_output.inputs["Surface"])

    # Propiedades de superficie del material.
    try:
        mat.blend_method = "HASHED"
    except Exception:
        pass
    try:
        mat.shadow_method = "HASHED"
    except Exception:
        pass
    # Activar refracción en pantalla (EEVEE clásico). En EEVEE Next la
    # propiedad puede no existir; el flag global de raytracing cubre el caso.
    try:
        if hasattr(mat, "use_screen_refraction"):
            mat.use_screen_refraction = True
    except Exception:
        pass

    # Marcador para detección O(1) en escenas multi-material.
    try:
        mat["is_lego_translucent"] = True
    except Exception:
        pass

    return mat


def _material_is_translucent(mat) -> bool:
    """True si el material fue construido por _build_translucent_lego_material."""
    if mat is None:
        return False
    try:
        return bool(mat.get("is_lego_translucent", False))
    except Exception:
        return False


def load_color_catalog():
    global COLOR_CATALOG
    if COLOR_CATALOG is not None:
        return COLOR_CATALOG
    COLOR_CATALOG = {}
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(project_root, "database", "color_catalog.json")
    if not os.path.exists(catalog_path):
        # Fallback to parent of project_root (LegoVision root)
        catalog_path = os.path.join(os.path.dirname(project_root), "database", "color_catalog.json")
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                COLOR_CATALOG = json.load(f)
        except Exception as e:
            print(f"[WARN] Error al cargar catálogo de colores: {e}")
    else:
        print(f"[WARN] color_catalog.json not found at {catalog_path}")
    return COLOR_CATALOG

def create_abs_plastic_material(color_value):
    """Crea un material fotorrealista basado en el catálogo de colores de Studio.
    
    Cross-Polarization Simulation (2026-06-13):
      Para plásticos sólidos, se aplica Specular=0.05 y Roughness=0.75 para
      simular polarización cruzada, eliminando los brillos especulares blancos
      y dejando solo el color difuso (albedo). Esto mejora la precisión del
      color en el espacio CIELAB para Machine Vision.
    
    Tipos de material soportados: solid, transparent, metallic, rubber.
    """
    if not IN_BLENDER:
        return None
    
    # Importar parámetros de cross-polarization desde scene_config
    try:
        from scene_config import PIECE_SPECULAR, PIECE_ROUGHNESS
    except ImportError:
        PIECE_SPECULAR = 0.05
        PIECE_ROUGHNESS = 0.75
        
    catalog = load_color_catalog()
    
    # Resolviendo color_value (puede ser un ID como "36" o un Hex como "#C91A09")
    color_def = None
    color_hex = "#FFFFFF"
    
    if str(color_value).startswith("#"):
        color_hex = color_value
        # Buscar en catálogo por valor hexadecimal
        search_hex = color_hex.lstrip("#").lower()
        for code, info in catalog.items():
            if info.get("hex", "").lstrip("#").lower() == search_hex:
                color_def = info
                break
    else:
        # Buscar por código BrickLink
        str_code = str(color_value)
        if str_code in catalog:
            color_def = catalog[str_code]
            color_hex = color_def.get("hex", "#FFFFFF")
            if not color_hex.startswith("#"):
                color_hex = "#" + color_hex
                
    # Nombre único del material
    mat_name = f"Studio_Mat_{color_value}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
        
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Convertir Hex a RGB
    hex_val = color_hex.lstrip('#')
    rgba = [int(hex_val[i:i+2], 16)/255.0 for i in (0, 2, 4)] + [1.0]

    # ── Detección temprana de plástico translúcido ──────────────────────
    # Si el color es translúcido (Trans Red / Trans Brown / Trans Clear …)
    # delegamos al builder especializado con topología Principled +
    # Transparent BSDF + Light Path + Mix Shader. Esto da:
    #   · refracción real con IOR=1.58 (policarbonato)
    #   · roughness bajo (0.07) y transmission=1
    #   · sombras coloreadas (no grises sólidas)
    #   · blend/shadow=HASHED y use_screen_refraction=True
    if color_def is not None:
        _mat_type_early = color_def.get("material_type", "solid")
        _alpha_early = color_def.get("alpha", 1.0)
        if _mat_type_early == "transparent" or _alpha_early < 1.0:
            return _build_translucent_lego_material(mat, rgba)

    node_principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_principled.location = (0, 0)

    # ══════════════════════════════════════════════════════════════════════
    # CROSS-POLARIZATION SIMULATION
    # ══════════════════════════════════════════════════════════════════════
    # Valores por defecto con cross-polarization para plástico ABS sólido:
    #   - Specular = 0.05 (casi sin brillo especular blanco)
    #   - Roughness = 0.75 (difuso suave, mate)
    # Esto simula el efecto de polarizadores cruzados en fotografía industrial.
    metallic = 0.0
    roughness = PIECE_ROUGHNESS  # 0.75 por defecto (cross-polarization)
    specular = PIECE_SPECULAR    # 0.05 por defecto (cross-polarization)
    transmission = 0.0
    subsurface = 0.0  # Desactivado para mejor fidelidad de color
    
    # Si tenemos definición detallada del catálogo, configurar según tipo
    if color_def:
        mat_type = color_def.get("material_type", "solid")

        if mat_type == "metallic":
            # Metalizado / Cromo - mantiene brillo especular
            metallic = 1.0
            roughness = 0.2
            specular = 0.5  # Metálicos mantienen specular
        elif mat_type == "rubber":
            # Goma / Mate - aún más difuso
            metallic = 0.0
            roughness = 0.95  # Más mate que plástico normal
            specular = 0.0    # Sin specular para goma
            
    # Configurar entradas del Principled BSDF
    node_principled.inputs['Base Color'].default_value = rgba
    node_principled.inputs['Roughness'].default_value = roughness
    
    # Configurar Specular (cross-polarization)
    # El nombre del input varía según versión de Blender
    if 'Specular IOR Level' in node_principled.inputs:
        node_principled.inputs['Specular IOR Level'].default_value = specular
    elif 'Specular' in node_principled.inputs:
        node_principled.inputs['Specular'].default_value = specular
    
    # Configurar metálico
    if 'Metallic' in node_principled.inputs:
        node_principled.inputs['Metallic'].default_value = metallic
        
    # Configurar transmisión (vidrio/cristal) - 0 para sólidos
    if 'Transmission Weight' in node_principled.inputs:
        node_principled.inputs['Transmission Weight'].default_value = transmission
    elif 'Transmission' in node_principled.inputs:
        node_principled.inputs['Transmission'].default_value = transmission
        
    # SSS desactivado para mejor fidelidad de color en cross-polarization
    # (el SSS difumina el color y reduce la precisión CIELAB)
            
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_output.location = (300, 0)
    links.new(node_principled.outputs['BSDF'], node_output.inputs['Surface'])
    
    return mat

def apply_bevel_modifier(obj, width=0.015, segments=3):
    """Aplica biselado sutil para capturar reflejos en los bordes de la pieza (fotorrealismo)."""
    if not IN_BLENDER or not obj:
        return
        
    # Añadir bisel para que los bordes capturen los brillos de estudio
    bevel = obj.modifiers.new(name="ABS_Bevel", type='BEVEL')
    bevel.width = width
    bevel.segments = segments
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit = 0.52 # ~30 grados

def setup_camera():
    """Configura la cámara para enfocar de forma óptima el punto de reposo."""
    if not IN_BLENDER:
        return
        
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add(location=(0, -6.0, 5.5))
        camera = bpy.context.active_object
        camera.name = "Camera"
        
    camera.location = (0.0, 0.0, 3.5)
    
    # Crear target vacío para que la cámara siempre apunte al centro
    if "Camera_Target" not in bpy.data.objects:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0.2))
        target = bpy.context.active_object
        target.name = "Camera_Target"
    else:
        target = bpy.data.objects["Camera_Target"]
        
    target.location = (0, 0, 0.0)
    
    # Limpiar restricciones antiguas
    camera.constraints.clear()
    
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    bpy.context.scene.camera = camera

def _extract_subref_from_obj_name(obj_name):
    """
    Recibe el nombre de un objeto importado por el addon LDraw (típicamente
    el nombre del subfile, ej. '15392.dat' o '15392.dat.001') y devuelve la
    ref BrickLink limpia: '15392'.
    Si no consigue extraer, devuelve el nombre tal cual (lower-case).
    """
    if not obj_name:
        return ""
    name = obj_name.strip()
    # Quitar sufijos numéricos de Blender (".001", ".002", ...)
    name = re.sub(r"\.\d{3,}$", "", name)
    # Quitar extensión .dat (case-insensitive)
    if name.lower().endswith(".dat"):
        name = name[:-4]
    # Algunos addons prefijan el nombre del archivo padre con "Lego_" o usan
    # carpetas en el nombre (s/15391s01) — quedarnos con el último segmento.
    name = name.replace("\\", "/").split("/")[-1]
    return name


def _try_import_color_resolver():
    """Importa scripts.ldraw_color_resolver de forma robusta."""
    import importlib, importlib.util
    # 1) Import normal (PYTHONPATH ya configurado).
    try:
        return importlib.import_module("ldraw_color_resolver")
    except Exception:
        pass
    # 2) Path absoluto al fichero junto a este script.
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "ldraw_color_resolver.py")
    if os.path.isfile(candidate):
        spec = importlib.util.spec_from_file_location(
            "ldraw_color_resolver", candidate
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def render_piece_pipeline(part_ref, color_hex, output_path,
                          set_code=None, color_bl_id=None):
    """Lanza el pipeline completo: carga, físicas, bisel, material y render.

    Parámetros:
      part_ref      ref BrickLink de la pieza (e.g. "15391")
      color_hex     hex del color BL del padre (legacy / fallback de display)
      output_path   ruta del PNG de salida
      set_code      código del set BrickLink (e.g. "75078-1") — opcional pero
                    recomendado para que el resolver pueda leer el inventario.
      color_bl_id   ID BrickLink del color del padre (e.g. "11"). Si se omite,
                    se intenta inferir desde color_hex.
    """
    print(f"\
--- Procesando pieza: {part_ref} (BL color={color_bl_id or color_hex}, set={set_code}) ---")

    part_path = get_ldraw_part_path(part_ref)

    # Si es minifigura (comienza con sw) y no se encuentra la malla, intentar ensamblarla
    if not part_path and part_ref.startswith("sw"):
        try:
            from scripts.assemble_minifig import build_minifig
            print(f"Generando malla de minifigura ensamblada para {part_ref}...")
            build_minifig(part_ref)
            part_path = get_ldraw_part_path(part_ref)
        except Exception as e:
            print(f"No se pudo ensamblar la minifigura {part_ref}: {e}")

    if not part_path:
        print(f"❌ Error: No se encontró la malla para {part_ref} en la ruta de Studio.")
        return False

    print(f"✅ Malla encontrada en: {part_path}")

    if not IN_BLENDER:
        print(f"[Simulación] Físicas de caída simuladas para {part_ref} (color: {color_hex}) -> {output_path}")
        return True

    # ── Resolver de colores BL para piezas compuestas (sólo no-minifigs) ──
    catalog = load_color_catalog()
    parent_bl = str(color_bl_id) if color_bl_id is not None else None
    if parent_bl is None and color_hex:
        # Inferir BL id desde hex.
        sh = str(color_hex).lstrip("#").lower()
        for bl_id, info in catalog.items():
            if info.get("hex", "").lstrip("#").lower() == sh:
                parent_bl = str(bl_id)
                break

    color_map = {}
    if parent_bl is not None and not part_ref.startswith("sw"):
        resolver = _try_import_color_resolver()
        if resolver is not None:
            try:
                color_map = resolver.resolve_subpart_colors(
                    part_ref=part_ref,
                    parent_bl_color=parent_bl,
                    set_code=set_code,
                    color_catalog=catalog,
                    allow_scrape=bool(set_code),
                    verbose=False,
                )
                print(f"   [Resolver] colores por subparte: {color_map}")
            except Exception as e:
                print(f"   [Resolver] fallo no crítico: {e}")
                color_map = {}

    # Limpiar malla antigua
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    if "Conveyor_Belt_Plane" in bpy.data.objects:
        bpy.data.objects["Conveyor_Belt_Plane"].select_set(False)
    bpy.ops.object.delete()

    # Conjunto de objetos pre-existentes para detectar después los importados.
    pre_existing = {o.name for o in bpy.context.scene.objects}

    # Importar malla LDraw
    try:
        bpy.ops.import_scene.importldr(filepath=part_path)
        print("✅ Importado vía Addon LDraw (importldr).")
    except Exception as e:
        print(f"⚠️ Addon LDraw falló o no disponible ({e}). Generando aproximación detallada...")
        generate_detailed_fallback_mesh(part_ref)

    # Identificar TODOS los meshes importados (excluyendo cinta y empties).
    imported_meshes = [
        o for o in bpy.context.scene.objects
        if o.type == 'MESH' and o.name != "Conveyor_Belt_Plane"
        and o.name not in pre_existing
    ]
    if not imported_meshes:
        print("❌ Error al importar/crear el objeto (no hay meshes nuevos).")
        return False

    # ── Aplicar material por subparte usando el color_map del resolver ──
    parent_color_value = parent_bl if parent_bl is not None else color_hex
    for mesh_obj in imported_meshes:
        sub_ref = _extract_subref_from_obj_name(mesh_obj.name)
        # Buscar BL color del subref (case-insensitive parcial: el resolver
        # guarda refs con su capitalización original).
        bl_for_sub = None
        for k, v in color_map.items():
            if k.lower() == sub_ref.lower():
                bl_for_sub = v
                break
        if bl_for_sub is None:
            bl_for_sub = parent_color_value  # fallback al color del padre

        mat = create_abs_plastic_material(bl_for_sub)
        try:
            mesh_obj.data.materials.clear()
            mesh_obj.data.materials.append(mat)
        except Exception as e:
            print(f"   ⚠️ No se pudo aplicar material a {mesh_obj.name}: {e}")
        print(f"   [mat] {mesh_obj.name} ({sub_ref}) ← BL {bl_for_sub}")

    # ── Activar SSR/Refraction (o Raytracing en 4.2+) si HAY translúcidos ──
    # Sólo afecta cuando el motor activo es EEVEE; en Cycles es no-op. Esto
    # permite que scripts EEVEE consumidores (generate_set_random_position,
    # generate_300_random_set, generate_test_set, etc.) levanten la
    # configuración global ÚNICAMENTE para piezas translúcidas, sin alterar
    # el resto del entorno físico, luces ni cámara.
    try:
        has_translucent = any(
            _material_is_translucent(m)
            for obj in imported_meshes
            for m in (obj.data.materials if obj and obj.data else [])
            if m is not None
        )
        if has_translucent:
            configure_eevee_for_translucent(bpy.context.scene)
            print("   [eevee] SSR + Refraction activados para pieza translúcida.")
    except Exception as _e:
        print(f"   [eevee] no se pudo activar refracción: {_e}")

    # ── Unir todos los meshes en un único objeto multi-material ──
    bpy.ops.object.select_all(action='DESELECT')
    for m in imported_meshes:
        try:
            m.select_set(True)
        except Exception:
            pass
    primary = imported_meshes[0]
    bpy.context.view_layer.objects.active = primary
    if len(imported_meshes) > 1:
        try:
            bpy.ops.object.join()
        except Exception as e:
            print(f"   ⚠️ join() falló (se mantienen meshes separados): {e}")

    active_obj = bpy.context.view_layer.objects.active
    if active_obj is None:
        active_obj = primary
    active_obj.name = f"Lego_{part_ref}"

    # Aplicar Bisel sobre el objeto unido
    apply_bevel_modifier(active_obj)

    # Aplicar Físicas y Caída
    apply_rigid_body_physics(active_obj)
    simulate_drop_to_rest(active_obj)

    # Configurar motor de renderizado Cycles
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.render.film_transparent = True
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.filepath = output_path

    # Guardar archivo .blend para depuración antes del render
    try:
        blend_path = output_path.replace('.png', '.blend')
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"Debug .blend guardado en: {blend_path}")
    except Exception as e:
        print(f"No se pudo guardar .blend de depuración: {e}")

    # Disparar renderizado
    bpy.ops.render.render(write_still=True)
    print(f"🎉 Renderizado final fotorrealista guardado en: {output_path}")
    return True

def generate_detailed_fallback_mesh(part_ref):
    """Genera geometría detallada fotorrealista para piezas específicas si no se usa el importador de Studio."""
    # Esta función simula la forma detallada con sus studs para asegurar que el render de demostración
    # capture la realidad de la pieza física exacta.
    bpy.ops.object.select_all(action='DESELECT')
    
    if part_ref == "4589" or part_ref == "4589b":
        # Cono 1x1 con ranura superior
        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.7, radius2=0.4, depth=1.4, location=(0,0,0.7))
        cone = bpy.context.active_object
        # Añadir stud arriba
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.25, depth=0.25, location=(0,0,1.5))
        stud = bpy.context.active_object
        # Unir piezas
        bpy.ops.object.select_all(action='DESELECT')
        cone.select_set(True)
        stud.select_set(True)
        bpy.context.view_layer.objects.active = cone
        bpy.ops.object.join()
        
    elif part_ref == "2540":
        # Placa 1x2 con manija
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,0,0.2))
        base = bpy.context.active_object
        base.scale = (1.4, 0.7, 0.3)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # Añadir manija (cilindro)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.1, depth=0.8, location=(0, 0.6, 0.4))
        handle = bpy.context.active_object
        handle.rotation_euler = (0, math.pi/2, 0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        # Unir
        bpy.ops.object.select_all(action='DESELECT')
        base.select_set(True)
        handle.select_set(True)
        bpy.context.view_layer.objects.active = base
        bpy.ops.object.join()
        
    elif part_ref == "51739" or part_ref == "51739.dat":
        # Placa Cuña 2x4 (Wedge) - Geometría trapezoidal exacta con 4 studs
        mesh = bpy.data.meshes.new(name="Wedge_Base")
        wedge = bpy.data.objects.new("Wedge_Base_Obj", mesh)
        bpy.context.collection.objects.link(wedge)
        bpy.context.view_layer.objects.active = wedge
        
        # Vértices (Lego 4x2 studs en escala: X de -2 a 2, Y de -1 a 1, Z de 0 a 0.32)
        verts = [
            (-2.0, -1.0, 0.0),   # 0: Inferior izquierda
            (2.0, -1.0, 0.0),    # 1: Inferior derecha
            (2.0, 1.0, 0.0),     # 2: Superior derecha
            (0.0, 1.0, 0.0),     # 3: Superior mitad (corte diagonal desde aquí a 0)
            (-2.0, -1.0, 0.32),  # 4: Superior-Inferior izquierda
            (2.0, -1.0, 0.32),   # 5: Superior-Inferior derecha
            (2.0, 1.0, 0.32),    # 6: Superior-Superior derecha
            (0.0, 1.0, 0.32),    # 7: Superior-Superior mitad
        ]
        faces = [
            (3, 2, 1, 0),        # Cara inferior
            (4, 5, 6, 7),        # Cara superior
            (0, 1, 5, 4),        # Frente (Y=-1)
            (1, 2, 6, 5),        # Derecha (X=2)
            (2, 3, 7, 6),        # Atrás (Y=1)
            (3, 0, 4, 7),        # Diagonal
        ]
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        
        # Agregar 4 studs en el área de 2x2 no recortada (a la derecha)
        studs = []
        stud_coords = [
            (1.5, 0.5, 0.32),
            (1.5, -0.5, 0.32),
            (0.5, 0.5, 0.32),
            (0.5, -0.5, 0.32),
        ]
        for sx, sy, sz in stud_coords:
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.25, depth=0.2, location=(sx, sy, sz + 0.1))
            studs.append(bpy.context.active_object)
            
        # Unir todos los studs a la base
        bpy.ops.object.select_all(action='DESELECT')
        for s in studs:
            s.select_set(True)
        wedge.select_set(True)
        bpy.context.view_layer.objects.active = wedge
        bpy.ops.object.join()
        
    elif part_ref == "61184":
        # Technic Pin Connector Hub
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.4, depth=1.2, location=(0,0,0.6))
        hub = bpy.context.active_object
        # Pin transversal
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.25, depth=0.8, location=(0.4, 0, 0.6))
        pin = bpy.context.active_object
        pin.rotation_euler = (0, math.pi/2, 0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        # Unir
        bpy.ops.object.select_all(action='DESELECT')
        hub.select_set(True)
        pin.select_set(True)
        bpy.context.view_layer.objects.active = hub
        bpy.ops.object.join()
        
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0,0,0.5))

def enable_metal_gpu_acceleration():
    """Configura las preferencias de Blender para acelerar Cycles usando la GPU Metal (Apple Silicon)."""
    if not IN_BLENDER:
        return
    try:
        preferences = bpy.context.preferences
        cycles_addon = preferences.addons.get("cycles")
        if cycles_addon:
            cycles_prefs = cycles_addon.preferences
            cycles_prefs.compute_device_type = 'METAL'
            cycles_prefs.get_devices()
            
            # Activar todos los dispositivos GPU de tipo Metal (M1/M2/M3/M4)
            metal_gpu_enabled = False
            for device in cycles_prefs.devices:
                if device.type == 'METAL':
                    device.use = True
                    metal_gpu_enabled = True
                    print(f"✅ GPU Metal activada para renderizado: {device.name}")
            
            if metal_gpu_enabled:
                bpy.context.scene.cycles.device = 'GPU'
                print("🚀 Cycles configurado en modo GPU (Metal).")
            else:
                print("⚠️ No se encontraron GPUs Metal. Se usará CPU por defecto.")
    except Exception as e:
        print(f"Error al activar la aceleración Metal: {e}")

if __name__ == "__main__":
    print("=== LegoVision Blender Physics and Photorealistic Rendering ===")
    
    # Separar argumentos pasados después de '--' en consola
    args = []
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        
    import argparse
    parser = argparse.ArgumentParser(description="Blender Synthetic Lego Renderer")
    parser.add_argument("--part_ref", type=str, default="", help="Lego part reference")
    parser.add_argument("--color_hex", type=str, default="", help="Hex color code")
    parser.add_argument("--output_path", type=str, default="", help="Output image file path")
    parsed_args = parser.parse_known_args(args)[0]
    
    if parsed_args.part_ref and parsed_args.color_hex and parsed_args.output_path:
        # MODO SUBPROCESO (Ejecutado individualmente en paralelo)
        if IN_BLENDER:
            enable_metal_gpu_acceleration()
            setup_physics_world()
            create_conveyor_belt_collider()
            setup_camera()
            setup_studio_lighting()
            render_piece_pipeline(parsed_args.part_ref, parsed_args.color_hex, parsed_args.output_path)
    else:
        # MODO DEMO / SECUENCIAL (Para testeo local)
        if IN_BLENDER:
            enable_metal_gpu_acceleration()
            setup_physics_world()
            create_conveyor_belt_collider()
            setup_camera()
            setup_studio_lighting()
            
        test_parts = [
            {"ref": "4589b", "color": "#FE8A18"}, # Bright Orange
            {"ref": "2540",  "color": "#A0A5A9"}, # Medium Stone Gray
            {"ref": "51739", "color": "#1B1B1B"}, # Black
            {"ref": "61184", "color": "#F2F3F2"}  # White
        ]
        
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch")
        for tp in test_parts:
            out_path = os.path.join(output_dir, f"render_realist_{tp['ref']}.png")
            render_piece_pipeline(tp['ref'], tp['color'], out_path)