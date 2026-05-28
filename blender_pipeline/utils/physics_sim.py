import bpy
import sys
import os

# Añadir directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def setup_physics_world(scene):
    """Inicializa la configuración de la simulación de cuerpo rígido en la escena."""
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
        
    world = scene.rigidbody_world
    world.enabled = True
    # Usar 60 FPS para la simulación
    scene.render.fps = 60
    world.time_scale = 1.0
    world.substeps_per_frame = 10
    world.solver_iterations = 10
    
    # Crear caché o configurar rango
    world.point_cache.frame_start = 1
    world.point_cache.frame_end = 100

def create_belt_collider(scene):
    """Crea la cinta transportadora como un colisionador pasivo."""
    # Eliminar plano de cinta previo si existe
    if "ConveyorBelt" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["ConveyorBelt"], do_unlink=True)
        
    # El plano de la cinta debe cubrir el FOV de la cámara
    # FOV es 250mm de largo, ancho cinta es 200mm
    ancho_m = config.BELT_WIDTH_MM * config.BLENDER_SCALE
    largo_m = config.FOV_LONGITUDINAL_MM * config.BLENDER_SCALE * 1.5  # Margen extra
    
    bpy.ops.mesh.primitive_plane_add(size=1.0, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    belt = bpy.context.active_object
    belt.name = "ConveyorBelt"
    belt.scale = (ancho_m, largo_m, 1.0)
    
    # Asignar un material negro opaco y mate
    mat = bpy.data.materials.new(name="BeltMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs['Base Color'].default_value = (0.01, 0.01, 0.01, 1.0) # Negro industrial
        principled.inputs['Roughness'].default_value = 0.85 # Mate
        principled.inputs['Specular IOR Level'].default_value = 0.1 # Muy pocos brillos
    belt.data.materials.append(mat)
    
    # Configurar cuerpo rígido pasivo
    bpy.ops.rigidbody.object_add()
    belt.rigid_body.type = 'PASSIVE'
    belt.rigid_body.collision_shape = 'BOX'
    belt.rigid_body.friction = 0.9  # Alta fricción
    belt.rigid_body.restitution = 0.1 # Baja elasticidad (piezas no rebotan indefinidamente)
    
    return belt

def apply_rigid_body_to_piece(obj):
    """Aplica cuerpo rígido activo a una pieza LEGO recién importada."""
    # Asegurar que esté seleccionado y activo
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Añadir Rigid Body
    bpy.ops.rigidbody.object_add()
    rb = obj.rigid_body
    rb.type = 'ACTIVE'
    rb.mass = 0.005 # ~5 gramos por pieza promedio
    rb.collision_shape = 'CONVEX_HULL'
    rb.friction = 0.6
    rb.restitution = 0.2
    # Estabilizar simulación añadiendo amortiguamiento
    rb.linear_damping = 0.1
    rb.angular_damping = 0.1

def run_simulation(scene, end_frame=60):
    """Ejecuta la física y posiciona la escena en el frame especificado para el render."""
    # Forzar actualización de la caché de cuerpo rígido
    scene.frame_set(1)
    
    # Simular paso a paso hasta el final
    for f in range(1, end_frame + 1):
        scene.frame_set(f)
        
    print(f"[LegoVision Physics] Simulación calculada hasta el frame {end_frame}")
