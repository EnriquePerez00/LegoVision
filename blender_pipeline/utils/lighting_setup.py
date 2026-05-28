import bpy
import sys
import os

# Añadir directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def setup_lighting(scene):
    """
    Crea una iluminación difusa industrial similar a un domo de luz.
    Usa 6 Area Lights posicionadas uniformemente para minimizar sombras fuertes
    y reflejos especulares excesivos en las piezas plásticas.
    """
    # Eliminar luces existentes
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
            
    lights_collection = bpy.data.collections.new("Lighting")
    scene.collection.children.link(lights_collection)
    
    # Parámetros comunes de las luces
    # Altura de las luces: ~300mm
    z_pos = 300.0 * config.BLENDER_SCALE
    # Radio o desplazamiento horizontal: ~180mm
    xy_offset = 180.0 * config.BLENDER_SCALE
    
    # 1. Cuatro luces angulares (esquinas) apuntando al centro
    # Coordenadas (x, y) de las 4 esquinas
    esquinas = [
        (-xy_offset, -xy_offset),
        (xy_offset, -xy_offset),
        (xy_offset, xy_offset),
        (-xy_offset, xy_offset)
    ]
    
    for idx, (x, y) in enumerate(esquinas):
        light_data = bpy.data.lights.new(name=f"CornerLight_{idx}", type='AREA')
        light_data.energy = 5.0  # Watts
        light_data.size = 150.0 * config.BLENDER_SCALE  # Difusa
        light_data.color = (1.0, 1.0, 1.0)
        
        light_obj = bpy.data.objects.new(name=f"CornerLight_{idx}", object_data=light_data)
        light_obj.location = (x, y, z_pos)
        
        # Orientar la luz hacia el centro (0, 0, 0)
        # Una forma rápida es usar un constraint Track To o calcular la rotación
        track_to = light_obj.constraints.new(type='TRACK_TO')
        # Crear un target vacío en el origen si no existe
        target = bpy.data.objects.get("OriginTarget")
        if not target:
            target = bpy.data.objects.new("OriginTarget", None)
            scene.collection.objects.link(target)
        track_to.target = target
        track_to.track_axis = 'TRACK_NEGATIVE_Z'
        track_to.up_axis = 'UP_Y'
        
        lights_collection.objects.link(light_obj)
        
    # 2. Dos luces cenitales directas (superior izquierda y superior derecha)
    cenitales = [
        (-50.0 * config.BLENDER_SCALE, 0.0),
        (50.0 * config.BLENDER_SCALE, 0.0)
    ]
    
    for idx, (x, y) in enumerate(cenitales):
        light_data = bpy.data.lights.new(name=f"CenitalLight_{idx}", type='AREA')
        light_data.energy = 8.0
        light_data.size = 200.0 * config.BLENDER_SCALE
        light_data.color = (1.0, 1.0, 1.0)
        
        light_obj = bpy.data.objects.new(name=f"CenitalLight_{idx}", object_data=light_data)
        light_obj.location = (x, y, z_pos * 1.2)  # Un poco más arriba
        light_obj.rotation_euler = (0.0, 0.0, 0.0)  # Apuntando directo abajo
        
        lights_collection.objects.link(light_obj)
        
    return lights_collection
