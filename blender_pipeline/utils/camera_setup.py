import bpy
import sys
import os

# Añadir directorio padre al path por si se ejecuta dentro de Blender
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def setup_camera(scene):
    """Configura la cámara según los parámetros físicos del setup industrial."""
    # Buscar o crear cámara
    camera_data = bpy.data.cameras.new(name="IndustrialCamera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    
    # 1. Posición y orientación
    # Posición Cenital (WD = 355mm)
    z_pos = config.CAMERA_HEIGHT_MM * config.BLENDER_SCALE
    camera_obj.location = (0.0, 0.0, z_pos)
    # Apuntando verticalmente hacia abajo (0, 0, 0)
    camera_obj.rotation_euler = (0.0, 0.0, 0.0)
    
    # 2. Parámetros ópticos
    camera_data.lens = config.LENS_FOCAL_LENGTH_MM
    camera_data.sensor_width = config.SENSOR_WIDTH_MM
    camera_data.sensor_fit = 'HORIZONTAL'
    
    # Rango de clip (evitar recortes)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 2.0
    
    # 3. Resolución del sensor
    scene.render.resolution_x = config.RESOLUTION_X
    scene.render.resolution_y = config.RESOLUTION_Y
    scene.render.resolution_percentage = 100
    
    return camera_obj
