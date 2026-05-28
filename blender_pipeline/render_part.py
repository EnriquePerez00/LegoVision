import bpy
import sys
import os
import argparse
from mathutils import Vector, Euler

def clean_scene():
    """Limpia todos los objetos y colecciones de la escena."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)

def hex_to_rgb(hex_str):
    """Convierte un color hexadecimal (#RRGGBB) a una tupla RGB (0.0 a 1.0)."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def setup_render_env(width=256, height=256):
    """Configura la cámara, iluminación y motor de render (Eevee para velocidad)."""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True  # Fondo transparente
    
    # 1. Cámara
    camera_data = bpy.data.cameras.new(name="RenderCamera")
    camera_obj = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    
    # Posición isométrica del primer plano
    camera_obj.location = (0.05, -0.05, 0.04)
    # Rotación para vista en perspectiva isométrica
    camera_obj.rotation_euler = (Euler((1.047, 0.0, 0.785), 'XYZ')) # 60, 0, 45 grados
    
    # 2. Iluminación (Luz de estudio de 3 puntos)
    # Luz principal
    light_data1 = bpy.data.lights.new(name="KeyLight", type='AREA')
    light_data1.energy = 5.0
    light_data1.size = 0.5
    light_obj1 = bpy.data.objects.new(name="KeyLight", object_data=light_data1)
    light_obj1.location = (0.2, -0.2, 0.3)
    scene.collection.objects.link(light_obj1)
    
    # Luz de relleno
    light_data2 = bpy.data.lights.new(name="FillLight", type='AREA')
    light_data2.energy = 2.0
    light_data2.size = 0.5
    light_obj2 = bpy.data.objects.new(name="FillLight", object_data=light_data2)
    light_obj2.location = (-0.2, -0.1, 0.1)
    scene.collection.objects.link(light_obj2)
    
    # Contraluz
    light_data3 = bpy.data.lights.new(name="RimLight", type='SUN')
    light_data3.energy = 1.0
    light_obj3 = bpy.data.objects.new(name="RimLight", object_data=light_data3)
    light_obj3.location = (0.0, 0.2, 0.2)
    light_obj3.rotation_euler = (0.5, 0.0, 3.14)
    scene.collection.objects.link(light_obj3)

def import_and_color_part(ldraw_path, part_id, color_hex):
    """Importa la pieza LDraw y le aplica el color especificado."""
    part_filename = f"{part_id}.dat"
    filepath = os.path.join(ldraw_path, "parts", part_filename)
    
    if not os.path.exists(filepath):
        filepath = os.path.join(ldraw_path, "p", part_filename)
        
    if not os.path.exists(filepath):
        print(f"Error: No se encontró la pieza LDraw en {filepath}")
        return None
        
    # Importar
    bpy.ops.import_scene.importldr(filepath=filepath)
    
    # Encontrar el objeto malla importado
    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break
            
    if not mesh_obj:
        print("Error: No se importó ninguna malla.")
        return None
        
    # Centrar la pieza en el origen (0,0,0)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
    mesh_obj.location = (0.0, 0.0, 0.0)
    
    # Escalar si es necesario (el addon suele importar a metros pero LDraw es grande)
    # Por defecto, forzar un tamaño consistente
    # Medir la diagonal del bounding box
    bbox = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    size = (bbox[0] - bbox[6]).length
    if size > 0:
        scale_fac = 0.03 / size  # Normalizar tamaño a unos 3 cm
        mesh_obj.scale = (scale_fac, scale_fac, scale_fac)
        bpy.ops.object.transform_apply(scale=True)
        
    # Aplicar color
    rgb = hex_to_rgb(color_hex)
    
    # Crear material LEGO personalizado brillante
    mat = bpy.data.materials.new(name="LegoCustomColor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs['Base Color'].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        principled.inputs['Roughness'].default_value = 0.15  # Brillo plástico
        principled.inputs['Specular IOR Level'].default_value = 0.5
        
    # Limpiar materiales antiguos y asignar el nuevo
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)
    
    return mesh_obj

def main():
    # Procesar argumentos pasados después de "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Renderizador de Piezas LEGO individual bajo demanda")
    parser.add_argument("--part_id", type=str, required=True, help="ID de la pieza LDraw, ej. 3004")
    parser.add_argument("--color_hex", type=str, required=True, help="Color Hexadecimal, ej. #A0A5A9")
    parser.add_argument("--ldraw_path", type=str, default="./data/ldraw", help="Path al catálogo LDraw")
    parser.add_argument("--output", type=str, required=True, help="Path de salida del archivo PNG")
    
    args = parser.parse_args(argv)
    
    clean_scene()
    setup_render_env()
    
    obj = import_and_color_part(args.ldraw_path, args.part_id, args.color_hex)
    if not obj:
        print("Error al importar la pieza.")
        sys.exit(1)
        
    # Render
    bpy.context.scene.render.filepath = args.output
    bpy.ops.render.render(write_still=True)
    print(f"Renderizado guardado con éxito en: {args.output}")

if __name__ == "__main__":
    main()
