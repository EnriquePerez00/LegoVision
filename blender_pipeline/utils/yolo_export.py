import bpy
import bpy_extras
from mathutils import Vector

def get_yolo_bbox(obj, camera, scene):
    """
    Calcula el Bounding Box 2D normalizado para YOLO de un objeto y sus hijos de tipo malla.
    Retorna (x_center, y_center, width, height) o None si el objeto está totalmente fuera de cámara.
    """
    from mathutils import Vector
    import bpy_extras

    # Encontrar todos los sub-objetos de malla (jerarquía LDraw)
    mesh_objs = []
    if obj.type == 'MESH' and len(obj.data.polygons) > 0:
        mesh_objs.append(obj)
    
    # Recorrer recursivamente los hijos para encontrar mallas
    def get_mesh_children(parent):
        for child in parent.children:
            if child.type == 'MESH' and len(child.data.polygons) > 0:
                mesh_objs.append(child)
            get_mesh_children(child)
            
    get_mesh_children(obj)
    
    if not mesh_objs:
        return None
        
    x_coords = []
    y_coords = []
    
    # Proyectar las 8 esquinas de la caja delimitadora (bound_box) de cada malla a la vista 2D
    for m_obj in mesh_objs:
        mw = m_obj.matrix_world
        for corner in m_obj.bound_box:
            v_world = mw @ Vector(corner)
            co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, v_world)
            # co_2d.z es la profundidad (debe estar delante de la cámara)
            if co_2d.z > 0:
                x_coords.append(co_2d.x)
                y_coords.append(co_2d.y)
                
    if not x_coords or not y_coords:
        return None
        
    # Encontrar extremos (min y max)
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    # Recortar al borde de la imagen (0.0 a 1.0)
    min_x = max(0.0, min_x)
    max_x = min(1.0, max_x)
    min_y = max(0.0, min_y)
    max_y = min(1.0, max_y)
    
    # Verificar si tiene un tamaño visible mínimo en cámara
    width = max_x - min_x
    height = max_y - min_y
    
    if width <= 0.005 or height <= 0.005:
        # Objeto demasiado pequeño o fuera de la pantalla
        return None
        
    # En YOLO, el origen (0,0) es la esquina superior izquierda
    # En Blender, world_to_camera_view pone (0,0) en la esquina inferior izquierda.
    x_center = (min_x + max_x) / 2.0
    y_center = 1.0 - ((min_y + max_y) / 2.0)
    
    return (x_center, y_center, width, height)

def save_yolo_label(filepath, detections):
    """
    Guarda las detecciones de una imagen en formato de etiqueta YOLO (.txt).
    detections: lista de diccionarios con {"class_idx": int, "bbox": (x, y, w, h)}
    """
    with open(filepath, 'w') as f:
        for det in detections:
            class_idx = det["class_idx"]
            x, y, w, h = det["bbox"]
            f.write(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
