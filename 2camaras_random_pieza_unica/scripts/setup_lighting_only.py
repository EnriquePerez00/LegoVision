# -*- coding: utf-8 -*-
"""setup_lighting_only.py
Configura UNICAMENTE iluminacion + EEVEE en la escena actual de Blender.
NO toca geometria, camaras ni materiales.

Acciones:
  1) Borra todas las luces existentes (objetos LIGHT + data huerfana).
  2) Configura World a gris neutro (0.5,0.5,0.5) con strength=0.4.
  3) Crea un domo industrial de 3 luces:
     - Cenital  AREA SQUARE 0.4 m  @ (0,0,0.25) apuntando -Z   15 W
     - Lateral1 AREA RECT 0.3x0.1  @ (0,+0.15,0.05) apunta a O  5 W
     - Lateral2 AREA RECT 0.3x0.1  @ (0,-0.15,0.05) apunta a O  5 W
  4) Configura EEVEE: AO + SSR + High Bitdepth shadows + View
     Transform=Standard / Look=Medium Contrast.

Las luces emiten por -Z local. Energia en watts (light.energy es float
y Blender lo interpreta como W para AREA).
"""
import bpy
import math
from mathutils import Vector


# ─── CONFIG (ajusta aqui si cambias unidades) ───────────────────────
KEY_LOC      = (0.0, 0.0, 0.25)
KEY_SIZE     = 0.4   # m, SQUARE
KEY_POWER_W  = 15.0

FILL1_LOC    = (0.0,  0.15, 0.05)
FILL2_LOC    = (0.0, -0.15, 0.05)
FILL_SIZE_X  = 0.3   # m, RECTANGLE largo
FILL_SIZE_Y  = 0.1   # m, RECTANGLE ancho
FILL_POWER_W = 5.0

WORLD_RGB      = (0.5, 0.5, 0.5)   # gris neutro lineal
WORLD_STRENGTH = 0.4
TARGET_POINT   = Vector((0.0, 0.0, 0.0))


# ─── Helpers ────────────────────────────────────────────────────────
def _aim_at(obj, target_world):
    """Orienta el -Z local de `obj` hacia `target_world`."""
    direction = (target_world - obj.location).normalized()
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('-Z', 'Y')


def _aim_down(obj):
    """Orienta el -Z local hacia abajo (luz cenital)."""
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Vector((0.0, 0.0, -1.0)).to_track_quat('-Z', 'Y')


def _make_area_light(name, location, shape, size_x, size_y, power_w):
    """Crea una luz AREA con tamano/forma/potencia explicitos."""
    light_data = bpy.data.lights.new(name=name, type='AREA')
    light_data.shape = shape
    light_data.size = size_x
    if shape == 'RECTANGLE':
        light_data.size_y = size_y
    light_data.energy = power_w
    light_data.color = (1.0, 1.0, 1.0)

    obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    return obj


# ─── 1) Limpieza de luces ───────────────────────────────────────────
def clear_all_lights():
    for obj in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(obj, do_unlink=True)
    # Eliminar tambien data-blocks huerfanos.
    for ld in [l for l in bpy.data.lights if l.users == 0]:
        bpy.data.lights.remove(ld)


# ─── 2) World gris neutro ───────────────────────────────────────────
def configure_world():
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if bg is None:
        bg = nt.nodes.new(type='ShaderNodeBackground')
        out = nt.nodes.get("World Output") or nt.nodes.new(type='ShaderNodeOutputWorld')
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = (*WORLD_RGB, 1.0)
    bg.inputs["Strength"].default_value = WORLD_STRENGTH


# ─── 3) Domo industrial (key + 2 fills) ─────────────────────────────
def build_lighting_dome():
    key = _make_area_light(
        name="Key_Cenital",
        location=KEY_LOC,
        shape='SQUARE',
        size_x=KEY_SIZE, size_y=KEY_SIZE,
        power_w=KEY_POWER_W,
    )
    _aim_down(key)

    fill1 = _make_area_light(
        name="Fill_Lateral_+Y",
        location=FILL1_LOC,
        shape='RECTANGLE',
        size_x=FILL_SIZE_X, size_y=FILL_SIZE_Y,
        power_w=FILL_POWER_W,
    )
    _aim_at(fill1, TARGET_POINT)

    fill2 = _make_area_light(
        name="Fill_Lateral_-Y",
        location=FILL2_LOC,
        shape='RECTANGLE',
        size_x=FILL_SIZE_X, size_y=FILL_SIZE_Y,
        power_w=FILL_POWER_W,
    )
    _aim_at(fill2, TARGET_POINT)
    return key, fill1, fill2


# ─── 4) EEVEE para vision artificial ────────────────────────────────
def configure_eevee():
    scene = bpy.context.scene
    try:
        scene.render.engine = 'BLENDER_EEVEE'
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'

    eevee = scene.eevee

    # AO
    if hasattr(eevee, "use_gtao"):
        eevee.use_gtao = True

    # SSR (EEVEE clasico)
    if hasattr(eevee, "use_ssr"):
        eevee.use_ssr = True
        if hasattr(eevee, "use_ssr_refraction"):
            eevee.use_ssr_refraction = True
    # EEVEE Next: raytracing
    if hasattr(eevee, "use_raytracing"):
        eevee.use_raytracing = True

    # Shadows: High Bitdepth
    if hasattr(eevee, "use_shadow_high_bitdepth"):
        eevee.use_shadow_high_bitdepth = True

    # Color Management
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'Medium Contrast'


# ─── Orquestador ────────────────────────────────────────────────────
def main():
    clear_all_lights()
    configure_world()
    build_lighting_dome()
    configure_eevee()
    print("[setup_lighting_only] OK - luces + EEVEE configurados. "
          "Geometria/camaras/materiales NO modificados.")


if __name__ == "__main__":
    main()
