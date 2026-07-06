# -*- coding: utf-8 -*-
# scripts/ldraw_primitives_map.py
# Diccionario de mapeo exhaustivo que relaciona los archivos de primitivas de LDraw
# (carpeta p/ y parts/ básicos) con las 8 clases topológicas requeridas.

PRIMITIVES_MAP = {
    # --- stud_solid ---
    "stud.dat": "stud_solid",
    "stud-logo.dat": "stud_solid",
    "stud-logo2.dat": "stud_solid",
    "stud-logo3.dat": "stud_solid",
    "stud-logo4.dat": "stud_solid",
    "stud-logo5.dat": "stud_solid",
    "studs.dat": "stud_solid",
    "stud1.dat": "stud_solid",
    "studla.dat": "stud_solid",

    # --- stud_hollow ---
    "stud2.dat": "stud_hollow",
    "stud2-logo.dat": "stud_hollow",
    "stud2-logo2.dat": "stud_hollow",
    "stud2-logo3.dat": "stud_hollow",
    "stud2-logo4.dat": "stud_hollow",
    "stud2-logo5.dat": "stud_hollow",
    "stud23d.dat": "stud_hollow",
    "stud2s2e.dat": "stud_hollow",
    "stud9.dat": "stud_hollow",
    "stud10.dat": "stud_hollow",
    "stud11.dat": "stud_hollow",
    "stud21.dat": "stud_hollow",
    "stud22.dat": "stud_hollow",
    "stud22a.dat": "stud_hollow",
    "hipstudh.dat": "stud_hollow",
    "filstud2.dat": "stud_hollow",
    "filstud3.dat": "stud_hollow",

    # --- technic_hole_round ---
    "hole.dat": "technic_hole_round",
    "holey.dat": "technic_hole_round",
    "hole3.dat": "technic_hole_round",
    "hole4.dat": "technic_hole_round",
    "peghole.dat": "technic_hole_round",
    "peghole2.dat": "technic_hole_round",
    "peghole3.dat": "technic_hole_round",
    "peghole4.dat": "technic_hole_round",
    "peghole5.dat": "technic_hole_round",
    "peghole6.dat": "technic_hole_round",
    "beamhole.dat": "technic_hole_round",
    "connhole.dat": "technic_hole_round",
    "dnpeghole.dat": "technic_hole_round",
    "npeghole.dat": "technic_hole_round",
    "clikhole.dat": "technic_hole_round",

    # --- technic_hole_cross ---
    "holex.dat": "technic_hole_cross",
    "holey2.dat": "technic_hole_cross",
    "holey3.dat": "technic_hole_cross",
    "axlehole.dat": "technic_hole_cross",
    "axl2hole.dat": "technic_hole_cross",
    "axl3hole.dat": "technic_hole_cross",
    "axl4hole.dat": "technic_hole_cross",
    "daxlehole.dat": "technic_hole_cross",

    # --- clip_jaw ---
    "clip.dat": "clip_jaw",
    "clip1.dat": "clip_jaw",
    "clip2.dat": "clip_jaw",
    "clip3.dat": "clip_jaw",
    "clip4.dat": "clip_jaw",
    "clip5.dat": "clip_jaw",
    "clip6.dat": "clip_jaw",
    "clip7.dat": "clip_jaw",
    "clip8.dat": "clip_jaw",
    "clip9.dat": "clip_jaw",
    "clip10.dat": "clip_jaw",
    "clip11.dat": "clip_jaw",
    "clip12.dat": "clip_jaw",
    "clip13.dat": "clip_jaw",
    "clip16.dat": "clip_jaw",

    # --- bar_handle ---
    "bar.dat": "bar_handle",
    "bar1.dat": "bar_handle",
    "bar2.dat": "bar_handle",
    "bar3.dat": "bar_handle",
    "bar4.dat": "bar_handle",
    "barl.dat": "bar_handle",
    "barla.dat": "bar_handle",
    "barlb.dat": "bar_handle",

    # --- bottom_tube ---
    "stud4.dat": "bottom_tube",
    "stud4a.dat": "bottom_tube",
    "stud4h.dat": "bottom_tube",
    "stud4h2.dat": "bottom_tube",
    "stud16.dat": "bottom_tube",
    "stud17.dat": "bottom_tube",
    "stud22a.dat": "bottom_tube",
    "stud23d.dat": "bottom_tube",
    "1-4stud4.dat": "bottom_tube",
    "1-8stud4.dat": "bottom_tube",
    "2-4stud4.dat": "bottom_tube",
    "2-4stud4a.dat": "bottom_tube",
    "2-4stud4f1w.dat": "bottom_tube",
    "2-4stud4t45.dat": "bottom_tube",
    "3-4stud4.dat": "bottom_tube",
    "1-16stud4.dat": "bottom_tube",
    "3-16stud4.dat": "bottom_tube",
    "3-16stud4t4.dat": "bottom_tube",
    "5-16stud4.dat": "bottom_tube",
    "tube.dat": "bottom_tube",
    "tubex.dat": "bottom_tube",

    # --- bottom_pin ---
    "stud3.dat": "bottom_pin",
    "stud3a.dat": "bottom_pin",
    "pin.dat": "bottom_pin",
    "pin1.dat": "bottom_pin",
    "pin2.dat": "bottom_pin",
    "pin3.dat": "bottom_pin",
    "pin4.dat": "bottom_pin",
    "duplopin.dat": "bottom_pin",
}

# Creamos una versión en minúsculas por seguridad en la búsqueda
PRIMITIVES_MAP_LOWER = {k.lower(): v for k, v in PRIMITIVES_MAP.items()}
