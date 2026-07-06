import os, sys, subprocess, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as ef:
        for line in ef:
            if "=" in line and not line.startswith("#"):
                pts = line.strip().split("=", 1)
                if len(pts) == 2:
                    k = pts[0].strip()
                    v = pts[1].strip().strip(chr(34)).strip(chr(39))
                    os.environ[k] = v

TMP_DIR              = os.path.join(PROJECT_ROOT, "data", "tmp")
GLB_OUT_DIR          = os.path.join(PROJECT_ROOT, "gui", "static", "models")
UNOFFICIAL_PARTS_DIR = os.path.join(PROJECT_ROOT, "data", "ldraw", "Unofficial", "parts")

MINIFIG_DATABASE = {
    "sw0614": {
        "name": "Stormtrooper (Rebels) with Azure Vents",
        "components": [
            {"part_key": "torso",      "part_file": "973.dat",   "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Torso"},
            {"part_key": "hips",       "part_file": "3815.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Caderas"},
            {"part_key": "left_leg",   "part_file": "3816.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Pierna Izq"},
            {"part_key": "right_leg",  "part_file": "3817.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Pierna Der"},
            {"part_key": "left_arm",   "part_file": "3819.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Brazo Izq"},
            {"part_key": "right_arm",  "part_file": "3818.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Brazo Der"},
            {"part_key": "left_hand",  "part_file": "3820.dat",  "ldraw_color": "0",  "color_hex": "#1B1B1B", "color_name": "Black", "label": "Mano Izq"},
            {"part_key": "right_hand", "part_file": "3820.dat",  "ldraw_color": "0",  "color_hex": "#1B1B1B", "color_name": "Black", "label": "Mano Der"},
            {"part_key": "head",       "part_file": "3626c.dat", "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh", "label": "Cabeza"},
            {"part_key": "helmet",     "part_file": "30408.dat", "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Casco"},
        ]
    },
    "sw1093": {
        "name": "Clone Trooper 501st Legion",
        "components": [
            {"part_key": "torso",      "part_file": "973.dat",   "ldraw_color": "1",  "color_hex": "#0A3C9F", "color_name": "Blue",  "label": "Torso"},
            {"part_key": "hips",       "part_file": "3815.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Caderas"},
            {"part_key": "left_leg",   "part_file": "3816.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Pierna Izq"},
            {"part_key": "right_leg",  "part_file": "3817.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Pierna Der"},
            {"part_key": "left_arm",   "part_file": "3819.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Brazo Izq"},
            {"part_key": "right_arm",  "part_file": "3818.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Brazo Der"},
            {"part_key": "left_hand",  "part_file": "3820.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Mano Izq"},
            {"part_key": "right_hand", "part_file": "3820.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Mano Der"},
            {"part_key": "head",       "part_file": "3626c.dat", "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh", "label": "Cabeza"},
            {"part_key": "helmet",     "part_file": "2446.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White", "label": "Casco Clone"},
        ]
    },
    "sw0886": {
        "name": "Luke Skywalker X-Wing Pilot",
        "components": [
            {"part_key": "torso",      "part_file": "973.dat",   "ldraw_color": "25", "color_hex": "#FE8A18", "color_name": "Orange", "label": "Torso"},
            {"part_key": "hips",       "part_file": "3815.dat",  "ldraw_color": "25", "color_hex": "#FE8A18", "color_name": "Orange", "label": "Caderas"},
            {"part_key": "left_leg",   "part_file": "3816.dat",  "ldraw_color": "25", "color_hex": "#FE8A18", "color_name": "Orange", "label": "Pierna Izq"},
            {"part_key": "right_leg",  "part_file": "3817.dat",  "ldraw_color": "25", "color_hex": "#FE8A18", "color_name": "Orange", "label": "Pierna Der"},
            {"part_key": "left_arm",   "part_file": "3819.dat",  "ldraw_color": "25", "color_hex": "#FE8A18", "color_name": "Orange", "label": "Brazo Izq"},
            {"part_key": "right_arm",  "part_file": "3818.dat",  "ldraw_color": "25", "color_hex": "#FE8A18", "color_name": "Orange", "label": "Brazo Der"},
            {"part_key": "left_hand",  "part_file": "3820.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White",  "label": "Mano Izq"},
            {"part_key": "right_hand", "part_file": "3820.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White",  "label": "Mano Der"},
            {"part_key": "head",       "part_file": "3626c.dat", "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh",  "label": "Cabeza"},
            {"part_key": "helmet",     "part_file": "30408.dat", "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White",  "label": "Casco X-Wing"},
        ]
    },
    "sw0778": {
        "name": "Luke Skywalker (Tatooine, White Legs)",
        "components": [
            {"part_key": "torso",      "part_file": "973.dat",   "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White",  "label": "Torso"},
            {"part_key": "hips",       "part_file": "3815.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White",  "label": "Caderas"},
            {"part_key": "left_leg",   "part_file": "3816.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White",  "label": "Pierna Izq"},
            {"part_key": "right_leg",  "part_file": "3817.dat",  "ldraw_color": "15", "color_hex": "#FFFFFF", "color_name": "White",  "label": "Pierna Der"},
            {"part_key": "left_arm",   "part_file": "3819.dat",  "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh",  "label": "Brazo Izq"},
            {"part_key": "right_arm",  "part_file": "3818.dat",  "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh",  "label": "Brazo Der"},
            {"part_key": "left_hand",  "part_file": "3820.dat",  "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh",  "label": "Mano Izq"},
            {"part_key": "right_hand", "part_file": "3820.dat",  "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh",  "label": "Mano Der"},
            {"part_key": "head",       "part_file": "3626c.dat", "ldraw_color": "78", "color_hex": "#F5C5A3", "color_name": "Flesh",  "label": "Cabeza"},
            {"part_key": "hair",       "part_file": "92081.dat", "ldraw_color": "3",  "color_hex": "#F2CD37", "color_name": "Yellow", "label": "Pelo"},
        ]
    }
}

STANDING_TEMPLATE = [
    ("right_leg",   0,      -28,  0,  "1 0 0 0 1 0 0 0 1"),
    ("left_leg",    0,      -28,  0,  "1 0 0 0 1 0 0 0 1"),
    ("hips",        0,      -40,  0,  "1 0 0 0 1 0 0 0 1"),
    ("torso",       0,      -72,  0,  "1 0 0 0 1 0 0 0 1"),
    ("head",        0,     -100,  0,  "1 0 0 0 1 0 0 0 1"),
    ("helmet",      0,     -100,  0,  "1 0 0 0 1 0 0 0 1"),
    ("hair",        0,     -100,  0,  "1 0 0 0 1 0 0 0 1"),
    ("right_arm",  -15.552, -63,  0,  "0.9855 -0.1699 0 0.1699 0.9855 0 0 0 1"),
    ("left_arm",    15.552, -63,  0,  "0.9855 0.1699 0 -0.1699 0.9855 0 0 0 1"),
    ("right_hand", -23.552, -46, -10, "0.942 0.335 0.0072 -0.2404 0.6906 -0.6821 -0.2336 0.6409 0.7312"),
    ("left_hand",   23.552, -46, -10, "0.942 -0.335 0.0072 0.2404 0.6906 -0.6821 0.2336 0.6409 0.7312"),
]


def get_minifigs_from_test_sets():
    test_sets = ["75078-1", "75280-1", "75218-1", "75337-1", "911943-1"]
    result = []
    seen_refs = set()
    try:
        from core.db import set_catalog
        for set_id in test_sets:
            for mfig in set_catalog.REAL_SETS.get(set_id, {}).get("minifigures", []):
                ref = mfig["ref"]
                if ref in MINIFIG_DATABASE and ref not in seen_refs:
                    seen_refs.add(ref)
                    result.append({"ref": ref, "name": mfig["name"], "set_id": set_id, "qty": mfig["qty"]})
    except Exception as e:
        print("[assemble_minifig] Error:", e)
    return result


def generate_ldr_content(minifig_id):
    if minifig_id not in MINIFIG_DATABASE:
        return None
    config = MINIFIG_DATABASE[minifig_id]
    cbk = {c["part_key"]: c for c in config["components"]}
    parts = [
        "0 Minifig Assembly: " + config["name"],
        "0 Name: " + minifig_id + "_assembly.ldr",
        "0 BFC CERTIFY CCW",
        ""
    ]
    for pk, x, y, z, rot in STANDING_TEMPLATE:
        c = cbk.get(pk)
        if c:
            parts.append("1 " + c["ldraw_color"] + " " + str(x) + " " + str(y) + " " + str(z) + " " + rot + " " + c["part_file"])
    return chr(10).join(parts) + chr(10)


def _make_color_map(components):
    cm = {}
    for c in components:
        hx = c["color_hex"].lstrip("#")
        cm[c["part_key"]] = {
            "r": int(hx[0:2], 16) / 255.0,
            "g": int(hx[2:4], 16) / 255.0,
            "b": int(hx[4:6], 16) / 255.0,
            "name": c["color_name"],
            "label": c["label"]
        }
    return cm

def assemble_minifig_blend(minifig_id, ldr_path, dat_path, glb_path, components):
    blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
    blender_script = os.path.join(TMP_DIR, "blender_assemble.py")
    color_map = _make_color_map(components)
    NL = chr(10)
    Q = chr(39)
    sl = []
    sl.append("import bpy, json, os, shutil")
    sl.append("bpy.ops.object.select_all(action=" + repr("SELECT") + ")")
    sl.append("bpy.ops.object.delete()")
    sl.append("ldr_filepath = " + repr(ldr_path))
    sl.append("glb_output   = " + repr(glb_path))
    sl.append("dat_output   = " + repr(dat_path))
    sl.append("color_map    = " + json.dumps(color_map))
    sl.append("try:")
    sl.append("    bpy.ops.import_scene.importldr(filepath=ldr_filepath)")
    sl.append("except Exception as e:")
    sl.append("    try:")
    sl.append("        bpy.ops.import_scene.importldraw(filepath=ldr_filepath, ldrawPath=" + repr("/Applications/Studio 2.0/ldraw") + ", importCameras=False, positionOnGround=False)")
    sl.append("    except Exception as e2: print(" + repr("Importadores fallaron:") + ", e2)")
    sl.append("meshes = [o for o in bpy.context.scene.objects if o.type == " + repr("MESH") + "]")
    sl.append("print(" + repr("Meshes:") + ", len(meshes))")
    sl.append("if meshes:")
    sl.append("    keys = list(color_map.keys())")
    sl.append("    for i, obj in enumerate(meshes):")
    sl.append("        bpy.ops.object.select_all(action=" + repr("DESELECT") + ")")
    sl.append("        obj.select_set(True)")
    sl.append("        bpy.context.view_layer.objects.active = obj")
    sl.append("        ck = keys[i % len(keys)] if keys else " + repr("default"))
    sl.append("        cm = color_map.get(ck, {" + repr("r") + ":0.8," + repr("g") + ":0.8," + repr("b") + ":0.8})")
    sl.append("        mat = bpy.data.materials.new(name=" + repr("Mat_") + " + ck)")
    sl.append("        mat.use_nodes = True")
    sl.append("        bsdf = mat.node_tree.nodes.get(" + repr("Principled BSDF") + ") or mat.node_tree.nodes.new(" + repr("ShaderNodeBsdfPrincipled") + ")")
    sl.append("        bsdf.inputs[" + repr("Base Color") + "].default_value = (cm[" + repr("r") + "], cm[" + repr("g") + "], cm[" + repr("b") + "], 1.0)")
    sl.append("        if not obj.data.materials: obj.data.materials.append(mat)")
    sl.append("        else: obj.data.materials[0] = mat")
    sl.append("    bpy.ops.object.select_all(action=" + repr("SELECT") + ")")
    sl.append("    bpy.context.view_layer.objects.active = meshes[0]")
    sl.append("    if len(meshes) > 1: bpy.ops.object.join()")
    sl.append("    bpy.context.active_object.name = " + repr("Minifig_" + minifig_id))
    sl.append("    bpy.ops.object.origin_set(type=" + repr("ORIGIN_GEOMETRY") + ", center=" + repr("BOUNDS") + ")")
    sl.append("    os.makedirs(os.path.dirname(glb_output), exist_ok=True)")
    sl.append("    try:")
    sl.append("        bpy.ops.export_scene.gltf(filepath=glb_output, export_format=" + repr("GLB") + ", use_selection=True)")
    sl.append("        print(" + repr("GLB exportado:") + ", glb_output)")
    sl.append("    except Exception as eg: print(" + repr("Error GLB:") + ", eg)")
    sl.append("    os.makedirs(os.path.dirname(dat_output), exist_ok=True)")
    sl.append("    shutil.copyfile(ldr_filepath, dat_output)")
    sl.append("else:")
    sl.append("    print(" + repr("Sin mallas.") + ")")
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(blender_script, "w") as f:
        f.write(NL.join(sl) + NL)
    cmd = [blender_path, "-b", "-P", blender_script]
    print("Ejecutando:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    if os.path.exists(blender_script):
        os.remove(blender_script)

def build_minifig(minifig_id, save_to_db=True):
    if minifig_id not in MINIFIG_DATABASE:
        return {"success": False, "error": "Minifig no encontrada"}
    config = MINIFIG_DATABASE[minifig_id]
    components = config["components"]
    ldr_content = generate_ldr_content(minifig_id)
    if not ldr_content:
        return {"success": False, "error": "LDR no generado"}
    os.makedirs(TMP_DIR, exist_ok=True)
    ldr_path = os.path.join(TMP_DIR, minifig_id + "_assembly.ldr")
    with open(ldr_path, "w") as f:
        f.write(ldr_content)
    dat_path = os.path.join(UNOFFICIAL_PARTS_DIR, minifig_id + ".dat")
    glb_path = os.path.join(GLB_OUT_DIR, minifig_id + ".glb")
    assemble_minifig_blend(minifig_id, ldr_path, dat_path, glb_path, components)
    if os.path.exists(ldr_path):
        os.remove(ldr_path)
    glb_exists = os.path.exists(glb_path)
    dat_exists = os.path.exists(dat_path)
    print("  GLB:", "OK" if glb_exists else "FALLO", "| DAT:", "OK" if dat_exists else "FALLO")
    if save_to_db:
        try:
            from core.db import supabase_client
            glb_data = None
            if glb_exists:
                with open(glb_path, "rb") as gf:
                    glb_data = gf.read()
            supabase_client.save_minifig_assembly(
                minifig_ref=minifig_id,
                name=config["name"],
                glb_path=glb_path,
                components=components,
                glb_data=glb_data
            )
        except Exception as e:
            print("BD Error:", e)
    return {"success": glb_exists, "glb_path": glb_path if glb_exists else None, "dat_path": dat_path if dat_exists else None}


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "sw0614"
    build_minifig(mid)
