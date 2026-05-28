"""
scripts/install_blender_addon.py
Instala el addon ldr_tools_blender en Blender.
Ejecutar con: $BLENDER_PATH --background --python scripts/install_blender_addon.py
"""
import sys
import os
import urllib.request
import tempfile

import platform
system = platform.system().lower()
machine = platform.machine().lower()

if system == "darwin":
    if "arm" in machine or "aarch64" in machine:
        asset_name = "ldr_tools_blender_macos_apple_silicon.zip"
    else:
        asset_name = "ldr_tools_blender_macos_intel.zip"
elif system == "windows":
    asset_name = "ldr_tools_blender_win_x64.zip"
else:
    asset_name = "ldr_tools_blender_linux_x64.zip"

ADDON_URL = f"https://github.com/ScanMountGoat/ldr_tools_blender/releases/download/0.5.1/{asset_name}"
ADDON_MODULE = "ldr_tools_blender"

def main():
    try:
        import bpy
    except ImportError:
        print("[LegoVision] ERROR: Este script debe ejecutarse dentro de Blender.")
        print("  Uso: $BLENDER_PATH --background --python scripts/install_blender_addon.py")
        sys.exit(1)

    blender_version = bpy.app.version_string
    print(f"[LegoVision] Blender {blender_version} detectado")
    print(f"[LegoVision] Descargando addon {ADDON_MODULE}...")

    tmp_zip = None
    try:
        # Descargar addon
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            tmp_zip = f.name

        urllib.request.urlretrieve(ADDON_URL, tmp_zip)
        size_kb = os.path.getsize(tmp_zip) / 1024
        print(f"[LegoVision] Descargado: {size_kb:.1f} KB → {tmp_zip}")

        # Instalar en Blender
        bpy.ops.preferences.addon_install(filepath=tmp_zip, overwrite=True)
        print(f"[LegoVision] Addon instalado correctamente")

        # Activar addon
        bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
        print(f"[LegoVision] Addon '{ADDON_MODULE}' activado")

        # Guardar preferencias
        bpy.ops.wm.save_userpref()
        print(f"[LegoVision] Preferencias guardadas")

        # Verificar instalación
        if ADDON_MODULE in bpy.context.preferences.addons:
            print(f"[LegoVision] ✅ Addon verificado: {ADDON_MODULE} activo")
        else:
            print(f"[LegoVision] ⚠️  Addon instalado pero puede necesitar reinicio de Blender")

    except urllib.error.URLError as e:
        print(f"[LegoVision] ERROR descargando addon: {e}")
        print(f"  Descarga manual: {ADDON_URL}")
        print("  Instala en Blender: Edit > Preferences > Add-ons > Install")
        sys.exit(1)
    except Exception as e:
        print(f"[LegoVision] ERROR instalando addon: {e}")
        sys.exit(1)
    finally:
        if tmp_zip and os.path.exists(tmp_zip):
            os.unlink(tmp_zip)


if __name__ == "__main__":
    main()
