# LegoVision — Pipeline de Descarga e Importación LDraw

Este documento detalla la descarga del catálogo completo de LDraw y la integración del addon de importación nativa en Blender.

## 1. Descarga del Catálogo LDraw
El catálogo oficial de LDraw se descarga automáticamente usando el script [`scripts/download_ldraw.sh`](file:///Users/I764690/Code_personal/LegoVision/scripts/download_ldraw.sh).

El script realiza lo siguiente:
1. Descarga el paquete completo `complete.zip` desde la librería oficial de LDraw (~500MB).
2. Extrae las subcarpetas `parts/` (partes principales `.dat`) y `p/` (primitivas geométricas) en `data/ldraw/`.
3. Ejecuta el indexador `blender_pipeline/ldraw_catalog.py` para analizar los headers de todos los archivos `.dat`, filtrar piezas no deseadas (minifiguras, pegatinas, cables) y asignarles un índice de clase único para YOLOv8.

---

## 2. Addon Blender: `ldr_tools_blender`
Para permitir que Blender lea directamente los archivos nativos de LDraw (`.dat`) sin requerir conversiones manuales a `.obj` o `.fbx`, utilizamos el addon de alto rendimiento en Rust `ldr_tools_blender` (de *ScanMountGoat*).

### Instalación Automática
El script [`scripts/install_blender_addon.py`](file:///Users/I764690/Code_personal/LegoVision/scripts/install_blender_addon.py) se encarga de:
1. Detectar el sistema operativo y arquitectura del procesador (por ejemplo, macOS Apple Silicon M4).
2. Descargar el archivo `.zip` del release oficial correspondiente (por ejemplo, `ldr_tools_blender_macos_apple_silicon.zip`).
3. Instalar y activar el addon a través de la API `bpy` de Blender de forma headless.

### Uso en Blender
El addon registra un operador nativo de importación:
```python
# Importación nativa en Python
bpy.ops.import_scene.ldr(filepath="ruta/a/pieza.dat")
```
Esto genera las mallas con materiales PBR optimizados para el motor Cycles de Blender de forma instanciada, lo que permite renderizar decenas de piezas con un consumo de memoria mínimo.
