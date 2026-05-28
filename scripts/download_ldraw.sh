#!/usr/bin/env bash
# =============================================================
# LegoVision — Descarga catálogo LDraw completo
# + Instala addon ldr_tools_blender en Blender
# Ejecutar: bash scripts/download_ldraw.sh
# =============================================================
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

LDRAW_URL="https://library.ldraw.org/library/updates/complete.zip"
LDRAW_DEST="./data/ldraw"
TMP_ZIP="/tmp/ldraw_complete_$(date +%s).zip"

# ── Cargar .env si existe ──────────────────────────────────────
[ -f ".env" ] && source .env

BLENDER_EXEC="${BLENDER_PATH:-/opt/homebrew/Caskroom/blender/5.1.1/Blender.app/Contents/MacOS/Blender}"

# ══════════════════════════════════════════════════════════════
# 1. DESCARGAR CATÁLOGO LDRAW
# ══════════════════════════════════════════════════════════════
echo -e "\n${BLUE}━━━ 1/3 Descargando catálogo LDraw completo ━━━${NC}"

if [ -d "$LDRAW_DEST/parts" ] && [ "$(ls -1 $LDRAW_DEST/parts/*.dat 2>/dev/null | wc -l)" -gt 1000 ]; then
    PART_COUNT=$(ls -1 "$LDRAW_DEST/parts"/*.dat 2>/dev/null | wc -l | tr -d ' ')
    warn "Catálogo ya existe ($PART_COUNT partes). Saltando descarga."
    warn "Para re-descargar: rm -rf $LDRAW_DEST && bash scripts/download_ldraw.sh"
else
    mkdir -p "$LDRAW_DEST"
    info "Descargando desde $LDRAW_URL (~500MB)..."
    info "Esto puede tardar 5-20 min dependiendo de tu conexión."
    wget -q --show-progress -O "$TMP_ZIP" "$LDRAW_URL"
    ok "Descarga completada: $(du -sh $TMP_ZIP | cut -f1)"

    info "Descomprimiendo..."
    # El zip de LDraw descomprime en una carpeta "ldraw/"
    unzip -q "$TMP_ZIP" -d /tmp/ldraw_extract_$$
    # Mover contenido a nuestro destino
    if [ -d "/tmp/ldraw_extract_$$/ldraw" ]; then
        cp -r /tmp/ldraw_extract_$$/ldraw/. "$LDRAW_DEST/"
    else
        cp -r /tmp/ldraw_extract_$$/*. "$LDRAW_DEST/"
    fi
    rm -rf "$TMP_ZIP" "/tmp/ldraw_extract_$$"
    ok "Catálogo descomprimido en $LDRAW_DEST"
fi

# Estadísticas
PARTS=$(ls -1 "$LDRAW_DEST/parts/"*.dat 2>/dev/null | wc -l | tr -d ' ')
PRIMS=$(ls -1 "$LDRAW_DEST/p/"*.dat    2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "  📦 Partes principales: $PARTS"
echo "  🔧 Primitivas:         $PRIMS"
echo "  💾 Tamaño total:       $(du -sh $LDRAW_DEST | cut -f1)"

# ══════════════════════════════════════════════════════════════
# 2. INDEXAR CATÁLOGO (genera catalog_index.json)
# ══════════════════════════════════════════════════════════════
echo -e "\n${BLUE}━━━ 2/3 Indexando catálogo LDraw ━━━${NC}"
python3 blender_pipeline/ldraw_catalog.py --catalog-dir "$LDRAW_DEST" --output "$LDRAW_DEST/catalog_index.json"
ok "catalog_index.json generado"

# ══════════════════════════════════════════════════════════════
# 3. INSTALAR ADDON ldr_tools_blender EN BLENDER
# ══════════════════════════════════════════════════════════════
echo -e "\n${BLUE}━━━ 3/3 Instalando addon ldr_tools_blender en Blender ━━━${NC}"

if [ ! -f "$BLENDER_EXEC" ]; then
    warn "Blender no encontrado en: $BLENDER_EXEC"
    warn "Ajusta BLENDER_PATH en .env y vuelve a ejecutar."
    warn "Puedes instalar Blender con: brew install --cask blender"
    exit 0
fi

info "Usando Blender: $BLENDER_EXEC"
"$BLENDER_EXEC" --background --python scripts/install_blender_addon.py 2>&1 | tail -5
ok "Addon instalado"

echo ""
echo -e "${GREEN}━━━ ✅ LDraw listo para usar ━━━${NC}"
echo ""
echo "  Catálogo: $LDRAW_DEST"
echo "  Índice:   $LDRAW_DEST/catalog_index.json"
echo ""
echo "  Siguiente paso:"
echo "  \$BLENDER_PATH --background --python blender_pipeline/generate_dataset.py -- --num_images 10"
echo ""
