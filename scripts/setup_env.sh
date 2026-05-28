#!/usr/bin/env bash
# =============================================================
# LegoVision — Setup Completo del Entorno
# Ejecutar desde la raíz del proyecto: bash scripts/setup_env.sh
# =============================================================
set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

banner() { echo -e "\n${BLUE}━━━ $1 ━━━${NC}"; }
ok()     { echo -e "${GREEN}✅ $1${NC}"; }
warn()   { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()    { echo -e "${RED}❌ $1${NC}"; exit 1; }

banner "LegoVision — Setup de Entorno"
echo "Directorio: $(pwd)"
echo "Fecha: $(date)"

# --- 1. Verificar requisitos ---
banner "1/7 Verificando requisitos"
command -v brew   >/dev/null || err "Homebrew no encontrado. Instala desde https://brew.sh"
command -v docker >/dev/null || err "Docker no encontrado. Instala Docker Desktop."
command -v git    >/dev/null || ok  "git: $(git --version)"
ok "Homebrew: $(brew --version | head -1)"
ok "Docker: $(docker --version)"

# --- 2. Instalar herramientas CLI ---
banner "2/7 Instalando herramientas CLI"
for pkg in git-lfs wget gh; do
    if brew list "$pkg" &>/dev/null; then
        ok "$pkg ya instalado"
    else
        echo "Instalando $pkg..."
        brew install "$pkg"
        ok "$pkg instalado"
    fi
done

# Git LFS global
git lfs install --skip-smudge 2>/dev/null || true
ok "Git LFS inicializado"

# --- 3. Detectar Blender ---
banner "3/7 Detectando Blender"
BLENDER_APP="/opt/homebrew/Caskroom/blender/5.1.1/Blender.app/Contents/MacOS/Blender"
if [ -f "$BLENDER_APP" ]; then
    ok "Blender encontrado: $BLENDER_APP"
    BLENDER_PATH="$BLENDER_APP"
else
    # Buscar otras versiones
    BLENDER_PATH=$(find /opt/homebrew/Caskroom/blender -name "Blender" -type f 2>/dev/null | head -1)
    if [ -n "$BLENDER_PATH" ]; then
        ok "Blender encontrado: $BLENDER_PATH"
    else
        warn "Blender no encontrado. Instalar con: brew install --cask blender"
        BLENDER_PATH="BLENDER_NOT_FOUND"
    fi
fi

# --- 4. Crear .env ---
banner "4/7 Configurando .env"
if [ -f ".env" ]; then
    warn ".env ya existe — no se sobreescribe"
else
    cp .env.example .env
    # Actualizar path de Blender en .env
    sed -i '' "s|BLENDER_PATH=.*|BLENDER_PATH=$BLENDER_PATH|" .env
    ok ".env creado desde .env.example"
fi

# --- 5. Instalar dependencias Python ---
banner "5/7 Instalando dependencias Python"
if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual (.venv)..."
    python3 -m venv .venv
fi
echo "Activando entorno virtual (.venv)..."
source .venv/bin/activate

pip install --upgrade pip --quiet

# Dependencias de base
pip install --quiet \
    python-dotenv \
    psycopg2-binary \
    requests \
    numpy \
    pillow \
    tqdm

ok "Dependencias base instaladas"

# --- 6. Levantar Docker (Supabase) ---
banner "6/7 Levantando Supabase local (Docker)"
if ! docker info &>/dev/null; then
    err "Docker no está corriendo. Inicia Docker Desktop y vuelve a ejecutar."
fi

docker compose pull --quiet 2>/dev/null || true
docker compose up -d

echo "Esperando que PostgreSQL esté listo..."
for i in $(seq 1 30); do
    if docker compose exec -T legvision-db pg_isready -U postgres -d legvision &>/dev/null; then
        ok "PostgreSQL listo en localhost:5434"
        break
    fi
    sleep 2
    if [ "$i" -eq 30 ]; then
        err "PostgreSQL no respondió en 60 segundos"
    fi
done

echo "Esperando PostgREST..."
sleep 3
if curl -s http://localhost:5437/ &>/dev/null; then
    ok "PostgREST API lista en http://localhost:5437"
else
    warn "PostgREST puede tardar unos segundos más en iniciar"
fi

# --- 7. Verificar conexión DB ---
banner "7/7 Verificando conexión a BD"
python database/supabase_client.py && ok "Conexión a BD verificada" || warn "Error conectando a BD"

# --- Resumen final ---
banner "✅ Setup Completado"
echo ""
echo "  📁 Proyecto:      $(pwd)"
echo "  🐘 PostgreSQL:    localhost:5434 (DB: legvision)"
echo "  🔌 PostgREST API: http://localhost:5437"
echo "  📚 Swagger UI:    http://localhost:5438"
if [ -n "$BLENDER_PATH" ] && [ "$BLENDER_PATH" != "BLENDER_NOT_FOUND" ]; then
echo "  🎨 Blender:       $BLENDER_PATH"
fi
echo ""
echo "  Próximos pasos:"
echo "  1. bash scripts/download_ldraw.sh   # Descargar catálogo LDraw"
echo "  2. blender_pipeline/generate_dataset.py  # Generar dataset"
echo ""
