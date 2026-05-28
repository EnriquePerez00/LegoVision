#!/usr/bin/env bash
# =============================================================
# LegoVision — Script de Ejecución Unificado
# Puerto API: 8005 (configurable en .env → API_PORT)
# Ejecutar desde la raíz: ./run.sh
# =============================================================
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Puerto de la API — default 8005, sobreescribible en .env
API_PORT=8005
if [ -f ".env" ]; then
    ENV_PORT=$(grep "^API_PORT=" .env | cut -d'=' -f2 | tr -d ' #' | head -1)
    [ -n "$ENV_PORT" ] && API_PORT="$ENV_PORT"
fi

echo -e "${BLUE}━━━ LegoVision — Inicializando Aplicación ━━━${NC}"
echo -e "${BLUE}    API Inferencia → http://127.0.0.1:${API_PORT}${NC}"
echo -e "${BLUE}    API Docs       → http://127.0.0.1:${API_PORT}/docs${NC}"

# 1. Asegurar y activar el entorno virtual
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Entorno virtual (.venv) no encontrado. Creándolo...${NC}"
    python3 -m venv .venv
fi

echo -e "${GREEN}✅ Activando entorno virtual (.venv)...${NC}"
source .venv/bin/activate

# 2. Instalar dependencias solo si requirements.txt cambió o nunca se instalaron
MARKER=".venv/.deps_installed"
if [ ! -f "$MARKER" ] || [ "requirements.txt" -nt "$MARKER" ]; then
    echo -e "${BLUE}ℹ️  Instalando dependencias (primera vez o requirements.txt actualizado)...${NC}"
    pip install -r requirements.txt --quiet
    touch "$MARKER"
    echo -e "${GREEN}✅ Dependencias instaladas.${NC}"
else
    echo -e "${GREEN}✅ Dependencias ya instaladas (sin cambios).${NC}"
fi

# 3. Levantar Supabase (Docker) si no está activo
if docker compose ps 2>/dev/null | grep -q "Up"; then
    echo -e "${GREEN}✅ Base de datos (Supabase local) ya está corriendo.${NC}"
else
    echo -e "${YELLOW}⚠️  Levantando base de datos Supabase en Docker...${NC}"
    docker compose up -d
    echo "Esperando que PostgreSQL esté listo..."
    sleep 5
fi

# 4. Liberar el puerto si está ocupado por un proceso anterior
OLD_PID=$(lsof -ti "tcp:${API_PORT}" 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo -e "${YELLOW}⚠️  Puerto ${API_PORT} ocupado (PID: $OLD_PID). Liberando...${NC}"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

# 5. Lanzar API de Inferencia en segundo plano
echo -e "${BLUE}🚀 Arrancando API de Inferencia en puerto ${API_PORT}...${NC}"
echo -e "${YELLOW}   (Primera vez: ~30s para descargar yolov8n.pt si no hay modelo entrenado)${NC}"
python -m uvicorn inference.api:app \
    --host 127.0.0.1 \
    --port "${API_PORT}" \
    > /tmp/legovision_api.log 2>&1 &
API_PID=$!

# Función de limpieza al salir
cleanup() {
    echo -e "\n${YELLOW}🛑 Deteniendo API (PID: $API_PID)...${NC}"
    kill "$API_PID" 2>/dev/null || true
    echo -e "${GREEN}✅ Finalizado.${NC}"
}
trap cleanup EXIT INT TERM

# 6. Esperar a que el API responda (máximo 90s)
echo "Esperando que el API esté lista..."
for i in $(seq 1 90); do
    if curl -s "http://127.0.0.1:${API_PORT}/" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API lista en http://127.0.0.1:${API_PORT}${NC}"
        echo -e "${GREEN}   Docs en   http://127.0.0.1:${API_PORT}/docs${NC}"
        break
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "   ... esperando API (${i}s)"
    fi
    sleep 1
    if [ "$i" -eq 90 ]; then
        echo -e "${RED}❌ El API no respondió en 90s. Últimas líneas del log:${NC}"
        tail -20 /tmp/legovision_api.log
        exit 1
    fi
done

# 7. Lanzar GUI en el hilo principal
echo -e "${GREEN}🎨 Lanzando Interfaz Gráfica (PyWebView)...${NC}"
python gui/app.py
