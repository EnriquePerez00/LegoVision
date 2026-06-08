# LegoVision

Sistema de **vision artificial** para detectar y clasificar piezas LEGO sobre una cinta transportadora negra.
Utiliza YOLO11 para deteccion y embeddings DINOv2 con K-NN consensus voting para clasificacion.

## Pipeline de 2 Fases

**Fase 1 - Deteccion (YOLO11n):** bounding boxes con clase generica (lego_piece / minifigure).

**Fase 2 - Clasificacion (DINOv2 + MLP + K-NN):** identifica la pieza exacta por similitud de embeddings.

## Hardware de Referencia

| Parametro | Valor |
|-----------|-------|
| Camara | Sony IMX264 (Global Shutter) |
| Resolucion | 5MP (2448 x 2048 px) |
| Lente | 12mm C-mount |
| Working Distance | 355 mm |
| FOV | 250 mm (cinta 200mm) |
| Velocidad cinta | max. 83.3 mm/s (5 m/min) |
| Latencia objetivo | menor de 200 ms (5 FPS) |

## Quick Start

### Configuracion inicial (una sola vez)

    cp .env.example .env
    # Editar .env: configurar BLENDER_PATH
    bash scripts/setup_env.sh
    docker compose up -d
    psql -h localhost -p 5434 -U postgres -d legvision -f database/schema.sql
    python scripts/migrate_sets.py

### Arranque del sistema

    ./run.sh

El script levanta: entorno virtual, Docker/Supabase, API FastAPI (:8005) y GUI PyWebView.
Al cerrar la GUI todos los procesos se detienen ordenadamente.

## Estructura del Proyecto

    LegoVision/
      inference/   API FastAPI, detector YOLO11, clasificador DINOv2 K-NN
      gui/         Interfaz grafica PyWebView (SPA HTML/JS + Python bridge)
      database/    Cliente PostgreSQL, schema SQL, catalogo de sets
      training/    Entrenamiento YOLO11, indexacion DINOv2, evaluacion
      scripts/     Scripts Blender para generacion de datos y validacion
      data/        Datos generados (no versionados en git)
      models/      Modelos entrenados (.pt)
      docs/        Documentacion tecnica

## Documentacion

- **[Documentacion Tecnica Completa](docs/SYSTEM_DOCUMENTATION.md)** - Arquitectura, flujos, algoritmos, modelo de datos, API
- [Hardware Setup](docs/hardware_setup.md) - Especificaciones Sony IMX264
- [Latency Budget](docs/latency_budget.md) - Analisis de latencia

## Stack Tecnologico

| Componente | Tecnologia |
|-----------|------------|
| Deteccion | YOLO11n (Ultralytics) |
| Clasificacion | DINOv2 ViT-S/14 + MLP + K-NN |
| API | FastAPI + Uvicorn |
| GUI | PyWebView + HTML/CSS/JS |
| Base de datos | PostgreSQL 16 en Docker (Supabase local) |
| Generacion datos | Blender 4.x con Python + LDraw |

## Base de Datos

- **PostgreSQL:** localhost:5434
- **DB:** legvision
- Gestion: docker compose up/down

## Sets LEGO Soportados

| Set ID | Nombre |
|--------|--------|
| 75078-1 | Imperial Troop Transport (Star Wars) |
| 75280-1 | 501st Legion Clone Troopers |
| 75218-1 | X-Wing Starfighter |
| 75337-1 | AT-TE Walker |
| 10692-1 | Creative Bricks |
