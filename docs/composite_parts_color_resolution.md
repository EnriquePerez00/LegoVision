# Resolución de colores para piezas compuestas (LDraw shortcuts)

## Problema

Algunas piezas LEGO en LDraw son **shortcuts** (`*c01.dat`, `*c02.dat`, …):
ensamblajes que combinan varios subfiles, cada uno con su propio color
*hardcodeado* en el `.dat`. Ejemplo real:

```
parts/15391c01.dat:
  1 16 ... 15391.dat        ← cuerpo del blaster (color heredable: 16)
  1 72 ... 15392.dat        ← gatillo en LDraw 72 (Light Bluish Gray) — HARDCODED
```

Cuando un pipeline de render trata el shortcut como una pieza simple y
le aplica un único material en función del color BrickLink del padre, los
subfiles con color hardcoded **conservan su color literal**, produciendo
incoherencias (p. ej. la pieza 15391 BL color **11 = Black** se renderizaba
parcialmente gris por culpa del gatillo `72`).

## Regla aplicada (vigente)

```
PARA cada subparte S referenciada en el .dat de la pieza padre P:
  1) Si S tiene LDraw color 16 (placeholder heredable):
       color(S) = color BrickLink del padre (el del inventario)

  2) Si S tiene un color LDraw hardcoded ≠ 16:
       a) Buscar (S.ref) en lego_set_parts (BD Postgres) para set_code
          → usar su color BL
       b) Si no está → re-scrape on-demand de BrickLink (lego_set_parts
          + minifig_parts) y persistir en BD; reintentar (a)
       c) Si tampoco se encuentra → mapear LDraw color_id → BrickLink
          color via inverso de database/color_catalog.json (campo
          ldraw_code)
       d) Fallback final: color del padre

EXCEPCIÓN: minifiguras (refs sw* / fig*) quedan fuera de este flujo —
se gestionan en su pipeline propio (assemble_minifig).
```

Cada decisión se persiste en `subpart_color_overrides` con su `source`
(`inventory`, `bricklink`, `ldraw_map`, `manual`) para auditoría y para
servir como **cache determinística** en futuras invocaciones.

## Implementación

| Componente | Ruta |
|---|---|
| Migración SQL (tabla cache) | `database/migrations/004_add_subpart_color_overrides.sql` |
| Resolver Python | `scripts/ldraw_color_resolver.py` |
| Resolver (subproyecto) | `2camaras_random_pieza_unica/scripts/ldraw_color_resolver.py` |
| Pipeline render (raíz) | `scripts/generate_synthetic_set.py::render_piece_pipeline` |
| Pipeline render (subproyecto) | `2camaras_random_pieza_unica/scripts/generate_synthetic_set.py::render_piece_pipeline` |

### Interacción con el addon `ldr_tools_blender`

Confirmado leyendo `importldr.py`: el addon importa shortcuts como un
**grafo de objetos jerárquicos** — un mesh por subfile, cada uno
nombrado con la base del `.dat` (`15392`, `4073`, …) y con su material
LDraw ya resuelto. Por eso el pipeline ahora:

1. Marca `pre_existing` antes de importar.
2. Tras importar, identifica TODOS los meshes nuevos (no sólo el primero).
3. Para cada mesh extrae `sub_ref` desde su nombre y aplica el material
   correspondiente del `color_map` que devolvió el resolver.
4. Hace `bpy.ops.object.join()` para unir los meshes preservando
   multi-material slots; las físicas trabajan sobre un único cuerpo
   rígido.

## Caso 15391 (verificado)

| Subparte | LDraw | Resolución | Color final |
|---|---|---|---|
| `15391c01` (raíz, padre) | — | inventario | BL 11 (#202020 Black) |
| `15391` (cuerpo) | 16 | hereda padre | BL 11 (#202020 Black) ✅ |
| `15392` (gatillo) | 72 | LDraw→BL via catálogo | BL 85 (#6B6D67 Dark BG) ✅ |
| `4073` (proyectil, sólo `c02`) | 57 | LDraw→BL via catálogo | BL 98 (#EF8E1B Trans-Orange) ✅ |

## CLI de pruebas

```bash
.venv/bin/python scripts/ldraw_color_resolver.py \
    --part_ref 15391c01 \
    --color 11 \
    --set_code 75078-1 \
    --verbose
```

- `--no-scrape` evita el re-scrape on-demand.
- Sin `--set_code` se usan sólo las reglas (1) heredable y (2c) mapeo
  LDraw→BL.

## Tabla `subpart_color_overrides`

```sql
CREATE TABLE subpart_color_overrides (
    parent_ref    TEXT NOT NULL,
    parent_color  TEXT NOT NULL,         -- BL color del padre
    sub_ref       TEXT NOT NULL,
    sub_color     TEXT NOT NULL,         -- BL color resuelto
    sub_color_hex TEXT,
    sub_color_name TEXT,
    source        TEXT NOT NULL CHECK (
                  source IN ('bricklink','ldraw_map','manual','inventory')),
    set_code      TEXT,
    minifig_ref   TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (parent_ref, parent_color, sub_ref)
);
```

Aplicar la migración:

```bash
PGPASSWORD=$SUPABASE_DB_PASSWORD psql \
    -h localhost -p 5434 -U postgres -d legvision \
    -f database/migrations/004_add_subpart_color_overrides.sql
```

## Override manual

Si en algún caso especial el mapeo LDraw→BL no es el correcto (algo
muy raro con piezas oficiales), se puede insertar un override manual:

```sql
INSERT INTO subpart_color_overrides
    (parent_ref, parent_color, sub_ref, sub_color,
     sub_color_hex, sub_color_name, source)
VALUES
    ('15391c01', '11', '15392', '85', '#6B6D67', 'Dark Bluish Gray', 'manual')
ON CONFLICT (parent_ref, parent_color, sub_ref) DO UPDATE SET
    sub_color = EXCLUDED.sub_color,
    source    = 'manual',
    updated_at = NOW();
```

El resolver siempre prioriza la cache antes que cualquier paso del flujo.