# MCP PostgreSQL Setup — LegoVision

Este documento describe cómo configurar el servidor MCP (Model Context Protocol) oficial de PostgreSQL para que asistentes de IA como **Cline** puedan consultar la base de datos local del proyecto.

## 🎯 Qué hace el MCP de PostgreSQL

- Permite a un asistente de IA ejecutar consultas **de solo lectura** (`SELECT`) sobre la BD `legvision`.
- Expone el **esquema** de las tablas como recursos navegables.
- **Seguro por diseño**: cada consulta corre en una transacción `READ ONLY` con `ROLLBACK` al final → no puede modificar datos.

## 📦 Servidor utilizado

- **Paquete**: [`@modelcontextprotocol/server-postgres`](https://www.npmjs.com/package/@modelcontextprotocol/server-postgres) (oficial de Anthropic)
- **Versión**: `0.6.2`
- **Ejecución**: vía `npx` (no requiere instalación global)

## 🔌 Conexión a la BD del proyecto

La BD corre en Docker (ver `docker-compose.yml`):

| Parámetro | Valor |
|---|---|
| Host | `localhost` |
| Puerto | `5434` (mapeado al `5432` del contenedor) |
| Database | `legvision` |
| Usuario | `postgres` |
| Password | `legvision_pass_2024` |

**Cadena de conexión:**
```
postgresql://postgres:legvision_pass_2024@localhost:5434/legvision
```

> Asegúrate de que el contenedor está corriendo:
> ```bash
> docker compose up -d legvision-db
> docker ps --filter name=legvision-postgres
> ```

## ⚙️ Configuración en Cline (VS Code)

Edita el archivo `cline_mcp_settings.json`:

- **macOS**: `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Linux**: `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- **Windows**: `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

Contenido:

```json
{
  "mcpServers": {
    "legvision-postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://postgres:legvision_pass_2024@localhost:5434/legvision"
      ],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

Después: **reinicia Cline** (cierra y reabre la pestaña/extensión).

## ⚙️ Configuración en Claude Desktop (alternativa)

Edita `~/Library/Application Support/Claude/claude_desktop_config.json` con el mismo bloque `mcpServers`.

## ✅ Verificación rápida

1. **Conexión `psql` desde el host:**
   ```bash
   PGPASSWORD="legvision_pass_2024" psql -h localhost -p 5434 -U postgres -d legvision -c "\dt"
   ```

2. **Handshake del servidor MCP:**
   ```bash
   (printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}\n'; sleep 2) \
     | npx -y @modelcontextprotocol/server-postgres "postgresql://postgres:legvision_pass_2024@localhost:5434/legvision"
   ```
   Debe responder con `serverInfo` y `capabilities.tools`.

3. **Dentro de Cline:** pídele algo como _"lista las tablas del schema public"_ y debe usar la herramienta `query` del MCP.

## 🛡️ Notas de seguridad

- El password está en `docker-compose.yml` (entorno **local de desarrollo**); no commitear credenciales reales de producción.
- `cline_mcp_settings.json` vive en tu home, **fuera del repo** → la URL con password no se comparte por git.
- El MCP solo permite `SELECT` (transacción `READ ONLY`).

## 🐛 Troubleshooting

| Problema | Solución |
|---|---|
| `connection refused` | El contenedor no está arriba: `docker compose up -d legvision-db` |
| `password authentication failed` | Revisa `docker-compose.yml` (`POSTGRES_PASSWORD`) y la URL en el JSON |
| `port 5434 in use` | Cambia el mapeo en `docker-compose.yml` y la URL |
| El MCP no aparece en Cline | Reinicia Cline tras editar el JSON; valida JSON con `jq . cline_mcp_settings.json` |
| `npx` muy lento la primera vez | Es normal: descarga el paquete; queda cacheado en `~/.npm/_npx/` |