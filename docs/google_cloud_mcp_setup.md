# Google Cloud MCP Server — Global Setup on macOS

## Overview

The **Google Cloud MCP Server** (`@google-cloud/mcp-server`) enables AI assistants (Claude Desktop, VS Code with Cline/Copilot) to interact with Google Cloud services directly through the Model Context Protocol (MCP).

Installed version: **0.1.8**  
Installation date: May 2026

---

## Prerequisites Verified

| Tool | Version / Status |
|------|-----------------|
| Node.js | v22.16.0 (via nvm) |
| npm | 10.9.2 |
| gcloud CLI | 523.0.0 |
| gcloud auth | `enriqueperez.b@gmail.com` (active) |
| gcloud project | `legobrickidentifier` |

---

## Installation

```bash
npm install -g @google-cloud/mcp-server
```

**Installed paths:**
- **Binary:** `/Users/I764690/.nvm/versions/node/v22.16.0/bin/google-cloud-mcp`
- **Package:** `/Users/I764690/.nvm/versions/node/v22.16.0/lib/node_modules/@google-cloud/mcp-server/`
- **Entry point:** `dist/index.js`

> ⚠️ **Important:** Because Node.js is managed via **nvm**, you must use the **full absolute path** to `node` and the script in all MCP configurations (not just `google-cloud-mcp`). This ensures the server starts correctly even when launched from applications that don't source your shell profile.

---

## Configuration

### 1. Claude Desktop

**File:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "google-cloud": {
      "command": "/Users/I764690/.nvm/versions/node/v22.16.0/bin/node",
      "args": [
        "/Users/I764690/.nvm/versions/node/v22.16.0/lib/node_modules/@google-cloud/mcp-server/dist/index.js"
      ]
    }
  }
}
```

After editing this file, **restart Claude Desktop** for the changes to take effect.

---

### 2. VS Code — Cline Extension

**File:** `~/Library/Application Support/Code/User/settings.json`

```json
{
  "cline.mcpServers": {
    "google-cloud": {
      "command": "/Users/I764690/.nvm/versions/node/v22.16.0/bin/node",
      "args": [
        "/Users/I764690/.nvm/versions/node/v22.16.0/lib/node_modules/@google-cloud/mcp-server/dist/index.js"
      ],
      "disabled": false,
      "alwaysAllow": []
    }
  }
}
```

---

### 3. VS Code — Native MCP Support (Copilot Agent / future)

**File:** `~/Library/Application Support/Code/User/settings.json`

```json
{
  "mcp": {
    "servers": {
      "google-cloud": {
        "type": "stdio",
        "command": "/Users/I764690/.nvm/versions/node/v22.16.0/bin/node",
        "args": [
          "/Users/I764690/.nvm/versions/node/v22.16.0/lib/node_modules/@google-cloud/mcp-server/dist/index.js"
        ]
      }
    }
  }
}
```

---

## Verification

```bash
# Check the package is installed globally
npm list -g @google-cloud/mcp-server
# → @google-cloud/mcp-server@0.1.8

# Check the binary resolves
which google-cloud-mcp
# → /Users/I764690/.nvm/versions/node/v22.16.0/bin/google-cloud-mcp

# Check the version
google-cloud-mcp --version
# → 0.1.8
```

---

## Updating

When a new version is released:

```bash
npm update -g @google-cloud/mcp-server
```

> After updating, if the Node.js version also changes via nvm, **update the absolute paths** in all three config files above.

---

## Troubleshooting

### MCP server not connecting

1. Verify the binary path is correct for your current nvm node version:
   ```bash
   node --version   # e.g. v22.16.0
   which node       # full path
   ```
2. Update the paths in the config files if the node version has changed.
3. Restart Claude Desktop / reload VS Code window.

### Authentication issues

```bash
gcloud auth list                    # check active account
gcloud auth login                   # re-authenticate if needed
gcloud config set project PROJECT   # set your project
```

### Re-install from scratch

```bash
npm uninstall -g @google-cloud/mcp-server
npm install -g @google-cloud/mcp-server