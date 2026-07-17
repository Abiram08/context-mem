# ContextVault

**Local MCP memory for coding agents** — with a **small-codebase context engine**.

Any MCP client (Claude Code, Claude Desktop, Cursor, OpenCode, …) can call three tools — `start`, `remember`, `recall` — and keep durable project knowledge across sessions.

| Goal | How ContextVault does it |
|------|---------------------------|
| No data breach via a memory cloud | **SQLite on your machine only** (`contextvault.db`) |
| Easy MCP setup | One Python process, no API keys, no Docker |
| Useful on small repos (&lt; ~50k LOC) | Typed memories, path/symbol ranking, session **context packs** |

> Optional later: implement `MemoryStore` against Postgres/Supabase for **shared team** memory. Default remains local.

## Quick start

```bash
git clone <this-repo> context-mem
cd context-mem
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python test_client.py
```

## MCP tools

| Tool | What it does |
|------|----------------|
| `start(agent_id, session_id?)` | Open/resume session; returns a **context pack** (architecture, decisions, file maps, …) |
| `remember(content, kind?, tags?, paths?, symbols?, importance?)` | Store a durable coding memory **locally** |
| `recall(query, limit?, kind?, path?)` | Multi-signal retrieval with match reasons |

### Memory kinds (coding ontology)

`decision` · `convention` · `bug` · `file_map` · `api` · `preference` · `gotcha` · `architecture` · `todo` · `context`

### Context engine (small codebases)

For repos under ~50k lines, dense structured memory beats huge chat dumps:

1. **FTS5** (porter tokenizer) over a denormalized `search_doc` (content + kind + tags + paths + symbols)
2. **Path / symbol / tag** boosts — “where is auth?” prefers `file_map` + path hits
3. **Intent routing** — why/bug/style/locate queries boost the right kinds
4. **Importance + usage** — critical and frequently recalled notes stick
5. **Kind diversity** — results aren’t five near-duplicate decisions
6. **Session context pack** — on `start()`, load a balanced set of anchors into the agent’s working set

Everything above runs **locally**. No embeddings, no network for memory.

## Connect to Claude Code / Desktop

Use absolute paths:

```json
{
  "mcpServers": {
    "contextvault": {
      "command": "Z:/Projects/Working/context-mem/.venv/Scripts/python.exe",
      "args": ["Z:/Projects/Working/context-mem/server.py"]
    }
  }
}
```

Optional custom DB path (still local):

```json
{
  "env": {
    "CONTEXTVAULT_DB": "C:/Users/you/.contextvault/my-project.db"
  }
}
```

### Agent workflow

1. `start` with `agent_id` = project name  
2. Read `context_pack.anchors` as durable truth  
3. `recall` before non-trivial edits  
4. `remember` decisions, file maps, bugs, conventions (with `paths` when you can)

Example `remember`:

```text
kind: file_map
paths: ["src/middleware/auth.ts"]
content: JWT middleware lives in src/middleware/auth.ts; verifyJwt checks access tokens (15m).
importance: 5
```

## Project layout

```
schema.sql          SQLite schema + FTS5 (search_doc)
storage.py          MemoryStore interface + SQLite (local)
context_engine.py   Kinds, extraction, scoring, context packs
ranking.py          FTS + multi-signal recall
server.py           FastMCP entrypoint (3 tools)
test_client.py      End-to-end smoke test
```

## Requirements

- Python 3.11+
- `fastmcp>=2.0.0`

## License

MIT (or your choice when you publish).
