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
python test_client.py            # engine smoke test (direct calls)
python test_mcp_client.py        # real MCP protocol + concurrent sessions
```

Both scripts use temporary DBs — they never touch your real `contextvault.db`.

## Fully local — and proven

- **Zero network at runtime**: project code imports only stdlib + fastmcp; the only
  third-party deps (`fastmcp`/`mcp`/`httpx`) are used for the stdio transport, never
  for network calls. No embeddings, no API keys, no telemetry.
- **Proven over the wire**: `test_mcp_client.py` spawns the real server as a subprocess
  (exactly how Claude Code connects), completes the MCP handshake, lists tools, and
  round-trips start → remember → recall.
- **Concurrency-safe**: two MCP sessions writing the same DB file at once is the normal
  product case (each client session spawns its own server process). SQLite runs in WAL
  mode with `busy_timeout`, and agent upserts are `INSERT OR IGNORE` — the test runs
  two concurrent sessions against one DB and reconnects a third to verify persistence.

## MCP tools

| Tool | What it does |
|------|----------------|
| `start(agent_id, session_id?, token_budget?)` | Open/resume session; returns a **token-budgeted working context** (verbatim + compressed + coverage index) |
| `remember(content, kind?, tags?, paths?, symbols?, importance?)` | Store a durable coding memory **locally** |
| `recall(query, limit?, kind?, path?)` | Multi-signal retrieval with match reasons |

### Memory kinds (coding ontology)

`decision` · `convention` · `bug` · `file_map` · `api` · `preference` · `gotcha` · `architecture` · `todo` · `context`

### Context engine (small codebases)

For repos under ~50k lines, dense structured memory beats huge chat dumps:

1. **FTS5** (porter tokenizer) over a denormalized `search_doc` (content + kind + tags + paths + symbols)
2. **Path / symbol / tag** boosts — "where is auth?" prefers `file_map` + path hits
3. **Intent routing** — why/bug/style/locate queries boost the right kinds
4. **Importance + usage** — critical and frequently recalled notes stick
5. **Kind diversity** — results aren't five near-duplicate decisions
6. **Token-budgeted working context** — on `start()`, the WHOLE vault is represented: hot
   memories verbatim, everything else extractively compressed, plus a coverage index of
   every stored path/symbol. Nothing silently drops out of the agent's view.

### Token economy (how “all context, fewer tokens” works)

`compression.py` is fully local — no LLM, no embeddings, no network:

- **Offline token estimator** — a deterministic, monotonic token count used as the
  budget currency (swap for tiktoken later without changing callers).
- **Two-tier pack** — up to `MAX_VERBATIM` (4) hot memories stay word-for-word; the rest
  are **extractively compressed**: content is split into path/abbreviation-safe segments,
  the highest-signal sentences (path/symbol/tag/importance-scored) are kept under a
  per-memory budget, and the rest reported as `+N more`.
- **Coverage index** — every stored path/symbol not already mentioned in pack text is
  listed in `index.paths` / `index.symbols`, so the agent can always `recall(path=...)`
  to drill into anything.
- **Redundancy collapse** — near-duplicate memories (high token overlap, or shared path
  with loose overlap) are merged at pack time: paths/symbols/tags union, max importance,
  summed usage. Originals stay in the vault — `recall()` still sees them.
- **Honest accounting** — every pack reports `tokens_used / token_budget` and `coverage`
  (% of memories represented).

Budget knobs: `start(..., token_budget=N)` (256–16000) or env `CONTEXTVAULT_TOKEN_BUDGET`
(default 1200). `recall(..., compact=True)` returns ~220-token distilled snippets when
you want cheap results.

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

Optional smaller/larger working-context budget (still local):

```json
{
  "env": {
    "CONTEXTVAULT_TOKEN_BUDGET": "800"
  }
}
```

### Agent workflow

1. `start` with `agent_id` = project name; read `context_pack` — `verbatim` anchors are
   exact, `compressed` anchors are distilled (call `recall()` for full detail), and
   `index.paths`/`index.symbols` cover every stored path/symbol.  
2. `recall` before non-trivial edits — use `compact=True` when you only need cheap hits  
3. `remember` decisions, file maps, bugs, conventions (with `paths` when you can)

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
storage.py          MemoryStore interface + SQLite (local, WAL)
compression.py      Token estimator, segment splitting, extractive compression, dedupe
context_engine.py   Kinds, extraction, scoring, working-context packs
ranking.py          FTS + multi-signal recall
server.py           FastMCP entrypoint (3 tools)
test_client.py      Engine smoke test (direct calls)
test_mcp_client.py  Real MCP protocol + concurrent-session test
```

## Requirements

- Python 3.11+
- `fastmcp>=2.0.0`

## Token-economy example

A vault of 50 memories collapses to a working context like:

```text
4 verbatim + 46 compressed = 611/1200 tokens, 100% coverage, 3 duplicates merged
index.paths:   src/middleware/auth.ts, lib/db/schema.sql, …
index.symbols: verifyJwt, build_search_doc, …
```

Each compressed anchor carries `+N more` when sentences were dropped, so the agent
knows when to `recall()` for the full note.

## License

MIT (or your choice when you publish).
