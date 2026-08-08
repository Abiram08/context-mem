# CLAUDE.md — ContextVault

Instructions for Claude Code (or any agentic coding tool) building this project from scratch.

## What this is

ContextVault is an MCP (Model Context Protocol) server that gives AI agents persistent
memory across sessions. Any MCP-compatible client (Claude Code, Claude Desktop, OpenCode,
Cursor, etc.) can connect to it and call three tools to start a memory session, save
memories, and recall relevant ones later.

This is a portfolio/demo project. It needs to actually run, be genuinely useful, and be
demoable in under 2 minutes (clone → install → connect → use).

## Core tools (the whole API surface)

1. **`start(agent_id: str, session_id: str | None)`**
   Initializes or resumes a memory context for an agent. Returns a session handle and a
   short summary of what's already known about this agent/context (if anything).

2. **`remember(content: str, tags: list[str] | None, importance: int | None)`**
   Stores a memory. `content` is free text. `tags` are optional labels for filtering.
   `importance` (1-5, default 3) affects ranking in recall.

3. **`recall(query: str, limit: int = 5)`**
   Retrieves the most relevant stored memories for a query, ranked by relevance and
   importance. This is the tool agents call constantly — it needs to be fast and actually
   return useful results, not just keyword-match noise.

Keep the API surface to exactly these three tools for v1. Do not add more tools unless
explicitly asked — the value here is that it's small, sharp, and reliable.

## Tech stack

- **Python 3.11+**
- **FastMCP** for the MCP server layer (`pip install fastmcp`)
- **SQLite** for storage (file-based, zero setup — `contextvault.db` in the project root)
  - Use plain SQLite for v1, not an ORM. Keep the schema visible and simple.
  - Design the storage layer behind a small interface (`storage.py`) so swapping SQLite
    for Postgres/Supabase later is a one-file change, not a rewrite.
- **Embeddings for recall**: use a lightweight local embedding approach first —
  SQLite FTS5 (full-text search) as the baseline ranking signal. If time allows, add a
  simple embedding-based re-rank (e.g. `sentence-transformers` with a small model, or an
  API embedding call) behind a feature flag, so the project still runs with zero external
  API keys out of the box.
- **No web dashboard for v1.** A CLI smoke-test script is enough. Don't build a frontend
  unless explicitly asked — it's not the point of the demo.

## Project structure

```
contextvault/
  server.py          # FastMCP server entrypoint, tool definitions
  storage.py          # Storage interface + SQLite implementation
  compression.py      # Token estimator, segment splitting, extractive compression, dedupe
  context_engine.py   # Kinds, extraction, scoring, token-budgeted working-context packs
  ranking.py          # FTS5 query + relevance/importance scoring for recall
  schema.sql           # SQLite schema (agents, sessions, memories tables)
  test_client.py       # Engine smoke test (direct calls)
  test_mcp_client.py   # Real MCP protocol round trip + concurrent sessions
  README.md            # Setup + usage, written for someone evaluating this in 2 minutes
  pyproject.toml       # or requirements.txt
  CLAUDE.md            # this file
```

## Schema (starting point — adjust as needed)

```sql
CREATE TABLE agents (
  agent_id TEXT PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  agent_id TEXT REFERENCES agents(agent_id),
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT REFERENCES agents(agent_id),
  content TEXT NOT NULL,
  tags TEXT,              -- JSON array as text
  importance INTEGER DEFAULT 3,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', content_rowid='id');
```

## Build order (do this in sequence, don't skip ahead)

1. `schema.sql` + `storage.py` — get SQLite reads/writes working, test with a raw script
   before touching MCP at all.
2. `ranking.py` — FTS5 query wired to `memories_fts`, returning ranked results with a
   combined relevance + importance score.
3. `context_engine.py` — kinds, path/symbol extraction, intent routing, scoring, and the
   **token-budgeted working context** (`build_context_pack`): every memory represented as
   verbatim hot notes + extractively compressed notes + a coverage index of all
   paths/symbols, within a token budget (default 1200, see `compression.py` knobs).
4. `server.py` — wrap the storage/ranking/compression logic in the three FastMCP tools.
   Follow FastMCP's decorator pattern (`@mcp.tool()`).
5. `test_client.py` — a script that: starts a session, remembers 3-4 things, recalls
   against a query, and prints what came back. This is the thing to run to prove it works.
6. Only after all of the above works end-to-end: consider the optional embedding re-rank.

## MCP client config (for the README, and for testing locally)

Claude Code / Desktop config uses the `mcpServers` key:

```json
{
  "mcpServers": {
    "contextvault": {
      "command": "python",
      "args": ["/absolute/path/to/contextvault/server.py"]
    }
  }
}
```

Note: OpenCode uses a different key (`mcp`, not `mcpServers`) and expects `command` as an
array — if you add OpenCode instructions later, don't copy the Claude Desktop block
verbatim.

## Conventions

- Type hints on everything. This is a portfolio piece — code quality is the pitch, not
  just working demo.
- No global mutable state outside the storage layer. Keep `server.py` thin — it should
  just wire tools to `storage.py`/`ranking.py` calls.
- Every tool function gets a docstring MCP can surface to the client describing exactly
  what it does and its parameters — agents calling this tool blind rely on that
  description.
- Errors should return useful messages back through MCP, not stack traces.
- Commit early and often with real messages. This repo's commit history is part of what
  gets evaluated.

## What "done" looks like for v1

- `git clone`, `pip install -r requirements.txt` (or `uv sync`), `python test_client.py`
  runs clean and shows a remember → recall round trip working.
- README has a config block that lets someone plug this into Claude Code in under a
  minute.
- All three tools work when called from an actual MCP client, not just the test script.

## Explicitly out of scope for v1

- Web dashboard / UI
- Multi-user auth
- Postgres/Supabase (keep the swap path open, don't build it now)
- Vector DB (Pinecone, Weaviate, etc.) — FTS5 is enough to prove the concept
