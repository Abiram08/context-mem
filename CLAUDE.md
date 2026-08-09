# CLAUDE.md — ContextVault

Instructions for Claude Code (or any agentic coding tool) building this project from scratch.

## What this is

ContextVault is an MCP (Model Context Protocol) server that gives AI agents persistent
memory across sessions. Any MCP-compatible client (Claude Code, Claude Desktop, OpenCode,
Cursor, etc.) can connect to it and call five tools — `start`, `remember`, `recall`,
`update`, `forget` — to run a memory session, save and correct memories, and recall
relevant ones later.

This is a portfolio/demo project. It needs to actually run, be genuinely useful, and be
demoable in under 2 minutes (clone → install → connect → use).

## Using your memory vault (Claude Code sessions in this repo)

This repo's `.mcp.json` registers ContextVault as an MCP server, so Claude Code
sessions here get five tools: `start`, `remember`, `recall`, `update`, `forget`.

Recommended rhythm:

1. **`start(agent_id=...)`** at the beginning of a session → read `context_pack`:
   `verbatim` anchors are exact, `compressed` anchors are distilled (call `recall()`
   for full detail), `index.paths`/`index.symbols` cover every stored path/symbol,
   and `recent_events` shows what changed since the last session.
2. **`recall(query, compact=True)`** before non-trivial edits — cheap hits, then
   `recall()` without compact when you need full content.
3. **`remember(...)`** decisions, file maps, bugs, conventions, gotchas as they're
   discovered — near-duplicates auto-merge, so restating a fact updates it.
4. **`update(memory_id, ...)`** to correct a stale note; **`forget(memory_id)`** when
   one is superseded.

The vault is a single local SQLite file (default `contextvault.db` next to
`server.py`, or per-project via `CLAUDE_PROJECT_DIR`). Never leak it to a cloud.

## Core tools (the API surface)

1. **`start(agent_id: str, session_id: str | None, token_budget: int | None)`**
   Initializes or resumes a memory context for an agent. Returns a session handle and a
   token-budgeted **working context** (verbatim + compressed + coverage index) covering
   everything already known about this agent/context.

2. **`remember(content: str, kind: str | None, tags: list[str] | None, paths: list[str] | None, symbols: list[str] | None, importance: int | None, dedup: bool = True)`**
   Stores a memory. `content` is free text; paths/symbols are auto-extracted. `kind` is
   a coding-memory type (decision, file_map, bug, …); `importance` (1-5, default 3)
   affects ranking in recall. Near-duplicates **auto-merge** into the existing memory
   (mem0-style write-time dedup) unless `dedup=False`.

3. **`recall(query: str, limit: int = 5, kind: str | None, path: str | None, tags: list[str] | None, compact: bool)`**
   Retrieves the most relevant stored memories for a query, ranked by relevance,
   recency, and importance (the Letta triad). This is the tool agents call constantly —
   it needs to be fast and actually return useful results, not just keyword-match noise.
   `tags` is a hard filter; `compact=true` returns distilled snippets to save tokens.

4. **`update(memory_id: int, …)`** — partial-update a memory (correct stale notes instead
   of duplicating). Content changes re-run extraction; FTS stays in sync via trigger.
   Every change is logged to the `memory_events` audit trail.

5. **`forget(memory_id: int)`** — delete a memory permanently, scoped to the active agent
   (also logged to the audit trail).

Keep the surface small and sharp; add more tools only when the user explicitly asks.

## Tech stack

- **Python 3.11+**
- **FastMCP** for the MCP server layer (`pip install fastmcp`)
- **SQLite** for storage (file-based, zero setup — `contextvault.db` in the project root)
  - Use plain SQLite for v1, not an ORM. Keep the schema visible and simple.
  - Design the storage layer behind a small interface (`storage.py`) so swapping SQLite
    for Postgres/Supabase later is a one-file change, not a rewrite.
- **Embeddings for recall**: SQLite FTS5 (full-text search) is the baseline ranking
  signal and the default — zero keys, zero network. An OPTIONAL semantic re-rank exists
  in `embeddings.py`, behind `CONTEXTVAULT_EMBEDDINGS=openai|local` (env), so the project
  still runs with zero external API keys out of the box. Default must never regress to
  requiring embeddings.
- **No web dashboard for v1.** A CLI smoke-test script is enough. Don't build a frontend
  unless explicitly asked — it's not the point of the demo.

## Project structure

```
contextvault/
  server.py          # FastMCP entrypoint, tool definitions (5 tools)
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
6. Done — `embeddings.py` ships an opt-in semantic re-rank (openai API or local).

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
- All five tools work when called from an actual MCP client, not just the test script
  (`test_mcp_client.py` proves it over stdio, including concurrent sessions).

## Explicitly out of scope for v1

- Web dashboard / UI
- Multi-user auth
- Postgres/Supabase (keep the swap path open, don't build it now)
- Vector DB (Pinecone, Weaviate, etc.) — FTS5 is enough to prove the concept
