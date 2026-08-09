# ContextVault

**Local MCP memory for coding agents** — with a **small-codebase context engine**.

Any MCP client (Claude Code, Claude Desktop, Cursor, OpenCode, …) can call five tools —
`start`, `remember`, `recall`, `update`, `forget` — and keep durable project knowledge
across sessions.

| Goal | How ContextVault does it |
|------|---------------------------|
| No data breach via a memory cloud | **SQLite on your machine only** (`contextvault.db`) |
| Easy MCP setup | One Python process, no API keys, no Docker |
| Useful on small repos (&lt; ~50k LOC) | Typed memories, path/symbol ranking, session **context packs** |

> Optional later: implement `MemoryStore` against Postgres/Supabase for **shared team** memory. Default remains local.

## Install (download & use)

ContextVault is a standalone MCP server — not an AI, not a harness. Just a context
layer that keeps a lot in mind and serves it to any MCP client.

**From source (this repo):**

```bash
git clone <this-repo> context-mem && cd context-mem
python -m venv .venv
.venv\Scripts\activate        # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Or install it as a package** (works without the repo; the wheel ships the schema
embedded):

```bash
pip install .
# now `contextvault` runs the server; or connect Claude Code to it (see below)
```

**Optional extras:** `pip install .[embed]` pulls sentence-transformers for fully-local
semantic recall (see Integrations below).

## Quick start

```bash
python test_client.py            # engine smoke test (direct calls)
python test_mcp_client.py        # real MCP protocol + concurrent sessions
python verify_local_memory.py    # proves agent context lands in SQLite on disk
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
| `remember(content, kind?, tags?, paths?, symbols?, importance?, dedup?)` | Store a durable coding memory **locally**; near-duplicates auto-merge (mem0-style) |
| `recall(query, limit?, kind?, path?, tags?, compact?)` | Multi-signal retrieval (FTS5 + path/symbol/tag + intent + importance + recency); `tags` is a hard filter; `compact=true` returns ~220-token snippets |
| `update(memory_id, content?, kind?, tags?, paths?, symbols?, importance?)` | **Correct** a stale memory (partial update, pass only what changes) — FTS stays in sync |
| `forget(memory_id)` | **Delete** a memory permanently (scoped to the active agent, no undo) |

### Memory kinds (coding ontology)

`decision` · `convention` · `bug` · `file_map` · `api` · `preference` · `gotcha` · `architecture` · `todo` · `context`

### Context engine (small codebases)

For repos under ~50k lines, dense structured memory beats huge chat dumps:

1. **FTS5** (porter tokenizer) over a denormalized `search_doc` (content + kind + tags + paths + symbols)
2. **Path / symbol / tag** boosts — "where is auth?" prefers `file_map` + path hits
3. **Intent routing** — why/bug/style/locate queries boost the right kinds
4. **Importance + usage + recency** — critical, frequently recalled, and fresh
   notes surface first (the Letta/MemGPT relevance + recency + importance triad)
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
you want cheap results. Token counting defaults to the offline heuristic; set
`CONTEXTVAULT_TOKENIZER=tiktoken` (+ `pip install tiktoken`) to count with the real
cl100k tokenizer instead.

### Memory hygiene (mem0/Letta-inspired)

- **Write-time dedup** — `remember()` first looks for a near-duplicate (same kind +
  high content overlap, or a shared file path with loose overlap) and **merges into it**
  instead of inserting: content is sentence-deduped, paths/symbols/tags are unioned,
  importance is maxed. The response flags `deduplicated: true` + `merged_with`. Pass
  `dedup=false` only when you deliberately want a similar-looking separate note.
- **Audit trail** — every created/updated/merged/deleted change is appended to an
  immutable `memory_events` table (mem0 history / Zep temporal). `start()` returns
  `recent_events` so the agent sees exactly what changed since its last session.
- **Recency signal** — fresh notes and recently-revisited notes rank slightly higher,
  so stale memories sink (Supermemory's "forget what's no longer relevant", done
  locally without an LLM).
- **Search filters** — `recall(..., tags=[...])` is a hard filter (mem0 search filters);
  combines with `kind` and `path`.

> Deliberately *not* copied: mem0's LLM fact-extraction and Zep's knowledge graph — they
> need an LLM/embedding stack and contradict the zero-network promise. This project stays
> deterministic and offline; the trade is that memories must be written in reasonably
> atomic sentences (the tool docs nudge agents to do exactly that).

### Integrations & API keys (optional)

The default vault needs **zero keys and zero network** — recall is a local lexical
engine (FTS5 + signals). If you want semantic recall, opt in with environment
variables on the server process:

| Mode | Env | What it does | Sends data out? |
|------|-----|--------------|-----------------|
| `openai` | `CONTEXTVAULT_EMBEDDINGS=openai`, `OPENAI_API_KEY=...` | OpenAI-compatible `/embeddings` re-rank blended into recall (works with OpenAI, Azure, or any compatible gateway) | ⚠️ yes — memory content goes to the provider |
| `local` | `CONTEXTVAULT_EMBEDDINGS=local` (+ `pip install .[embed]`) | sentence-transformers re-rank on this machine | no — stays local (model downloads once) |

Tuning: `EMBEDDING_MODEL` (default `text-embedding-3-small` / `all-MiniLM-L6-v2`),
`EMBEDDING_BASE_URL` (default `https://api.openai.com/v1`), `EMBEDDING_WEIGHT`
(how much semantic counts vs lexical; default 0.30).

**Privacy note:** `openai` mode sends your notes' text to the embedding provider.
`local` mode keeps everything on this machine. Both are opt-in — the default
configuration never transmits a byte.

Everything above runs **locally**. No embeddings, no network for memory.

## Connect Claude Code to ContextVault

This repo ships a committed **`.mcp.json`** — Claude Code auto-loads it whenever it
runs inside this project. `claude mcp list` should show:

```text
contextvault: <venv-python> <repo>/server.py - ✓ Connected
```

**One-liner for any other project** (keeps this project's paths):

```bash
claude mcp add --transport stdio --scope project contextvault -- \
  Z:/Projects/Working/context-mem/.venv/Scripts/python.exe \
  Z:/Projects/Working/context-mem/server.py
```

### Where the agent's memory lives

By default the vault is a single local SQLite file. The server resolves it in this
order (all local, never a cloud):

1. `CONTEXTVAULT_DB` env — explicit override, always wins.
2. `CLAUDE_PROJECT_DIR` — Claude Code sets this on every spawned MCP server, so
   **each project automatically gets its own vault** next to the code it works on
   (`<project>/contextvault.db`).
3. Repo root (`contextvault.db` next to `server.py`).

So two different projects can connect to the same server install and each get an
isolated, project-scoped memory — no shared context leaking between projects.

The `.mcp.json` in this repo pins an explicit `CONTEXTVAULT_DB` for determinism.

### Verify the local storage promise

```bash
python verify_local_memory.py
```

It reads `.mcp.json`, round-trips start → remember → recall over real stdio MCP,
prints the on-disk SQLite rows + audit trail, then reconnects in a fresh process to
show the context pack rebuilding from disk.

### Agent workflow

1. `start` with `agent_id` = project name; read `context_pack` — `verbatim` anchors are
   exact, `compressed` anchors are distilled (call `recall()` for full detail),
   `index.paths`/`index.symbols` cover every stored path/symbol, and `recent_events`
   shows what changed since your last session.  
2. `recall` before non-trivial edits — use `compact=True` when you only need cheap hits;
   add `tags=[...]` to filter  
3. `remember` decisions, file maps, bugs, conventions (with `paths` when you can) —
   near-duplicates auto-merge, so re-stating a fact updates it instead of duplicating  
4. `update(memory_id, …)` to correct a wrong/stale memory instead of re-remembering  
5. `forget(memory_id)` when a memory is superseded or no longer relevant

Example `remember`:

```text
kind: file_map
paths: ["src/middleware/auth.ts"]
content: JWT middleware lives in src/middleware/auth.ts; verifyJwt checks access tokens (15m).
importance: 5
```

## Project layout

```
schema.sql          SQLite schema + FTS5 (search_doc + memory_events audit trail)
storage.py          MemoryStore interface + SQLite (local, WAL)
compression.py      Token estimator, segment splitting, extractive compression, dedupe
context_engine.py   Kinds, extraction, scoring, working-context packs
ranking.py          FTS + multi-signal recall (+ optional semantic blend)
embeddings.py       Opt-in embedders (openai API / local) for semantic re-rank
server.py           FastMCP entrypoint (5 tools, per-connection state)
test_client.py      Engine smoke test (direct calls)
test_mcp_client.py  Real MCP protocol + concurrent-session test
verify_local_memory.py  Claude Code config + on-disk persistence proof
.mcp.json           Claude Code project-scoped MCP config (auto-loaded)
```

## Requirements

- Python 3.11+
- `fastmcp>=2.0.0`

`pip install .[embed]` adds sentence-transformers for the optional local semantic
re-rank. Everything else is stdlib + fastmcp.

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
