#!/usr/bin/env python3
"""ContextVault MCP server — local coding-agent memory + small-repo context engine.

Five tools: start, remember, recall, update, forget.
All memory is stored in local SQLite (contextvault.db) — no cloud required.
Tuned for small codebases (< ~50k lines): path/symbol-aware ranking and
token-budgeted working-context packs.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import Context, FastMCP

from compression import DEFAULT_TOKEN_BUDGET, extractive_compress
from context_engine import MEMORY_KINDS
from ranking import recall as rank_recall
from storage import MemoryStore, SQLiteStore, default_db_path, open_store

mcp = FastMCP(
    name="contextvault",
    instructions=(
        "ContextVault is a LOCAL persistent memory vault for coding agents on "
        "small codebases (<~50k LOC). Data never leaves the machine (SQLite). "
        "Workflow: call start() once per session to load a token-budgeted "
        "working context; remember() durable decisions, file maps, bugs, "
        "conventions, and gotchas with kind + paths when possible; recall() "
        "before non-trivial work; update() to correct a stale memory; forget() "
        "to delete one. "
        f"Memory kinds: {', '.join(sorted(MEMORY_KINDS))}."
    ),
)

_store: MemoryStore | None = None

# Per-connection session state, keyed by the MCP session id. A stdio client
# gets one process per connection, but HTTP/SSE can multiplex many sessions
# onto one process — so state lives per session, not in module globals.
_session_state: dict[str, dict[str, str | None]] = {}
# Direct-call fallback (tests, embedding) — no MCP Context available.
_legacy_active: dict[str, str | None] = {"agent_id": None, "session_id": None}


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        db = os.environ.get("CONTEXTVAULT_DB") or str(default_db_path())
        _store = open_store(db)
    return _store


def _active(ctx: Context | None) -> dict[str, str | None]:
    """Return the mutable session-state dict for this call."""
    if ctx is not None:
        try:
            sid = ctx.session_id
        except RuntimeError:  # no MCP session yet (e.g. during initialize)
            return _legacy_active
        return _session_state.setdefault(
            sid, {"agent_id": None, "session_id": None}
        )
    return _legacy_active


def _require_session(ctx: Context | None) -> tuple[MemoryStore, str, str]:
    store = _get_store()
    state = _active(ctx)
    agent_id = state.get("agent_id")
    session_id = state.get("session_id")
    if not agent_id or not session_id:
        raise RuntimeError(
            "No active memory session. Call start(agent_id=...) first."
        )
    return store, str(agent_id), str(session_id)


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _resolve_budget(token_budget: int | None) -> int:
    """Working-context token budget: explicit arg > env > default."""
    if token_budget is not None:
        budget = int(token_budget)
    else:
        env_b = os.environ.get("CONTEXTVAULT_TOKEN_BUDGET")
        try:
            budget = int(env_b) if env_b else DEFAULT_TOKEN_BUDGET
        except ValueError:
            budget = DEFAULT_TOKEN_BUDGET
    if not 256 <= budget <= 16_000:
        raise ValueError("token_budget must be an integer from 256 to 16000")
    return budget


@mcp.tool()
def start(
    agent_id: str,
    session_id: str | None = None,
    token_budget: int | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Open or resume a local coding-agent memory session.

    Call once at the start of work. Returns a **token-budgeted working
    context** for small codebases: every stored memory is represented — hot
    memories verbatim, the rest extractively compressed, plus an index of
    every stored path/symbol — so you recover the whole project shape for a
    fraction of the token cost. See context_pack for details.

    Args:
        agent_id: Stable id for this agent or repo context
            (e.g. "claude-code", "cursor", "repo:my-app").
        session_id: Optional existing session to resume. Omit to start fresh.
        token_budget: Max tokens for the working-context pack (256–16000).
            Default 1200; env override CONTEXTVAULT_TOKEN_BUDGET.

    Returns:
        session_id, summary, memory_count, and context_pack (verbatim +
        compressed + coverage index). All data is local SQLite only.
    """
    try:
        if not agent_id or not str(agent_id).strip():
            return _err("agent_id is required and must be non-empty")

        budget = _resolve_budget(token_budget)
        store = _get_store()
        info = store.start_session(
            str(agent_id).strip(),
            session_id,
            token_budget=budget,
        )
        state = _active(ctx)
        state["agent_id"] = info.agent_id
        state["session_id"] = info.session_id

        return {
            "ok": True,
            "session_id": info.session_id,
            "agent_id": info.agent_id,
            "is_new_session": info.is_new_session,
            "is_new_agent": info.is_new_agent,
            "memory_count": info.memory_count,
            "summary": info.summary,
            "context_pack": info.context_pack,
            "storage": "local_sqlite",
            "recent_memories": [m.to_dict() for m in info.recent_memories],
        }
    except ValueError as e:
        return _err(str(e))
    except Exception as e:  # noqa: BLE001
        return _err(f"Failed to start session: {e}")


@mcp.tool()
def remember(
    content: str,
    kind: str | None = None,
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    symbols: list[str] | None = None,
    importance: int | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Store a durable coding memory in the **local** vault.

    Use for facts that must survive across sessions on a small codebase:
    architecture decisions, where files live, bug root causes, conventions,
    API contracts, and gotchas. Prefer concise, self-contained notes.

    Paths and symbols are auto-extracted from content when omitted — but
    passing them explicitly improves recall a lot.

    Args:
        content: Free-text memory (required). Include file paths in the text
            when relevant (e.g. "Auth middleware lives in src/middleware/auth.ts").
        kind: One of: decision, convention, bug, file_map, api, preference,
            gotcha, architecture, todo, context. Inferred if omitted.
        tags: Optional labels (e.g. ["auth", "jwt"]).
        paths: Related file paths (e.g. ["src/middleware/auth.ts"]).
        symbols: Related functions/classes (e.g. ["AuthMiddleware", "verifyJwt"]).
        importance: 1 (low) – 5 (critical). Default 3.

    Returns:
        Stored memory record including inferred kind/paths/symbols.
    """
    try:
        store, agent_id, session_id = _require_session(ctx)
        if not content or not str(content).strip():
            return _err("content is required and must be non-empty")

        imp = 3 if importance is None else int(importance)
        if not 1 <= imp <= 5:
            return _err("importance must be an integer from 1 to 5")

        mem = store.remember(
            agent_id,
            str(content).strip(),
            session_id=session_id,
            kind=kind,
            tags=list(tags) if tags else None,
            paths=list(paths) if paths else None,
            symbols=list(symbols) if symbols else None,
            importance=imp,
        )
        return {
            "ok": True,
            "memory": mem.to_dict(),
            "storage": "local_sqlite",
        }
    except RuntimeError as e:
        return _err(str(e))
    except ValueError as e:
        return _err(str(e))
    except Exception as e:  # noqa: BLE001
        return _err(f"Failed to store memory: {e}")


@mcp.tool()
def update(
    memory_id: int,
    content: str | None = None,
    kind: str | None = None,
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    symbols: list[str] | None = None,
    importance: int | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Correct a stored memory (partial update — pass only what changes).

    Use this when a memory is wrong or stale instead of re-remembering and
    creating a duplicate. Changing content re-runs path/symbol extraction;
    fields you don't pass keep their current values. The FTS index stays in
    sync automatically, so recall() immediately sees the new text.

    Args:
        memory_id: The id of the memory to update (from a prior remember/
            recall/start response).
        content: New text. Empty string is rejected.
        kind: New kind (decision, file_map, bug, …). Defaults to keeping the
            current kind.
        tags: Replace tags. Omit to keep.
        paths: Replace related paths (auto-extraction from new content is
            still merged in). Omit to keep.
        symbols: Replace related symbols. Omit to keep.
        importance: New importance 1 (low) – 5 (critical). Omit to keep.

    Returns:
        The updated memory record.
    """
    try:
        store, agent_id, _session_id = _require_session(ctx)
        if int(memory_id) < 1:
            return _err("memory_id must be a positive integer")
        if all(
            v is None
            for v in (content, kind, tags, paths, symbols, importance)
        ):
            return _err("provide at least one field to update")

        mem = store.update_memory(
            int(memory_id),
            agent_id,
            content=content,
            kind=kind,
            tags=list(tags) if tags is not None else None,
            paths=list(paths) if paths is not None else None,
            symbols=list(symbols) if symbols is not None else None,
            importance=None if importance is None else int(importance),
        )
        return {
            "ok": True,
            "memory": mem.to_dict(),
            "storage": "local_sqlite",
        }
    except (RuntimeError, ValueError) as e:
        return _err(str(e))
    except Exception as e:  # noqa: BLE001
        return _err(f"Failed to update memory: {e}")


@mcp.tool()
def forget(
    memory_id: int,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Delete a stored memory permanently (local vault only).

    Use when a memory is wrong, superseded, or no longer relevant. The FTS
    index is cleaned up automatically. Deletion is permanent — there is no
    undo — and only works on memories owned by the active agent.

    Args:
        memory_id: The id of the memory to delete.

    Returns:
        deleted: true on success; a structured error if the memory does not
        belong to the active agent.
    """
    try:
        store, agent_id, _session_id = _require_session(ctx)
        if int(memory_id) < 1:
            return _err("memory_id must be a positive integer")
        deleted = store.delete_memory(int(memory_id), agent_id)
        if not deleted:
            return _err(f"memory {memory_id} not found for agent {agent_id!r}")
        return {
            "ok": True,
            "deleted": True,
            "memory_id": int(memory_id),
            "storage": "local_sqlite",
        }
    except (RuntimeError, ValueError) as e:
        return _err(str(e))
    except Exception as e:  # noqa: BLE001
        return _err(f"Failed to forget memory: {e}")


@mcp.tool()
def recall(
    query: str,
    limit: int = 5,
    kind: str | None = None,
    path: str | None = None,
    compact: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Retrieve the most relevant **local** coding memories for a task.

    Call before non-trivial edits, when restoring project context, or when
    looking for prior decisions/bugs. Ranking is a small-codebase context
    engine: FTS5 + path/symbol/tag hits + kind/intent + importance (local only).

    Args:
        query: Natural language or keywords
            (e.g. "where is auth middleware", "why sqlite not postgres",
            "login redirect bug").
        limit: Max results (default 5, max 50).
        kind: Optional filter: decision, file_map, bug, convention, etc.
        path: Optional path substring filter (e.g. "src/middleware").
        compact: When true, return extractively-compressed snippets (~220
            tokens each) instead of full content — cheaper for the context
            window; recall() again without compact for full detail.

    Returns:
        Ranked memories with scores, match reasons (`why`), and signal breakdown.
    """
    try:
        store, agent_id, _session_id = _require_session(ctx)
        if not query or not str(query).strip():
            return _err("query is required and must be non-empty")

        lim = int(limit) if limit is not None else 5
        if lim < 1:
            return _err("limit must be at least 1")
        lim = min(lim, 50)

        if not isinstance(store, SQLiteStore):
            return _err("Active store does not support FTS ranking")

        ranked = rank_recall(
            store.connection,
            agent_id,
            str(query).strip(),
            limit=lim,
            kind=kind,
            path=path,
        )

        # Usage feedback for sticky high-value memories
        store.mark_accessed([r.memory.id for r in ranked])

        out: list[dict[str, Any]] = []
        for r in ranked:
            d = r.to_dict()
            if compact:
                text, dropped, _used = extractive_compress(d["content"], 220, memory=r.memory)
                d["content"] = text
                d["compressed"] = True
                d["dropped_segments"] = dropped
            out.append(d)

        return {
            "ok": True,
            "query": query,
            "count": len(ranked),
            "storage": "local_sqlite",
            "memories": out,
        }
    except RuntimeError as e:
        return _err(str(e))
    except ValueError as e:
        return _err(str(e))
    except Exception as e:  # noqa: BLE001
        return _err(f"Failed to recall memories: {e}")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
