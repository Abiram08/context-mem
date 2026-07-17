#!/usr/bin/env python3
"""ContextVault MCP server — local coding-agent memory + small-repo context engine.

Three tools: start, remember, recall.
All memory is stored in local SQLite (contextvault.db) — no cloud required.
Tuned for small codebases (< ~50k lines): path/symbol-aware ranking and
dense session context packs.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP

from context_engine import MEMORY_KINDS
from ranking import recall as rank_recall
from storage import MemoryStore, SQLiteStore, default_db_path, open_store

mcp = FastMCP(
    name="contextvault",
    instructions=(
        "ContextVault is a LOCAL persistent memory vault for coding agents on "
        "small codebases (<~50k LOC). Data never leaves the machine (SQLite). "
        "Workflow: call start() once per session to load a context pack; "
        "remember() durable decisions, file maps, bugs, conventions, and gotchas "
        "with kind + paths when possible; call recall() before non-trivial work. "
        f"Memory kinds: {', '.join(sorted(MEMORY_KINDS))}."
    ),
)

_store: MemoryStore | None = None
_active_agent_id: str | None = None
_active_session_id: str | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        db = os.environ.get("CONTEXTVAULT_DB") or str(default_db_path())
        _store = open_store(db)
    return _store


def _require_session() -> tuple[MemoryStore, str, str]:
    store = _get_store()
    if not _active_agent_id or not _active_session_id:
        raise RuntimeError(
            "No active memory session. Call start(agent_id=...) first."
        )
    return store, _active_agent_id, _active_session_id


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


@mcp.tool()
def start(agent_id: str, session_id: str | None = None) -> dict[str, Any]:
    """Open or resume a local coding-agent memory session.

    Call once at the start of work. Returns a **context pack** tuned for small
    codebases: high-signal anchors (architecture, decisions, conventions,
    file maps) so you recover project shape without scanning the whole vault.

    Args:
        agent_id: Stable id for this agent or repo context
            (e.g. "claude-code", "cursor", "repo:my-app").
        session_id: Optional existing session to resume. Omit to start fresh.

    Returns:
        session_id, summary, memory_count, and context_pack (anchors + guidance).
        All data is local SQLite only.
    """
    global _active_agent_id, _active_session_id
    try:
        if not agent_id or not str(agent_id).strip():
            return _err("agent_id is required and must be non-empty")

        store = _get_store()
        info = store.start_session(str(agent_id).strip(), session_id)
        _active_agent_id = info.agent_id
        _active_session_id = info.session_id

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
        store, agent_id, session_id = _require_session()
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
def recall(
    query: str,
    limit: int = 5,
    kind: str | None = None,
    path: str | None = None,
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

    Returns:
        Ranked memories with scores, match reasons (`why`), and signal breakdown.
    """
    try:
        store, agent_id, _session_id = _require_session()
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

        return {
            "ok": True,
            "query": query,
            "count": len(ranked),
            "storage": "local_sqlite",
            "memories": [r.to_dict() for r in ranked],
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
