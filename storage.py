"""Storage interface and SQLite implementation for ContextVault.

Local-first by design: all agent memory lives in a single SQLite file.
Swap this module for a Postgres/Supabase backend later without touching
server.py or the context engine — implement MemoryStore the same way.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from context_engine import (
    build_context_pack,
    build_search_doc,
    enrich_memory_fields,
    normalize_kind,
)
from compression import DEFAULT_TOKEN_BUDGET


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _loads_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


@dataclass(frozen=True)
class Memory:
    """A single stored coding-agent memory."""

    id: int
    agent_id: str
    content: str
    kind: str
    tags: list[str]
    paths: list[str]
    symbols: list[str]
    importance: int
    created_at: str
    session_id: str | None = None
    access_count: int = 0
    last_accessed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "content": self.content,
            "kind": self.kind,
            "tags": self.tags,
            "paths": self.paths,
            "symbols": self.symbols,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SessionInfo:
    """Result of starting or resuming a memory session."""

    session_id: str
    agent_id: str
    is_new_session: bool
    is_new_agent: bool
    memory_count: int
    summary: str
    recent_memories: list[Memory]
    context_pack: dict[str, Any]


class MemoryStore(ABC):
    """Backend-agnostic storage API for agent memory."""

    @abstractmethod
    def start_session(
        self,
        agent_id: str,
        session_id: str | None = None,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> SessionInfo:
        """Ensure agent exists; create or resume a session; return context pack."""

    @abstractmethod
    def remember(
        self,
        agent_id: str,
        content: str,
        *,
        session_id: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        paths: list[str] | None = None,
        symbols: list[str] | None = None,
        importance: int = 3,
    ) -> Memory:
        """Persist a structured coding memory for the agent."""

    @abstractmethod
    def get_memory(self, memory_id: int) -> Memory | None:
        """Fetch one memory by id."""

    @abstractmethod
    def list_memories(
        self,
        agent_id: str,
        *,
        limit: int = 20,
        kind: str | None = None,
    ) -> list[Memory]:
        """List recent memories for an agent (newest first)."""

    @abstractmethod
    def count_memories(self, agent_id: str) -> int:
        """Total memories stored for an agent."""

    @abstractmethod
    def update_memory(
        self,
        memory_id: int,
        agent_id: str,
        *,
        content: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        paths: list[str] | None = None,
        symbols: list[str] | None = None,
        importance: int | None = None,
    ) -> Memory:
        """Partial-update a memory owned by *agent_id*; FTS stays in sync."""

    @abstractmethod
    def delete_memory(self, memory_id: int, agent_id: str) -> bool:
        """Delete a memory owned by *agent_id*; False if not found."""

    @abstractmethod
    def mark_accessed(self, memory_ids: list[int]) -> None:
        """Bump access stats after a successful recall."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


class SQLiteStore(MemoryStore):
    """File-based SQLite store — zero network, zero credentials."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Multi-process safe: MCP clients spawn one server process per session,
        # so several processes may share this DB file. WAL allows concurrent
        # readers + one writer, busy_timeout waits instead of erroring, and
        # synchronous=NORMAL keeps durability with WAL's atomic commits.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        sql = schema_path.read_text(encoding="utf-8")
        self._conn.executescript(sql)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for older local vaults."""
        cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        alters: list[str] = []
        if "kind" not in cols:
            alters.append(
                "ALTER TABLE memories ADD COLUMN kind TEXT NOT NULL DEFAULT 'context'"
            )
        if "paths" not in cols:
            alters.append("ALTER TABLE memories ADD COLUMN paths TEXT")
        if "symbols" not in cols:
            alters.append("ALTER TABLE memories ADD COLUMN symbols TEXT")
        if "search_doc" not in cols:
            alters.append(
                "ALTER TABLE memories ADD COLUMN search_doc TEXT NOT NULL DEFAULT ''"
            )
        if "access_count" not in cols:
            alters.append(
                "ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
            )
        if "last_accessed_at" not in cols:
            alters.append("ALTER TABLE memories ADD COLUMN last_accessed_at TIMESTAMP")
        for stmt in alters:
            self._conn.execute(stmt)

        # Backfill search_doc for rows missing it
        rows = self._conn.execute(
            """
            SELECT id, content, kind, tags, paths, symbols
            FROM memories
            WHERE search_doc IS NULL OR search_doc = ''
            """
        ).fetchall()
        for row in rows:
            tags = _loads_list(row["tags"])
            paths = _loads_list(row["paths"] if "paths" in row.keys() else None)
            symbols = _loads_list(row["symbols"] if "symbols" in row.keys() else None)
            kind = row["kind"] if "kind" in row.keys() and row["kind"] else "context"
            doc = build_search_doc(
                content=str(row["content"]),
                kind=str(kind),
                tags=tags,
                paths=paths,
                symbols=symbols,
            )
            self._conn.execute(
                "UPDATE memories SET search_doc = ? WHERE id = ?",
                (doc, row["id"]),
            )

        # Ensure FTS exists with search_doc; rebuild if legacy content-only FTS
        self._ensure_fts()

    def _ensure_fts(self) -> None:
        """Recreate FTS table if it is missing search_doc (legacy vaults)."""
        try:
            info = self._conn.execute("PRAGMA table_info(memories_fts)").fetchall()
        except sqlite3.OperationalError:
            info = []
        col_names = {r["name"] for r in info} if info else set()
        needs_rebuild = not info or "search_doc" not in col_names
        if not needs_rebuild:
            # Also check if FTS is empty while memories exist
            mem_n = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()[
                "n"
            ]
            try:
                fts_n = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM memories_fts"
                ).fetchone()["n"]
            except sqlite3.OperationalError:
                fts_n = 0
            if mem_n > 0 and fts_n == 0:
                needs_rebuild = True
        if not needs_rebuild:
            return

        self._conn.execute("DROP TABLE IF EXISTS memories_fts")
        # Drop old triggers if present
        for name in ("memories_ai", "memories_ad", "memories_au"):
            self._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        self._conn.executescript(
            """
            CREATE VIRTUAL TABLE memories_fts USING fts5(
              search_doc,
              content='memories',
              content_rowid='id',
              tokenize='porter unicode61'
            );
            CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
              INSERT INTO memories_fts(rowid, search_doc) VALUES (new.id, new.search_doc);
            END;
            CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
              INSERT INTO memories_fts(memories_fts, rowid, search_doc)
                VALUES ('delete', old.id, old.search_doc);
            END;
            CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
              INSERT INTO memories_fts(memories_fts, rowid, search_doc)
                VALUES ('delete', old.id, old.search_doc);
              INSERT INTO memories_fts(rowid, search_doc) VALUES (new.id, new.search_doc);
            END;
            """
        )
        # Rebuild from content table
        self._conn.execute(
            """
            INSERT INTO memories_fts(rowid, search_doc)
            SELECT id, search_doc FROM memories
            """
        )

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose connection for ranking queries (same DB file)."""
        return self._conn

    def start_session(
        self,
        agent_id: str,
        session_id: str | None = None,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> SessionInfo:
        agent_id = agent_id.strip()
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")

        is_new_agent = False
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO agents (agent_id, created_at) VALUES (?, ?)",
            (agent_id, _utc_now()),
        )
        # rowcount is 0 when the agent already exists (race-safe: two processes
        # may start() the same new agent_id concurrently — WAL + OR IGNORE).
        is_new_agent = cur.rowcount > 0

        is_new_session = False
        if session_id:
            session_id = session_id.strip()
            cur = self._conn.execute(
                "SELECT session_id, agent_id FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO sessions (session_id, agent_id, started_at, last_active_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, agent_id, _utc_now(), _utc_now()),
                )
                is_new_session = True
            elif row["agent_id"] != agent_id:
                raise ValueError(
                    f"session_id {session_id!r} belongs to agent {row['agent_id']!r}, "
                    f"not {agent_id!r}"
                )
            else:
                self._conn.execute(
                    "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
                    (_utc_now(), session_id),
                )
        else:
            session_id = str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO sessions (session_id, agent_id, started_at, last_active_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, agent_id, _utc_now(), _utc_now()),
            )
            is_new_session = True

        self._conn.commit()

        memory_count = self.count_memories(agent_id)
        # Feed the WHOLE vault to the pack builder — compression is what
        # keeps it small, not truncation, so coverage stays ~100%.
        all_for_pack = self.list_memories(agent_id, limit=10_000)
        pack = build_context_pack(
            agent_id,
            all_for_pack,
            is_new_agent=is_new_agent or memory_count == 0,
            token_budget=token_budget,
        )
        recent = all_for_pack[:5]

        return SessionInfo(
            session_id=session_id,
            agent_id=agent_id,
            is_new_session=is_new_session,
            is_new_agent=is_new_agent,
            memory_count=memory_count,
            summary=pack.summary,
            recent_memories=recent,
            context_pack=pack.to_dict(),
        )

    def remember(
        self,
        agent_id: str,
        content: str,
        *,
        session_id: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        paths: list[str] | None = None,
        symbols: list[str] | None = None,
        importance: int = 3,
    ) -> Memory:
        agent_id = agent_id.strip()
        content = content.strip()
        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not content:
            raise ValueError("content must be a non-empty string")
        if not 1 <= importance <= 5:
            raise ValueError("importance must be an integer from 1 to 5")

        # Race-safe agent upsert (see start_session).
        self._conn.execute(
            "INSERT OR IGNORE INTO agents (agent_id, created_at) VALUES (?, ?)",
            (agent_id, _utc_now()),
        )

        fields = enrich_memory_fields(
            content,
            kind=kind,
            tags=tags,
            paths=paths,
            symbols=symbols,
        )
        created = _utc_now()

        cur = self._conn.execute(
            """
            INSERT INTO memories (
              agent_id, session_id, content, kind, tags, paths, symbols,
              importance, search_doc, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                session_id,
                content,
                fields["kind"],
                json.dumps(fields["tags"]),
                json.dumps(fields["paths"]),
                json.dumps(fields["symbols"]),
                importance,
                fields["search_doc"],
                created,
            ),
        )
        self._conn.commit()
        memory_id = int(cur.lastrowid)

        return Memory(
            id=memory_id,
            agent_id=agent_id,
            session_id=session_id,
            content=content,
            kind=fields["kind"],
            tags=fields["tags"],
            paths=fields["paths"],
            symbols=fields["symbols"],
            importance=importance,
            created_at=created,
            access_count=0,
            last_accessed_at=None,
        )

    def get_memory(self, memory_id: int) -> Memory | None:
        cur = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = cur.fetchone()
        return _row_to_memory(row) if row else None

    def list_memories(
        self,
        agent_id: str,
        *,
        limit: int = 20,
        kind: str | None = None,
    ) -> list[Memory]:
        if kind:
            k = normalize_kind(kind)
            cur = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE agent_id = ? AND kind = ?
                ORDER BY importance DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (agent_id, k, limit),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE agent_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (agent_id, limit),
            )
        return [_row_to_memory(r) for r in cur.fetchall()]

    def count_memories(self, agent_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE agent_id = ?",
            (agent_id,),
        )
        return int(cur.fetchone()["n"])

    def update_memory(
        self,
        memory_id: int,
        agent_id: str,
        *,
        content: str | None = None,
        kind: str | None = None,
        tags: list[str] | None = None,
        paths: list[str] | None = None,
        symbols: list[str] | None = None,
        importance: int | None = None,
    ) -> Memory:
        """Partial update: pass only the fields that change.

        Content changes re-run kind/path/symbol extraction (merged with any
        explicitly passed fields); unmentioned fields keep their values.
        The memories_au trigger keeps the FTS index in sync automatically.
        """
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ? AND agent_id = ?",
            (memory_id, agent_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"memory {memory_id} not found for agent {agent_id!r}")

        existing = _row_to_memory(row)
        if content is not None and not str(content).strip():
            raise ValueError("content must be a non-empty string")
        if importance is not None and not 1 <= int(importance) <= 5:
            raise ValueError("importance must be an integer from 1 to 5")

        new_content = str(content).strip() if content is not None else existing.content
        final_kind = normalize_kind(kind) if kind else existing.kind
        fields = enrich_memory_fields(
            new_content,
            kind=final_kind,
            tags=[str(t).strip() for t in tags] if tags is not None else existing.tags,
            paths=(
                [str(p).replace("\\", "/").strip() for p in paths]
                if paths is not None
                else existing.paths
            ),
            symbols=(
                [str(s).strip() for s in symbols]
                if symbols is not None
                else existing.symbols
            ),
        )
        final_importance = int(importance) if importance is not None else existing.importance

        self._conn.execute(
            """
            UPDATE memories
            SET content = ?, kind = ?, tags = ?, paths = ?, symbols = ?,
                importance = ?, search_doc = ?
            WHERE id = ? AND agent_id = ?
            """,
            (
                new_content,
                fields["kind"],
                json.dumps(fields["tags"]),
                json.dumps(fields["paths"]),
                json.dumps(fields["symbols"]),
                final_importance,
                fields["search_doc"],
                memory_id,
                agent_id,
            ),
        )
        self._conn.commit()
        updated = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return _row_to_memory(updated)

    def delete_memory(self, memory_id: int, agent_id: str) -> bool:
        """Delete one memory (FTS cleaned up by the memories_ad trigger)."""
        cur = self._conn.execute(
            "DELETE FROM memories WHERE id = ? AND agent_id = ?",
            (memory_id, agent_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def mark_accessed(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        now = _utc_now()
        for mid in memory_ids:
            self._conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed_at = ?
                WHERE id = ?
                """,
                (now, mid),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _row_to_memory(row: sqlite3.Row) -> Memory:
    keys = set(row.keys())
    return Memory(
        id=int(row["id"]),
        agent_id=str(row["agent_id"]),
        session_id=row["session_id"],
        content=str(row["content"]),
        kind=str(row["kind"]) if "kind" in keys and row["kind"] else "context",
        tags=_loads_list(row["tags"]),
        paths=_loads_list(row["paths"] if "paths" in keys else None),
        symbols=_loads_list(row["symbols"] if "symbols" in keys else None),
        importance=int(row["importance"]),
        created_at=str(row["created_at"]),
        access_count=int(row["access_count"])
        if "access_count" in keys and row["access_count"] is not None
        else 0,
        last_accessed_at=row["last_accessed_at"] if "last_accessed_at" in keys else None,
    )


def default_db_path() -> Path:
    """Project-root SQLite file — portable local memory for coding agents."""
    return Path(__file__).resolve().parent / "contextvault.db"


def open_store(db_path: str | Path | None = None) -> SQLiteStore:
    """Factory used by server and tests."""
    return SQLiteStore(db_path or default_db_path())
