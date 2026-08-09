#!/usr/bin/env python3
"""Smoke test for ContextVault context engine — start → remember → recall.

Runs against a temporary SQLite DB (does not touch production contextvault.db).
Usage:
    python test_client.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _reset_session() -> None:
    """Forget the active session between test phases (direct-call mode)."""
    import server  # noqa: PLC0415 - lazy import keeps the DB env set first

    server._session_state.clear()
    server._legacy_active.update({"agent_id": None, "session_id": None})



def main() -> int:
    # Windows consoles often default to cp1252; force UTF-8 for smoke output.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    with tempfile.TemporaryDirectory(
        prefix="contextvault-test-",
        ignore_cleanup_errors=True,
    ) as tmp:
        db_path = Path(tmp) / "test.db"
        os.environ["CONTEXTVAULT_DB"] = str(db_path)

        import server
        import storage
        from context_engine import extract_paths, infer_kind

        server._store = None
        _reset_session()

        print("=" * 60)
        print("ContextVault context-engine smoke test")
        print(f"DB: {db_path}")
        print("=" * 60)

        # --- start ---
        print("\n[1] start(agent_id='coding-agent-demo')")
        started = server.start(agent_id="coding-agent-demo")
        _assert_ok(started, "start")
        assert started["storage"] == "local_sqlite"
        assert "context_pack" in started
        print(f"    session_id : {started['session_id']}")
        print(f"    storage    : {started['storage']}")
        print(f"    summary    : {started['summary'][:100]}...")
        pack0 = started["context_pack"]
        assert pack0["token_budget"] == 1200
        assert pack0["tokens_used"] <= pack0["token_budget"] + 32
        assert pack0["coverage"] == 1.0
        print(f"    pack       : {pack0['tokens_used']}/{pack0['token_budget']} tokens")
        print(f"                 coverage={pack0['coverage']} verbatim={len(pack0['verbatim'])} compressed={len(pack0['compressed'])}")

        # --- remember (structured coding memories) ---
        samples = [
            {
                "content": (
                    "Auth uses JWT access tokens (15m) + httpOnly refresh cookies. "
                    "Middleware lives in src/middleware/auth.ts."
                ),
                "kind": "file_map",
                "tags": ["auth", "jwt"],
                "paths": ["src/middleware/auth.ts"],
                "symbols": ["verifyJwt"],
                "importance": 5,
            },
            {
                "content": (
                    "Decision: keep agent memory in local SQLite (contextvault.db), "
                    "not Supabase, so coding-agent data never leaves the machine. "
                    "Storage is behind MemoryStore for a future optional remote backend."
                ),
                "kind": "decision",
                "tags": ["sqlite", "privacy", "architecture"],
                "paths": ["storage.py", "schema.sql"],
                "importance": 5,
            },
            {
                "content": (
                    "Recall ranking is a multi-signal context engine: FTS5 porter + "
                    "path/symbol/tag hits + kind/intent + importance. No embedding API."
                ),
                "kind": "architecture",
                "tags": ["fts5", "ranking", "context-engine"],
                "paths": ["ranking.py", "context_engine.py"],
                "importance": 4,
            },
            {
                "content": (
                    "User prefers conventional commits and type hints on all Python APIs. "
                    "Keep the MCP tool surface to start/remember/recall only."
                ),
                "kind": "convention",
                "tags": ["conventions", "mcp"],
                "importance": 3,
            },
            {
                "content": (
                    "Gotcha: on Windows, close SQLite connections before deleting "
                    "temp DB directories or cleanup raises WinError 32."
                ),
                "kind": "gotcha",
                "tags": ["windows", "sqlite"],
                "importance": 4,
            },
            {
                "content": (
                    "Bug fixed: single-result FTS normalization set relevance to 0 "
                    "when max_rel == min_rel; now credit full relevance."
                ),
                "kind": "bug",
                "tags": ["ranking", "fts"],
                "paths": ["ranking.py"],
                "importance": 3,
            },
        ]

        print("\n[2] remember() x 6 (structured kinds/paths)")
        for i, s in enumerate(samples, 1):
            result = server.remember(
                content=s["content"],
                kind=s.get("kind"),
                tags=s.get("tags"),
                paths=s.get("paths"),
                symbols=s.get("symbols"),
                importance=s.get("importance"),
            )
            _assert_ok(result, f"remember#{i}")
            mem = result["memory"]
            print(
                f"    id={mem['id']} kind={mem['kind']:<12} "
                f"paths={mem['paths'][:2]} imp={mem['importance']}"
            )
            assert mem["kind"] == s["kind"]

        # Auto-extract paths from content when omitted
        auto = server.remember(
            content="Entrypoint is server.py; schema lives in schema.sql for the vault."
        )
        _assert_ok(auto, "remember-auto")
        assert "server.py" in auto["memory"]["paths"] or infer_kind(auto["memory"]["content"])
        print(f"    auto-extract paths={auto['memory']['paths']} kind={auto['memory']['kind']}")

        # --- recall with intent + path awareness ---
        cases = [
            ("Where is auth middleware?", "file_map", "src/middleware"),
            ("why local sqlite not supabase", "decision", None),
            ("FTS5 ranking context engine", None, "ranking.py"),
            ("windows sqlite temp cleanup gotcha", "gotcha", None),
            ("conventional commits type hints", "convention", None),
        ]

        print("\n[3] recall() - small-codebase context engine")
        for q, expect_kind, path_filter in cases:
            result = server.recall(query=q, limit=3, path=path_filter)
            _assert_ok(result, f"recall:{q}")
            assert result["storage"] == "local_sqlite"
            print(f"\n    query: {q!r}" + (f" path={path_filter!r}" if path_filter else ""))
            print(f"    hits : {result['count']}")
            assert result["count"] >= 1, f"expected hits for {q!r}"
            top = result["memories"][0]
            print(
                f"      top kind={top['kind']:<12} score={top['score']:.3f} "
                f"why={top.get('why')}"
            )
            snippet = top["content"][:70].replace("\n", " ")
            print(f"      {snippet}...")
            if expect_kind:
                kinds = {m["kind"] for m in result["memories"]}
                assert expect_kind in kinds or top["kind"] == expect_kind, (
                    f"expected kind {expect_kind} in results for {q!r}, got {kinds}"
                )

        # Engine unit checks
        assert "src/middleware/auth.ts" in extract_paths(
            "see src/middleware/auth.ts for JWT"
        )

        # --- compact recall (token-cheap snippets) ---
        compact = server.recall(
            query="where is auth middleware", limit=3, compact=True
        )
        _assert_ok(compact, "recall-compact")
        assert compact["memories"][0]["compressed"] is True
        print("    compact recall: snippets compressed ok")

        # --- resume + context pack ---
        print("\n[4] resume agent - context pack anchors")
        _reset_session()
        resumed = server.start(agent_id="coding-agent-demo")
        _assert_ok(resumed, "resume")
        assert resumed["is_new_agent"] is False
        assert resumed["memory_count"] >= 6
        pack = resumed["context_pack"]
        assert pack["memory_count"] >= 6
        assert len(pack["anchors"]) >= 6
        assert len(pack["verbatim"]) >= 1
        assert len(pack["compressed"]) >= 1
        assert len(pack["anchors"]) == len(pack["verbatim"]) + len(pack["compressed"])
        assert pack["coverage"] >= 0.99
        assert pack["tokens_used"] <= pack["token_budget"] + 32
        # Every stored path is reachable: mentioned in pack text or in the index.
        pack_text = " ".join(a.get("content", "") for a in pack["anchors"]).lower()
        stored_paths = {p for s in samples if s.get("paths") for p in s["paths"]}
        for p in stored_paths:
            assert (
                p in pack["index"]["paths"] or p in pack_text
            ), f"stored path {p!r} not reachable in pack"
        print(f"    memory_count: {resumed['memory_count']}")
        print(f"    by_kind     : {pack['by_kind']}")
        print(f"    tokens      : {pack['tokens_used']}/{pack['token_budget']}")
        print(f"    coverage    : {pack['coverage']:.0%}  verbatim={len(pack['verbatim'])} compressed={len(pack['compressed'])} index_paths={len(pack['index']['paths'])}")
        print(f"    summary     : {pack['summary'][:150]}...")

        # --- update & forget ---
        print("\n[5] update() and forget()")

        upd = server.remember(
            content="Old note: caching lives in src/cache.ts (in-memory, no TTL).",
            kind="file_map",
            paths=["src/cache.ts"],
            importance=2,
        )
        _assert_ok(upd, "remember-for-update")
        mid = upd["memory"]["id"]

        # full update: content + paths + importance (kind preserved)
        changed = server.update(
            memory_id=mid,
            content="New: caching moved to src/redis_cache.ts with a 60s TTL.",
            paths=["src/redis_cache.ts"],
            importance=4,
        )
        _assert_ok(changed, "update")
        assert changed["memory"]["content"].startswith("New: caching")
        assert "src/redis_cache.ts" in changed["memory"]["paths"]
        assert changed["memory"]["importance"] == 4
        assert changed["memory"]["kind"] == "file_map"

        # recall must see the NEW text (FTS reindexed by trigger)
        hit = server.recall(query="redis cache ttl", limit=1)
        assert hit["memories"][0]["id"] == mid
        assert "redis_cache.ts" in hit["memories"][0]["content"]

        # partial update: only importance
        bumped = server.update(memory_id=mid, importance=5)
        _assert_ok(bumped, "update-importance")
        assert bumped["memory"]["importance"] == 5
        assert bumped["memory"]["content"].startswith("New: caching")

        # validation
        assert server.update(memory_id=mid, importance=9)["ok"] is False
        assert server.update(memory_id=mid)["ok"] is False  # no fields
        assert server.update(memory_id=999_999, content="x")["ok"] is False
        assert server.update(memory_id=0, content="x")["ok"] is False

        # cross-agent protection
        _reset_session()
        server.start(agent_id="other-agent")
        assert server.update(memory_id=mid, content="hijack")["ok"] is False
        assert server.forget(memory_id=mid)["ok"] is False

        # back to the owning agent: forget works, then is gone
        _reset_session()
        server.start(agent_id="coding-agent-demo")
        gone = server.forget(memory_id=mid)
        _assert_ok(gone, "forget")
        assert gone["deleted"] is True
        gone_again = server.forget(memory_id=mid)
        assert gone_again["ok"] is False
        print("    update      : content/paths/importance + partial + validation ok")
        print("    forget      : deleted, second forget errors, cross-agent blocked")

        # --- errors ---
        print("\n[6] error handling")
        bad = server.remember(content="")
        assert bad["ok"] is False
        print(f"    empty content -> {bad['error']}")

        _reset_session()
        no_sess = server.recall(query="anything")
        assert no_sess["ok"] is False
        print(f"    no session    -> {no_sess['error']}")

        store = storage.open_store(db_path)
        assert store.count_memories("coding-agent-demo") >= 6
        store.close()

        if server._store is not None:
            server._store.close()
            server._store = None
        _reset_session()

        # --- compression units (offline, deterministic) ---
        print("\n[7] compression units")
        from compression import (
            collapse_redundant,
            estimate_tokens,
            extractive_compress,
            split_segments,
        )
        from storage import Memory

        assert estimate_tokens("") == 0
        assert estimate_tokens("a" * 500) > estimate_tokens("a" * 50)
        segs = split_segments(
            "Auth middleware lives in src/middleware/auth.ts. "
            "It verifies JWT (15m). e.g. one more sentence."
        )
        assert len(segs) >= 2, segs
        assert any("src/middleware/auth.ts" in s for s in segs)
        text, dropped, used = extractive_compress(
            "First signal sentence. Second is filler. Third filler too.", 12
        )
        assert used <= 12 + 4, (text, used)
        m1 = Memory(
            id=1, agent_id="a",
            content="The auth middleware lives in src/middleware/auth.ts and checks JWT access tokens.",
            kind="file_map", tags=[], paths=["src/middleware/auth.ts"],
            symbols=["verifyJwt"], importance=4,
            created_at="2026-01-01 00:00:00",
        )
        m2 = Memory(
            id=2, agent_id="a",
            content="The auth middleware is in src/middleware/auth.ts; it verifies JWT access tokens.",
            kind="file_map", tags=[], paths=["src/middleware/auth.ts"],
            symbols=[], importance=3,
            created_at="2026-01-02 00:00:00",
        )
        merged, removed = collapse_redundant([m1, m2])
        assert removed == 1 and len(merged) == 1
        assert "verifyJwt" in merged[0].symbols  # union preserved
        print(f"    estimate_tokens monotonic   : ok")
        print(f"    split_segments path-safe    : {len(segs)} segments")
        print(f"    extractive_compress budget  : {used} tokens, {dropped} dropped")
        print(f"    collapse_redundant          : {removed} duplicate merged (symbol union kept)")

        print("\n" + "=" * 60)
        print("All checks passed (local storage + context engine)")
        print("=" * 60)
        return 0


def _assert_ok(result: dict, label: str) -> None:
    if not result.get("ok"):
        raise AssertionError(f"{label} failed: {result}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        raise
