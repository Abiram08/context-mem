#!/usr/bin/env python3
"""Real MCP-protocol end-to-end test, including multi-process concurrency.

Spawns server.py as subprocesses over stdio (exactly how Claude Code /
Claude Desktop connect), completes the initialize handshake, lists tools,
and exercises start -> remember -> recall through the actual MCP protocol —
not direct function calls.

It also stresses the product scenario the local SQLite store is built for:
two MCP client sessions running **concurrently against the same DB file**
(each session spawns its own server process), then a third process
reconnecting later to confirm everything persisted.

Usage:
    python test_mcp_client.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles often default to cp1252; force UTF-8 for smoke output.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _text(result: object) -> str:
    """Extract the text payload from an MCP CallToolResult."""
    parts: list[str] = []
    for c in getattr(result, "content", None) or []:
        if hasattr(c, "text"):
            parts.append(str(c.text))
    return "\n".join(parts)


def _json(result: object) -> dict:
    return json.loads(_text(result))


async def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="contextvault-mcp-",
        ignore_cleanup_errors=True,
    ) as tmp:
        db_path = Path(tmp) / "mcp.db"
        os.environ["CONTEXTVAULT_DB"] = str(db_path)

        print("=" * 60)
        print("ContextVault real MCP-protocol round trip (stdio)")
        print(f"DB: {db_path}")
        print("=" * 60)

        from fastmcp import Client

        def _config() -> dict:
            # The exact mcpServers shape users paste into Claude Code /
            # Desktop configs. Client(Path) does NOT forward os.environ to
            # the subprocess, so passing env here keeps tests isolated from
            # the real default vault.
            return {
                "mcpServers": {
                    "contextvault": {
                        "command": sys.executable,
                        "args": [str(ROOT / "server.py")],
                        "env": {**os.environ, "CONTEXTVAULT_DB": str(db_path)},
                    }
                }
            }

        # -- Two concurrent sessions against the same DB file ----------------
        # Each Client spawns its own server subprocess over stdio, so
        # this is two processes writing the same SQLite file at once (WAL).
        # Each session runs a DIFFERENT agent — proving session state is
        # isolated per connection, not a shared module global.
        async with (
            Client(_config()) as client_a,
            Client(_config()) as client_b,
        ):
            tools = await client_a.list_tools()
            names = sorted(t.name for t in tools)
            print(f"tools/list   : {names}")
            assert names == ["forget", "recall", "remember", "start", "update"], names
            for t in tools:
                props = (t.inputSchema or {}).get("properties", {})
                assert "ctx" not in props, f"ctx leaked into {t.name} schema"

            ra, rb = await asyncio.gather(
                client_a.call_tool("start", {"agent_id": "mcp-agent-a"}),
                client_b.call_tool("start", {"agent_id": "mcp-agent-b"}),
            )
            sa, sb = _json(ra), _json(rb)
            assert sa["ok"] and sb["ok"] and sa["storage"] == "local_sqlite"
            pack_a = sa["context_pack"]
            print(
                f"start x2     : concurrent sessions ok (agents isolated) "
                f"(pack {pack_a['tokens_used']}/{pack_a['token_budget']} tokens)"
            )

            # Concurrent writes from both processes.
            await asyncio.gather(
                client_a.call_tool(
                    "remember",
                    {
                        "content": (
                            "Rate limiter lives in src/limits.ts, "
                            "100 req/min per user."
                        ),
                        "kind": "file_map",
                        "paths": ["src/limits.ts"],
                        "importance": 4,
                    },
                ),
                client_b.call_tool(
                    "remember",
                    {
                        "content": (
                            "Decision: sessions in Redis, not in-memory, so "
                            "restarts keep users logged in."
                        ),
                        "kind": "decision",
                        "paths": ["src/session.ts"],
                        "importance": 4,
                    },
                ),
            )
            print("remember x2  : concurrent writes ok")

            # Session isolation: A must see its memory, never B's (and vice
            # versa) — even though both processes share one SQLite file.
            ja, jb = await asyncio.gather(
                client_a.call_tool("recall", {"query": "rate limiter", "limit": 5}),
                client_b.call_tool("recall", {"query": "rate limiter", "limit": 5}),
            )
            ja, jb = _json(ja), _json(jb)
            assert ja["ok"] and jb["ok"]
            a_ids = {m["id"] for m in ja["memories"]}
            b_ids = {m["id"] for m in jb["memories"]}
            assert a_ids and b_ids  # both got hits
            assert a_ids & b_ids == set(), (a_ids, b_ids)  # fully disjoint
            print(
                f"isolation    : A sees {a_ids}, B sees {b_ids} — no cross-agent leaks"
            )

            # Error paths travel over the wire as structured results.
            bad = _json(await client_a.call_tool("recall", {"query": ""}))
            assert bad["ok"] is False and "query is required" in bad["error"]
            print(f"error path   : {bad['error']}")

            # update + forget over the wire, scoped to the owning agent.
            id_a = next(iter(a_ids))
            upd = _json(
                await client_a.call_tool(
                    "update",
                    {
                        "memory_id": id_a,
                        "content": (
                            "Rate limiter moved to src/limits_v2.ts, "
                            "200 req/min per user."
                        ),
                        "importance": 5,
                    },
                )
            )
            assert upd["ok"] and upd["memory"]["importance"] == 5
            assert "src/limits_v2.ts" in upd["memory"]["paths"]
            rehit = _json(
                await client_a.call_tool(
                    "recall", {"query": "limits_v2 200 req", "limit": 3}
                )
            )
            assert rehit["memories"][0]["id"] == id_a
            assert "limits_v2" in rehit["memories"][0]["content"]

            # B cannot update/forget A's memory (cross-agent blocked).
            stolen = _json(
                await client_b.call_tool("forget", {"memory_id": id_a})
            )
            assert stolen["ok"] is False

            gone = _json(
                await client_a.call_tool("forget", {"memory_id": id_a})
            )
            assert gone["ok"] and gone["deleted"] is True
            empty = _json(
                await client_a.call_tool("recall", {"query": "rate limiter", "limit": 5})
            )
            assert empty["count"] == 0
            print("update/forget: wire-level edit + delete ok, cross-agent blocked")

        # -- A third process reconnects later: everything persisted ----------
        async with Client(_config()) as client_c:
            started_b = _json(
                await client_c.call_tool("start", {"agent_id": "mcp-agent-b"})
            )
            pack = started_b["context_pack"]
            assert started_b["ok"]
            assert started_b["memory_count"] == 1  # B's memory survived
            assert pack["coverage"] == 1.0
            stored_paths = {p for a in pack["anchors"] for p in a.get("paths", [])}
            assert stored_paths == {"src/session.ts"}
            started_a = _json(
                await client_c.call_tool("start", {"agent_id": "mcp-agent-a"})
            )
            assert started_a["memory_count"] == 0  # A forgot its memory
            print(
                f"reconnect    : agent B={started_b['memory_count']} memory, "
                f"agent A={started_a['memory_count']} (forgot) — persisted per-agent"
            )

        print("\nMCP protocol round-trip OK (stdio, concurrent sessions, WAL)")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
