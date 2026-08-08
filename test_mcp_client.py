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
        async with (
            Client(_config()) as client_a,
            Client(_config()) as client_b,
        ):
            tools = await client_a.list_tools()
            names = sorted(t.name for t in tools)
            print(f"tools/list   : {names}")
            assert names == ["recall", "remember", "start"], names

            ra, rb = await asyncio.gather(
                client_a.call_tool("start", {"agent_id": "mcp-demo"}),
                client_b.call_tool("start", {"agent_id": "mcp-demo"}),
            )
            sa, sb = _json(ra), _json(rb)
            assert sa["ok"] and sb["ok"] and sa["storage"] == "local_sqlite"
            pack_a = sa["context_pack"]
            print(
                f"start x2     : concurrent sessions ok "
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

            # Concurrent reads from both processes.
            ha, hb = await asyncio.gather(
                client_a.call_tool("recall", {"query": "where is the rate limiter", "limit": 5}),
                client_b.call_tool("recall", {"query": "where is the rate limiter", "limit": 5}),
            )
            ja, jb = _json(ha), _json(hb)
            assert ja["ok"] and jb["ok"]
            assert ja["count"] >= 1 and jb["count"] >= 1
            top = ja["memories"][0]
            print(
                f"recall x2    : both sessions see hits "
                f"(A={ja['count']}, B={jb['count']}) top={top['kind']} "
                f"score={top['score']:.3f}"
            )

            # Error paths travel over the wire as structured results.
            bad = _json(await client_a.call_tool("recall", {"query": ""}))
            assert bad["ok"] is False and "query is required" in bad["error"]
            print(f"error path   : {bad['error']}")

        # -- A third process reconnects later: everything persisted ----------
        async with Client(_config()) as client_c:
            started = _json(
                await client_c.call_tool("start", {"agent_id": "mcp-demo"})
            )
            pack = started["context_pack"]
            assert started["ok"]
            assert started["memory_count"] == 2  # exactly our two remembers
            assert pack["coverage"] == 1.0
            stored_paths = {p for a in pack["anchors"] for p in a.get("paths", [])}
            assert {"src/limits.ts", "src/session.ts"} <= stored_paths
            print(
                f"reconnect    : {started['memory_count']} memories persisted, "
                f"coverage {pack['coverage']:.0%}, paths {sorted(stored_paths)}"
            )

        print("\nMCP protocol round-trip OK (stdio, concurrent sessions, WAL)")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
