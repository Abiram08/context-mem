#!/usr/bin/env python3
"""Prove that agent context is stored LOCALLY, on disk, in SQLite.

Spawns server.py over stdio using the EXACT config from this repo's .mcp.json
(the same file Claude Code reads), round-trips start -> remember -> recall,
then opens the SQLite vault directly to show the on-disk rows and audit trail.
A fresh process reconnects to confirm the context pack rebuilds from disk.

Uses a temporary vault — never touches the real contextvault.db.

Usage:
    python verify_local_memory.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _text(result: object) -> str:
    parts: list[str] = []
    for c in getattr(result, "content", None) or []:
        if hasattr(c, "text"):
            parts.append(str(c.text))
    return "\n".join(parts)


def _json(result: object) -> dict:
    return json.loads(_text(result))


async def _round_trip(client: object) -> None:
    """start -> remember x2 -> recall over the real protocol."""
    started = _json(await client.call_tool("start", {"agent_id": "claude-demo"}))
    assert started["ok"]
    await client.call_tool(
        "remember",
        {
            "content": "Auth middleware lives in src/middleware/auth.ts (JWT, 15m).",
            "kind": "file_map",
            "paths": ["src/middleware/auth.ts"],
            "importance": 5,
        },
    )
    await client.call_tool(
        "remember",
        {
            "content": "Decision: keep the vault local-only (SQLite), never a cloud.",
            "kind": "decision",
            "paths": ["storage.py"],
            "importance": 5,
        },
    )
    hits = _json(
        await client.call_tool("recall", {"query": "where is auth middleware", "limit": 3})
    )
    assert hits["ok"] and hits["count"] >= 1


async def main() -> int:
    mcp_cfg = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
    server_cfg = mcp_cfg["mcpServers"]["contextvault"]
    print("Using Claude Code config from:", ROOT / ".mcp.json")
    print(f"  command : {server_cfg['command']}")
    print(f"  args    : {server_cfg['args']}")

    with tempfile.TemporaryDirectory(
        prefix="contextvault-verify-",
        ignore_cleanup_errors=True,
    ) as tmp:
        vault = Path(tmp) / "claude-code-vault.db"
        cfg = {
            **server_cfg,
            "env": {
                **server_cfg.get("env", {}),
                "CONTEXTVAULT_DB": str(vault),
            },
        }
        client_cfg = {"mcpServers": {"contextvault": cfg}}

        from fastmcp import Client

        # --- Session 1 (fresh server process) -------------------------------
        print("\n[1] Session 1 — real MCP protocol, fresh process")
        async with Client(client_cfg) as client:
            await _round_trip(client)
            print("    start -> remember x2 -> recall: ok (over stdio)")

        # --- Show what actually landed on disk ------------------------------
        print("\n[2] The vault file on disk")
        print(f"    location : {vault}")
        assert vault.exists(), "vault DB was not created on disk!"
        conn = sqlite3.connect(vault)
        conn.row_factory = sqlite3.Row
        print("    tables   :", [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )])
        print("    memories :")
        for r in conn.execute(
            "SELECT id, kind, importance, substr(content,1,58) AS c FROM memories"
        ):
            print(f"      #{r['id']} {r['kind']:<12} imp={r['importance']}  {r['c']}")
        print("    events   :")
        for r in conn.execute(
            "SELECT action, substr(coalesce(content_after, content_before, ''),1,48) AS c FROM memory_events"
        ):
            print(f"      {r['action']:<8} {r['c']}")
        conn.close()

        # --- Session 2 (fresh process): context rebuilds FROM DISK -----------
        print("\n[3] Session 2 — new process reconnects, reads the same vault")
        async with Client(client_cfg) as client:
            started = _json(
                await client.call_tool("start", {"agent_id": "claude-demo"})
            )
            pack = started["context_pack"]
            assert started["ok"]
            assert started["memory_count"] == 2
            assert pack["coverage"] == 1.0
            assert len(pack["anchors"]) == 2
            print(
                f"    memory_count={started['memory_count']} "
                f"coverage={pack['coverage']:.0%} — pack rebuilt from disk"
            )

        print("\n" + "=" * 60)
        print("LOCAL MEMORY VERIFIED: agent context is stored on this machine,")
        print("in a single SQLite file, and survives process restarts.")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
