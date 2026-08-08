#!/usr/bin/env python3
"""Portfolio demo — colorful start / remember / recall for screenshots.

Usage:
    python demo_portfolio.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

# ANSI (works in Windows Terminal / modern PowerShell)
C = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
    "white": "\033[97m",
    "gray": "\033[90m",
    "bg": "\033[48;5;236m",
}


def c(color: str, text: str) -> str:
    return f"{C.get(color, '')}{text}{C['reset']}"


def banner() -> None:
    print()
    print(c("cyan", "  ╔══════════════════════════════════════════════════════════╗"))
    print(c("cyan", "  ║") + c("bold", "  ContextVault") + c("dim", "  ·  local MCP memory for coding agents  ") + c("cyan", "║"))
    print(c("cyan", "  ║") + c("gray", "  SQLite  ·  FTS5 context engine  ·  no cloud  ·  3 tools  ") + c("cyan", "║"))
    print(c("cyan", "  ╚══════════════════════════════════════════════════════════╝"))
    print()


def step(n: int, title: str) -> None:
    print(c("magenta", f"  [{n}] ") + c("bold", title))


def ok(msg: str) -> None:
    print(c("green", "      + ") + msg)


def dim(msg: str) -> None:
    print(c("gray", f"        {msg}"))


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="contextvault-demo-",
        ignore_cleanup_errors=True,
    ) as tmp:
        db_path = Path(tmp) / "portfolio.db"
        os.environ["CONTEXTVAULT_DB"] = str(db_path)

        import server

        server._store = None
        server._active_agent_id = None
        server._active_session_id = None

        banner()

        # --- START ---
        step(1, "start(agent_id='portfolio-demo')")
        started = server.start(agent_id="portfolio-demo")
        assert started.get("ok"), started
        ok(f"session   {c('yellow', started['session_id'][:13] + '…')}")
        ok(f"storage   {c('green', started['storage'])}")
        ok(f"summary   {started['summary'][:72]}…")
        print()

        # --- REMEMBER ---
        samples = [
            (
                "file_map",
                5,
                ["src/middleware/auth.ts"],
                "Auth: JWT access (15m) + httpOnly refresh. Middleware in src/middleware/auth.ts.",
            ),
            (
                "decision",
                5,
                ["storage.py"],
                "Decision: local SQLite only — agent memory never leaves the machine.",
            ),
            (
                "architecture",
                4,
                ["ranking.py", "context_engine.py"],
                "Recall = FTS5 + path/symbol/tag + intent + importance (no embeddings).",
            ),
            (
                "convention",
                3,
                [],
                "Conventions: type hints everywhere; MCP surface = start / remember / recall.",
            ),
            (
                "gotcha",
                4,
                [],
                "Gotcha (Windows): close SQLite before deleting temp DB dirs (WinError 32).",
            ),
            (
                "bug",
                3,
                ["ranking.py"],
                "Bugfix: FTS single-hit normalization no longer zeros relevance.",
            ),
        ]

        step(2, "remember()  —  store structured coding memories")
        for kind, imp, paths, content in samples:
            r = server.remember(
                content=content,
                kind=kind,
                paths=paths or None,
                importance=imp,
            )
            assert r.get("ok"), r
            m = r["memory"]
            path_s = m["paths"][0] if m["paths"] else "—"
            print(
                c("green", "      + ")
                + c("blue", f"#{m['id']:<2}")
                + c("yellow", f" {kind:<12}")
                + c("dim", f" imp={imp}  ")
                + c("gray", path_s)
            )
        print()

        # --- RECALL ---
        queries = [
            "Where is auth middleware?",
            "why local sqlite not cloud",
            "FTS5 ranking context engine",
            "windows sqlite cleanup gotcha",
        ]

        step(3, "recall()  —  multi-signal context engine")
        for q in queries:
            result = server.recall(query=q, limit=2)
            assert result.get("ok"), result
            print(c("cyan", f"\n      ? {q}"))
            for hit in result["memories"]:
                why = ", ".join(hit.get("why") or [])[:56]
                print(
                    c("green", "        -> ")
                    + c("yellow", f"{hit['kind']:<12}")
                    + c("white", f" score={hit['score']:.3f}  ")
                    + c("gray", why)
                )
                snippet = hit["content"][:64].replace("\n", " ")
                dim(snippet + ("…" if len(hit["content"]) > 64 else ""))
        print()

        # --- RESUME / CONTEXT PACK ---
        step(4, "start() again  —  token-budgeted working context")
        server._active_agent_id = None
        server._active_session_id = None
        resumed = server.start(agent_id="portfolio-demo")
        assert resumed.get("ok"), resumed
        pack = resumed["context_pack"]
        ok(f"memories  {c('white', str(resumed['memory_count']))}")
        cov = f"{pack.get('coverage', 0):.0%}"
        ok(
            f"tokens    {c('white', str(pack.get('tokens_used', 0)))}"
            f"/{c('white', str(pack.get('token_budget', 0)))}"
            f"  ·  coverage {c('white', cov)}"
        )
        ok(
            f"tiers     {c('white', str(len(pack.get('verbatim', []))))} verbatim"
            f" + {c('white', str(len(pack.get('compressed', []))))} compressed"
        )
        ok(f"by_kind   {c('gray', str(pack.get('by_kind', {})))}")
        print()
        print(c("dim", "      top anchors:"))
        for a in pack.get("anchors", [])[:4]:
            kind = a.get("kind", "?")
            content = (a.get("content") or "")[:58].replace("\n", " ")
            print(
                c("green", "        * ")
                + c("yellow", f"{kind:<12}")
                + c("gray", content + "…")
            )
        print()

        if server._store is not None:
            server._store.close()
            server._store = None

        print(c("cyan", "  ────────────────────────────────────────────────────────────"))
        print(
            c("green", "  OK")
            + c("bold", "  start → remember → recall → context pack")
            + c("dim", "  ·  all local")
        )
        print(c("cyan", "  ────────────────────────────────────────────────────────────"))
        print()
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"\nDEMO FAILED: {exc}", file=sys.stderr)
        raise
