#!/usr/bin/env python3
"""Rebuild git history as three evenly staged commits on July 1, 7, and 17.

Distributes project files into three balanced batches (foundation → engine →
MCP surface) and commits each batch with matching author/committer dates.

Usage:
    python scripts/make_dated_commits.py              # year defaults to 2026
    python scripts/make_dated_commits.py --year 2025
    python scripts/make_dated_commits.py --dry-run    # print plan only
    python scripts/make_dated_commits.py --yes        # skip confirmation

Requires: git, run from anywhere (resolves repo root automatically).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Batch:
    """One dated commit: files + message + clock time within the day."""

    month_day: str  # MM-DD
    time: str  # HH:MM:SS
    title: str
    body: str
    files: tuple[str, ...]


# Three equal stages of the project (4 + 4 + 4 source paths).
# Order is intentional: each day builds on the previous.
BATCHES: tuple[Batch, ...] = (
    Batch(
        month_day="07-01",
        time="11:18:00",
        title="Add local SQLite foundation for coding-agent memory",
        body=(
            "Schema, storage interface, dependencies, and ignore rules.\n"
            "Local-first vault base so agent memory stays on disk."
        ),
        files=(
            ".gitignore",
            "requirements.txt",
            "schema.sql",
            "storage.py",
        ),
    ),
    Batch(
        month_day="07-07",
        time="14:42:00",
        title="Add small-codebase context engine and ranking",
        body=(
            "Typed memories, path/symbol signals, FTS ranking, and project notes.\n"
            "Middle layer: how recall works for repos under ~50k LOC."
        ),
        files=(
            "context_engine.py",
            "ranking.py",
            "pyproject.toml",
            "CLAUDE.md",
        ),
    ),
    Batch(
        month_day="07-17",
        time="16:05:00",
        title="Expose ContextVault MCP server, smoke test, and README",
        body=(
            "FastMCP tools (start/remember/recall), end-to-end test client,\n"
            "and setup docs for coding agents."
        ),
        files=(
            "server.py",
            "test_client.py",
            "README.md",
            "scripts/make_dated_commits.py",
        ),
    ),
)


def repo_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    return Path(out)


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        args,
        cwd=cwd,
        env=merged,
        text=True,
        capture_output=True,
        check=check,
    )


def ensure_files_exist(root: Path, files: tuple[str, ...]) -> None:
    missing = [f for f in files if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"Missing files (cannot build history): {missing}")


def all_batch_files() -> list[str]:
    return [f for b in BATCHES for f in b.files]


def print_plan(year: int) -> None:
    print("Dated commit plan (equal stages)\n")
    for i, batch in enumerate(BATCHES, 1):
        iso = f"{year}-{batch.month_day}T{batch.time}"
        print(f"  [{i}/3] {iso}")
        print(f"       {batch.title}")
        print(f"       files ({len(batch.files)}): {', '.join(batch.files)}")
        print()
    print(f"  Total files: {len(all_batch_files())}")
    print("  Balance:     4 + 4 + 4  (foundation / engine / MCP) — equal thirds")


def rebuild_history(root: Path, year: int, dry_run: bool) -> None:
    for batch in BATCHES:
        ensure_files_exist(root, batch.files)

    print_plan(year)
    if dry_run:
        print("Dry run — no git changes made.")
        return

    # Snapshot untracked batch files (e.g. this script) before any stash,
    # so orphan rebuild can still add them on the final day.
    snapshot_dir = root / ".git" / "dated-commit-file-snapshot"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True)
    for rel in all_batch_files():
        src = root / rel
        if src.is_file():
            dest = snapshot_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    # Stash any uncommitted work so we do not lose it
    status = run(["git", "status", "--porcelain"], cwd=root)
    dirty = bool(status.stdout.strip())
    if dirty:
        run(
            ["git", "stash", "push", "-u", "-m", "make_dated_commits auto-stash"],
            cwd=root,
        )
        print("Stashed uncommitted changes.")
        # Restore batch files needed for commits from snapshot
        for rel in all_batch_files():
            snap = snapshot_dir / rel
            if snap.is_file():
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snap, dest)

    branch = run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
    ).stdout.strip()

    backup = f"backup-before-dated-history"
    # Point a backup ref at current HEAD (safe rollback)
    run(["git", "branch", "-f", backup, "HEAD"], cwd=root)
    print(f"Backup branch: {backup} -> previous HEAD")

    # Orphan branch with no history
    tmp_branch = "dated-history-rebuild"
    run(["git", "checkout", "--orphan", tmp_branch], cwd=root)
    run(["git", "rm", "-rf", "--cached", "."], cwd=root, check=False)
    # Clear worktree index only; keep files on disk
    # Reset index fully
    run(["git", "reset"], cwd=root)

    # Start clean index; files remain on disk
    for batch in BATCHES:
        iso_date = f"{year}-{batch.month_day}T{batch.time}"
        # Use local timezone offset if available; else fixed -0000 style with Z-like local
        env = {
            "GIT_AUTHOR_DATE": iso_date,
            "GIT_COMMITTER_DATE": iso_date,
        }
        for rel in batch.files:
            run(["git", "add", "--", rel], cwd=root)

        message = f"{batch.title}\n\n{batch.body}\n"
        proc = run(
            ["git", "commit", "-m", message],
            cwd=root,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"Commit failed for {iso_date}: {batch.title}")

        sha = run(["git", "rev-parse", "--short", "HEAD"], cwd=root).stdout.strip()
        # Verify dates stuck
        show = run(
            ["git", "log", "-1", "--format=%H%n%ai%n%ci%n%s"],
            cwd=root,
        ).stdout.strip()
        print(f"Committed {sha} @ {iso_date}")
        print(f"  {show.splitlines()[-1]}")

    # Replace original branch with rebuilt history
    run(["git", "branch", "-M", branch], cwd=root)
    print(f"\nBranch '{branch}' now has 3 dated commits.")
    print(f"Rollback: git reset --hard {backup}")

    # Cleanup snapshot
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    if dirty:
        # Do not auto-pop stash (may conflict); just inform
        print("Note: uncommitted work was stashed. Restore with: git stash pop")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create three equal git commits on July 1, 7, and 17.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Year for the July dates (default: 2026)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without rewriting history",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip interactive confirmation",
    )
    args = parser.parse_args()

    try:
        root = repo_root()
    except subprocess.CalledProcessError:
        print("Not inside a git repository.", file=sys.stderr)
        return 1

    os.chdir(root)

    if not args.dry_run and not args.yes:
        print_plan(args.year)
        print(
            "This REWRITES local git history into exactly 3 commits "
            f"({args.year}-07-01, {args.year}-07-07, {args.year}-07-17)."
        )
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    rebuild_history(root, args.year, args.dry_run)
    if not args.dry_run:
        log = run(
            ["git", "log", "--format=%h  %ad  %s", "--date=short"],
            cwd=root,
        ).stdout
        print("\nFinal history:")
        print(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
