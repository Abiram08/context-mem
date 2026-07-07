"""Context-engine-backed recall for ContextVault.

Multi-signal ranking for small codebases (< ~50k LOC):
  FTS5 (porter) + path/symbol/tag hits + kind/intent + importance + usage.
Fully local — no embedding API required.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from context_engine import (
    ScoreBreakdown,
    build_fts_match,
    detect_intent,
    diversify,
    extract_paths,
    normalize_kind,
    score_memory,
    tokenize_query,
)
from storage import Memory, _row_to_memory


@dataclass(frozen=True)
class RankedMemory:
    """A memory with composite score + explainable breakdown."""

    memory: Memory
    score: float
    relevance: float
    importance: int
    breakdown: ScoreBreakdown

    def to_dict(self) -> dict[str, Any]:
        d = self.memory.to_dict()
        d["score"] = round(self.score, 4)
        d["relevance"] = round(self.relevance, 4)
        d["why"] = self.breakdown.reasons
        d["signals"] = self.breakdown.to_dict()
        return d


def recall(
    conn: sqlite3.Connection,
    agent_id: str,
    query: str,
    *,
    limit: int = 5,
    kind: str | None = None,
    path: str | None = None,
) -> list[RankedMemory]:
    """Return top coding memories for *query*, scoped to *agent_id*.

    Optimized for small codebases: path/symbol precision and kind diversity
    matter more than pure BM25 volume.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    agent_id = agent_id.strip()
    query = query.strip()
    tokens = tokenize_query(query)
    intent = detect_intent(query)
    query_paths = extract_paths(query)
    if path:
        query_paths = list(dict.fromkeys([*query_paths, path.replace("\\", "/")]))

    kind_filter = normalize_kind(kind) if kind else None
    match_expr = build_fts_match(tokens)
    pool = max(limit * 6, 30)

    candidates = _fts_candidates(conn, agent_id, match_expr, pool, kind_filter)
    # Merge fallback so path-only rows still surface
    fallback = _fallback_candidates(
        conn, agent_id, tokens, query_paths, pool, kind_filter
    )
    by_id: dict[int, tuple[float, Memory]] = {}
    for raw, mem in candidates:
        by_id[mem.id] = (raw, mem)
    for raw, mem in fallback:
        if mem.id not in by_id:
            by_id[mem.id] = (raw, mem)
        else:
            # Keep higher raw signal
            if raw > by_id[mem.id][0]:
                by_id[mem.id] = (raw, mem)

    ranked_pairs = list(by_id.values())
    if not ranked_pairs:
        return []

    # Optional hard path filter (substring on stored paths / content)
    if path:
        pl = path.lower().replace("\\", "/")
        ranked_pairs = [
            (raw, m)
            for raw, m in ranked_pairs
            if any(pl in p.lower() for p in m.paths)
            or pl in m.content.lower()
        ]
        if not ranked_pairs:
            return []

    max_rel = max(r for r, _ in ranked_pairs)
    min_rel = min(r for r, _ in ranked_pairs)
    span = max_rel - min_rel

    scored: list[tuple[Memory, ScoreBreakdown]] = []
    for raw_rel, mem in ranked_pairs:
        if span > 1e-12:
            fts_norm = (raw_rel - min_rel) / span
        else:
            fts_norm = 1.0 if raw_rel > 0 else 0.0

        bd = score_memory(
            content=mem.content,
            kind=mem.kind,
            tags=mem.tags,
            paths=mem.paths,
            symbols=mem.symbols,
            importance=mem.importance,
            access_count=mem.access_count,
            fts_raw=raw_rel,
            fts_norm=fts_norm,
            query=query,
            tokens=tokens,
            intent=intent,
            query_paths=query_paths,
        )
        scored.append((mem, bd))

    scored.sort(key=lambda x: x[1].total, reverse=True)
    diversified = diversify(scored, limit=limit, max_per_kind=max(2, limit // 2))

    results: list[RankedMemory] = []
    for mem, bd in diversified:
        results.append(
            RankedMemory(
                memory=mem,
                score=bd.total,
                relevance=bd.fts / 0.42 if bd.fts else 0.0,  # back to ~[0,1]
                importance=mem.importance,
                breakdown=bd,
            )
        )
    return results


def _fts_candidates(
    conn: sqlite3.Connection,
    agent_id: str,
    match_expr: str | None,
    pool: int,
    kind_filter: str | None,
) -> list[tuple[float, Memory]]:
    if not match_expr:
        return []

    sql = """
        SELECT m.*,
               bm25(memories_fts) AS bm25_rank
        FROM memories_fts
        JOIN memories m ON m.id = memories_fts.rowid
        WHERE memories_fts MATCH ?
          AND m.agent_id = ?
    """
    params: list[Any] = [match_expr, agent_id]
    if kind_filter:
        sql += " AND m.kind = ?"
        params.append(kind_filter)
    sql += " ORDER BY bm25_rank LIMIT ?"
    params.append(pool)

    try:
        cur = conn.execute(sql, params)
    except sqlite3.OperationalError:
        return []

    out: list[tuple[float, Memory]] = []
    for row in cur.fetchall():
        bm25_val = float(row["bm25_rank"])
        raw_rel = -bm25_val if bm25_val < 0 else (1.0 / (1.0 + bm25_val))
        out.append((raw_rel, _row_to_memory(row)))
    return out


def _fallback_candidates(
    conn: sqlite3.Connection,
    agent_id: str,
    tokens: list[str],
    query_paths: list[str],
    pool: int,
    kind_filter: str | None,
) -> list[tuple[float, Memory]]:
    sql = """
        SELECT * FROM memories
        WHERE agent_id = ?
    """
    params: list[Any] = [agent_id]
    if kind_filter:
        sql += " AND kind = ?"
        params.append(kind_filter)
    sql += " ORDER BY importance DESC, access_count DESC, created_at DESC, id DESC LIMIT ?"
    params.append(max(pool, 80))

    rows = conn.execute(sql, params).fetchall()
    scored: list[tuple[float, Memory]] = []
    path_needles = [p.lower().replace("\\", "/") for p in query_paths]

    for row in rows:
        mem = _row_to_memory(row)
        score = 0.0
        hay = " ".join(
            [
                mem.content.lower(),
                " ".join(t.lower() for t in mem.tags),
                " ".join(p.lower() for p in mem.paths),
                " ".join(s.lower() for s in mem.symbols),
                mem.kind,
            ]
        )
        if tokens:
            hits = sum(1 for t in tokens if t in hay)
            score = hits / len(tokens)
        for pn in path_needles:
            if any(pn in p.lower() for p in mem.paths) or pn in mem.content.lower():
                score = max(score, 0.85)
        if score <= 0 and mem.importance < 4 and not path_needles:
            continue
        scored.append((score, mem))

    if not scored and rows:
        for row in rows[:pool]:
            scored.append((0.0, _row_to_memory(row)))

    scored.sort(key=lambda x: (x[0], x[1].importance), reverse=True)
    return scored[:pool]
