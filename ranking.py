"""Context-engine-backed recall for Context-mem.

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
from embeddings import cosine
from storage import Memory, _row_to_memory

# Default weight of the optional semantic signal when blending with lexical.
SEMANTIC_WEIGHT = 0.30
# In-memory vector cache: memory_id -> (content_hash, vector). Keeps repeated
# recalls cheap without touching the schema.
_vector_cache: dict[int, tuple[int, list[float]]] = {}


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
    tags: list[str] | None = None,
    embedder: Any | None = None,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> list[RankedMemory]:
    """Return top coding memories for *query*, scoped to *agent_id*.

    Optimized for small codebases: path/symbol precision and kind diversity
    matter more than pure BM25 volume. Optional hard filters: kind, path
    substring, and tags (ALL requested tags must be present, mem0-style).

    When *embedder* is provided (opt-in API/local embeddings), a semantic
    re-rank is blended with the lexical score; any embedder failure degrades
    gracefully back to lexical-only.
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
    # Tag filter: guaranteed tag-matching candidates even with zero FTS overlap
    for raw, mem in _tag_candidates(conn, agent_id, tags, pool, kind_filter):
        if mem.id not in by_id or raw > by_id[mem.id][0]:
            by_id[mem.id] = (raw, mem)

    ranked_pairs = list(by_id.values())
    if tags:
        # Hard filter: ONLY memories carrying every requested tag survive.
        wanted = {t.lower() for t in tags}
        ranked_pairs = [
            (raw, m)
            for raw, m in ranked_pairs
            if wanted <= {t.lower() for t in m.tags}
        ]
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
            created_at=mem.created_at,
            last_accessed_at=mem.last_accessed_at,
        )
        scored.append((mem, bd))

    scored.sort(key=lambda x: x[1].total, reverse=True)

    # Optional semantic re-rank (opt-in; degrades gracefully to lexical).
    if embedder is not None and scored:
        try:
            q_vec = embedder.embed_texts([query])[0]
            for mem, bd in scored:
                vec = _vector_for(embedder, mem)
                if vec is None:
                    continue
                sim = cosine(q_vec, vec)
                bd.semantic = round(sim, 4)
                bd.total = round(
                    (1.0 - semantic_weight) * bd.total + semantic_weight * sim, 4
                )
                if sim >= 0.55:
                    bd.reasons.append(f"semantic {sim:.2f}")
            scored.sort(key=lambda x: x[1].total, reverse=True)
        except Exception:
            pass

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


def _vector_for(embedder: Any, mem: Memory) -> list[float] | None:
    """Embed one memory with an in-memory cache keyed by id + content hash."""
    h = hash(mem.content)
    entry = _vector_cache.get(mem.id)
    if entry is not None and entry[0] == h:
        return entry[1]
    try:
        vec = embedder.embed_texts([mem.content])[0]
    except Exception:
        return None
    _vector_cache[mem.id] = (h, vec)
    return vec


def _tag_candidates(
    conn: sqlite3.Connection,
    agent_id: str,
    tags: list[str] | None,
    pool: int,
    kind_filter: str | None,
) -> list[tuple[float, Memory]]:
    """Memories carrying ALL requested tags — a hard filter, not a boost."""
    if not tags:
        return []
    wanted = {t.lower() for t in tags}
    sql = "SELECT * FROM memories WHERE agent_id = ?"
    params: list[Any] = [agent_id]
    if kind_filter:
        sql += " AND kind = ?"
        params.append(kind_filter)
    sql += (
        " ORDER BY importance DESC, access_count DESC, created_at DESC, id DESC"
        " LIMIT ?"
    )
    params.append(max(pool, 100))

    out: list[tuple[float, Memory]] = []
    for row in conn.execute(sql, params).fetchall():
        mem = _row_to_memory(row)
        have = {t.lower() for t in mem.tags}
        if wanted <= have:
            out.append((0.5, mem))
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
