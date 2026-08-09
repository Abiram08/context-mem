"""Token-aware compression for small-codebase context packs.

The working-context problem: a coding agent should never have to re-learn the
vault, but it also shouldn't pay full-token price for every memory. This module
turns raw memories into a compact, token-budgeted representation **locally** —
no LLM, no embeddings, no network.

Pieces:
- ``estimate_tokens``  — deterministic offline token-budget currency.
- ``split_segments``   — sentence splitting that protects paths & abbreviations.
- ``extractive_compress`` — keep the highest-signal segments under a budget.
- ``collapse_redundant`` — merge near-duplicate memories so repeats cost ~0.

Everything here is deterministic and pure, so packs are reproducible and
testable without any external service.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any

# ---------------------------------------------------------------------------
# Budgets & knobs (tune per-project via server env / start(token_budget=...))
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET = 1200  # max tokens for a session working-context pack
MAX_VERBATIM = 4  # how many hot memories stay verbatim (highest signal)
VERBATIM_RATIO = 0.30  # share of budget reserved for verbatim tier
INDEX_RESERVE = 96  # tokens reserved for the coverage index lines
MIN_PER_MEMORY = 24  # floor budget for each compressed memory
MAX_MERGED_TOKENS = 420  # merged duplicates must stay under this
JACCARD_THRESHOLD = 0.45  # content-token overlap to consider a duplicate
PATH_JACCARD_THRESHOLD = 0.28  # lower bar when memories share a file path
MERGE_KINDS = frozenset({"file_map", "api", "context"})  # kinds that merge freely

# ---------------------------------------------------------------------------
# Token estimation (offline, deterministic)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")

_tiktoken_checked = False
_tiktoken_enc: Any | None = None


def _use_tiktoken() -> bool:
    """Opt-in real tokenizer: CONTEXTVAULT_TOKENIZER=tiktoken (+pip install
    tiktoken). Default stays fully offline, deterministic, dependency-free.
    """
    global _tiktoken_checked, _tiktoken_enc
    if not _tiktoken_checked:
        _tiktoken_checked = True
        if os.environ.get("CONTEXTVAULT_TOKENIZER", "").lower() == "tiktoken":
            try:
                import tiktoken  # type: ignore[import-not-found]

                _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            except Exception:  # noqa: BLE001 - missing package / vocab
                _tiktoken_enc = None
    return _tiktoken_enc is not None


def estimate_tokens(text: str) -> int:
    """Estimate tokens for mixed prose+code text, fully offline by default.

    With CONTEXTVAULT_TOKENIZER=tiktoken (and tiktoken installed), counts
    with the real cl100k_base tokenizer; otherwise a deterministic heuristic:
    ~1.35 tokens per prose word, ~1.0 per code-ish token (paths/identifiers
    with separators pack denser), with long tokens charging extra. Exactness
    isn't the goal — a stable, monotonic budget currency is.
    """
    if not text:
        return 0
    if _use_tiktoken():
        try:
            return max(1, len(_tiktoken_enc.encode(text)))
        except Exception:  # noqa: BLE001 - fall back to heuristic
            pass
    n_prose = 0
    n_codey = 0
    for w in _WORD_RE.findall(text):
        if len(w) > 3 and any(c in w for c in "/\\._-"):
            n_codey += 1 + max(0, (len(w) - 8)) // 6
        else:
            n_prose += 1 + max(0, (len(w) - 8)) // 6
    return max(1, int(round(n_prose * 1.35 + n_codey * 1.0)))


# ---------------------------------------------------------------------------
# Segment splitting
# ---------------------------------------------------------------------------

_ABBR_RE = re.compile(
    r"\b(?:e\.g\.|i\.e\.|vs\.|etc\.|dr\.|mr\.|mrs\.|ms\.|st\.|approx\.|no\.)\b",
    re.I,
)
_ABBR_PLACEHOLDER = "\x00abbr\x00"
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[{])")
# continuation fragments that shouldn't start a new segment
_FRAGMENT_RE = re.compile(r"^(?:and|but|or|then|which|that|it|this|while|with|to|for)\b", re.I)


def split_segments(text: str) -> list[str]:
    """Split content into sentence-sized segments.

    Protects file paths (``auth.ts`` stays whole), version numbers, and common
    abbreviations (``e.g.``, ``i.e.``). Used by extractive compression so a
    budget can keep the *informative* sentences and drop the filler.
    """
    text = (text or "").strip()
    if not text:
        return []
    guarded = _ABBR_RE.sub(_ABBR_PLACEHOLDER, text)
    parts = _SENT_SPLIT_RE.split(guarded)
    out: list[str] = []
    for p in parts:
        p = p.strip().replace(_ABBR_PLACEHOLDER, "e.g.")
        if not p:
            continue
        # A short, lowercase continuation belongs to the previous segment.
        if out and len(p) < 32 and _FRAGMENT_RE.match(p) and ":" not in p:
            out[-1] += " " + p
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Extractive compression
# ---------------------------------------------------------------------------

_SIGNAL_RE = re.compile(
    r"\b(decided|decision|because|must|never|always|root cause|broken|fixed|"
    r"works|lives|located|is|are|was|were|returns|throws|requires|blocks|"
    r"moved|renamed|replaced|migrat|key|main|entrypoint)\b",
    re.I,
)


def _score_segment(seg: str, memory: Any | None) -> float:
    """Score how much *this sentence* matters to keep in a compact pack."""
    score = 0.0
    if memory is None:
        return 0.0
    low = seg.lower()
    for p in (getattr(memory, "paths", None) or []):
        if str(p).lower() in low:
            score += 2.0
    for s in (getattr(memory, "symbols", None) or []):
        if str(s).lower() in low:
            score += 1.5
    for t in (getattr(memory, "tags", None) or []):
        if str(t).lower() in low:
            score += 1.0
    if _SIGNAL_RE.search(low):
        score += 0.7
    score += (int(getattr(memory, "importance", 3)) / 5.0) * 0.5
    return score


def _head_slice(text: str, budget_tokens: int) -> tuple[str, int]:
    """Degenerate case: even the best segment doesn't fit — keep its head."""
    words = text.split()
    cur = ""
    used = 0
    for w in words:
        t = estimate_tokens(w)
        if cur and used + t > budget_tokens:
            break
        if not cur and used + t > budget_tokens:
            cur = w[: max(8, budget_tokens)]
            break
        cur = cur + (" " if cur else "") + w
        used += t
    return cur, used


def extractive_compress(
    content: str,
    budget_tokens: int,
    *,
    memory: Any | None = None,
) -> tuple[str, int, int]:
    """Keep the highest-signal segments of *content* under a token budget.

    Returns ``(text, dropped_segments, tokens_used)``. The dropped count is
    surfaced so agents know a note was distilled, not lost.
    """
    if budget_tokens < 1:
        return "", len(split_segments(content)), 0
    segments = split_segments(content)
    if not segments:
        return "", 0, 0

    scored = [
        (i, _score_segment(seg, memory), seg) for i, seg in enumerate(segments)
    ]
    kept: list[tuple[int, str]] = []
    used = 0
    for i, _score, seg in sorted(scored, key=lambda x: x[1], reverse=True):
        t = estimate_tokens(seg)
        if kept and used + t > budget_tokens:
            continue  # this segment doesn't fit; try the next-best
        if not kept and used + t > budget_tokens:
            head, head_t = _head_slice(seg, budget_tokens)
            kept.append((i, head))
            used = head_t
            break
        kept.append((i, seg))
        used += t

    kept.sort(key=lambda x: x[0])  # restore original order for readability
    text = " ".join(s for _, s in kept)
    dropped = len(segments) - len(kept)
    if dropped > 0:
        text = f"{text} (+{dropped} more)"
        used += estimate_tokens(f" (+{dropped} more)")
    return text, dropped, used


# ---------------------------------------------------------------------------
# Redundancy collapse (pack-time only — originals stay in the vault)
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-./\\]*")


def _content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for t in _TOK_RE.findall(text.lower().replace("\\", "/")):
        if len(t) >= 3:
            out.add(t)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    denom = len(a | b)
    return (len(a & b) / denom) if denom else 0.0


def _mergeable(a: Any, b: Any, ta: set[str], tb: set[str]) -> bool:
    ka = getattr(a, "kind", "context")
    kb = getattr(b, "kind", "context")
    if ka != kb and {ka, kb} - MERGE_KINDS:
        return False
    jac = _jaccard(ta, tb)
    if jac >= JACCARD_THRESHOLD:
        return True
    pa = {p.lower().replace("\\", "/") for p in (getattr(a, "paths", None) or [])}
    pb = {p.lower().replace("\\", "/") for p in (getattr(b, "paths", None) or [])}
    if pa & pb and jac >= PATH_JACCARD_THRESHOLD:
        return True
    return False


def _merge_content(a: Any, b: Any) -> str:
    ca = (getattr(a, "content", "") or "").strip()
    cb = (getattr(b, "content", "") or "").strip()
    if not ca:
        return cb
    if not cb:
        return ca
    if ca.lower() in cb.lower():
        return cb
    if cb.lower() in ca.lower():
        return ca
    segs = split_segments(ca) + split_segments(cb)
    out: list[str] = []
    seen: set[str] = set()
    for s in segs:
        k = re.sub(r"\W+", " ", s.lower()).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    joined = " ".join(out)
    if estimate_tokens(joined) > MAX_MERGED_TOKENS:
        # Keep the higher-value original rather than a giant blob.
        a_val = int(getattr(a, "importance", 3)), len(ca)
        b_val = int(getattr(b, "importance", 3)), len(cb)
        return ca if a_val >= b_val else cb
    return joined


def _union(x: Any, y: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in [*(x or []), *(y or [])]:
        s = str(item)
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _merged(m: Any, **overrides: Any) -> Any:
    if isinstance(m, dict):
        d = dict(m)
        d.update(overrides)
        return d
    try:
        return replace(m, **overrides)
    except TypeError:  # not a dataclass — fall back to a plain namespace
        ns = type("MergedMemory", (), {})()
        for name in (
            "id", "agent_id", "session_id", "kind", "importance", "created_at",
            "last_accessed_at", "access_count",
        ):
            setattr(ns, name, getattr(m, name, None))
        for name in ("content", "paths", "symbols", "tags"):
            setattr(ns, name, overrides.get(name, getattr(m, name, [])))
        for name in overrides:
            if name not in {"content", "paths", "symbols", "tags"}:
                setattr(ns, name, overrides[name])
        return ns


def collapse_redundant(
    memories: list[Any],
    *,
    threshold: float = JACCARD_THRESHOLD,
) -> tuple[list[Any], int]:
    """Merge near-duplicate memories (pack-time only).

    Two notes merge when they're kind-compatible and either share enough
    content tokens or share a file path with loose overlap. Merging unions
    paths/symbols/tags, keeps the max importance, and sums access counts —
    so nothing is lost, just no longer double-charged.

    Returns ``(merged_list, removed_count)``. O(n^2) over the (small) vault —
    capped at 1500 memories for safety.
    """
    if len(memories) < 2:
        return list(memories), 0
    capped = memories[:1500]
    tok_sets = [_content_tokens(getattr(m, "content", "") or "") for m in capped]
    used = [False] * len(capped)
    out: list[Any] = []
    for i in range(len(capped)):
        if used[i]:
            continue
        rep = capped[i]
        for j in range(i + 1, len(capped)):
            if used[j]:
                continue
            if not _mergeable(rep, capped[j], tok_sets[i], tok_sets[j]):
                continue
            rep = _merged(
                rep,
                content=_merge_content(rep, capped[j]),
                paths=_union(getattr(rep, "paths", None), getattr(capped[j], "paths", None)),
                symbols=_union(
                    getattr(rep, "symbols", None), getattr(capped[j], "symbols", None)
                ),
                tags=_union(getattr(rep, "tags", None), getattr(capped[j], "tags", None)),
                importance=max(
                    int(getattr(rep, "importance", 3)),
                    int(getattr(capped[j], "importance", 3)),
                ),
                access_count=int(getattr(rep, "access_count", 0))
                + int(getattr(capped[j], "access_count", 0)),
            )
            used[j] = True
        out.append(rep)
    return out, len(memories) - len(out)
