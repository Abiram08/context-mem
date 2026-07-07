"""Small-codebase context engine for coding-agent memory.

Designed for repos under ~50k lines where a dense, structured memory vault
outperforms raw chat history and heavyweight vector DBs:

- Typed memories (decision, file_map, bug, convention, …)
- Path / symbol extraction and matching
- Intent-aware ranking signals
- Session context packs that fit an agent's working window

All local — no network, no embeddings required.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Memory kinds (coding-agent ontology)
# ---------------------------------------------------------------------------

MEMORY_KINDS: frozenset[str] = frozenset(
    {
        "decision",  # architectural / product choices and why
        "convention",  # style, process, commit, naming rules
        "bug",  # root causes, fixed failures, landmines
        "file_map",  # where things live in the tree
        "api",  # endpoints, public functions, contracts
        "preference",  # user/agent preferences
        "gotcha",  # non-obvious traps and footguns
        "architecture",  # system shape, layers, data flow
        "todo",  # open work, deferred items
        "context",  # general durable notes (default)
    }
)

# Intent hints inferred from natural-language recall queries
_INTENT_KIND_BOOSTS: dict[str, dict[str, float]] = {
    "locate": {
        "file_map": 0.28,
        "api": 0.18,
        "architecture": 0.12,
    },
    "why": {
        "decision": 0.28,
        "architecture": 0.16,
        "gotcha": 0.10,
    },
    "bug": {
        "bug": 0.30,
        "gotcha": 0.20,
        "decision": 0.08,
    },
    "style": {
        "convention": 0.30,
        "preference": 0.20,
    },
    "api": {
        "api": 0.28,
        "file_map": 0.12,
    },
    "structure": {
        "architecture": 0.28,
        "file_map": 0.16,
        "decision": 0.10,
    },
    "open_work": {
        "todo": 0.30,
        "bug": 0.12,
    },
    "general": {},
}

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "locate",
        re.compile(
            r"\b(where|which file|path to|located|lives in|find file|"
            r"look for|entrypoint|entry point)\b",
            re.I,
        ),
    ),
    (
        "why",
        re.compile(r"\b(why|reason|rationale|decided|decision|chose|instead of)\b", re.I),
    ),
    (
        "bug",
        re.compile(
            r"\b(bug|fix|broken|error|exception|crash|regression|root cause|"
            r"failing|issue)\b",
            re.I,
        ),
    ),
    (
        "style",
        re.compile(
            r"\b(convention|style|prefer|preference|format|lint|naming|"
            r"commit message|type hint)\b",
            re.I,
        ),
    ),
    (
        "api",
        re.compile(
            r"\b(api|endpoint|route|handler|signature|public function|"
            r"interface|contract)\b",
            re.I,
        ),
    ),
    (
        "structure",
        re.compile(
            r"\b(architecture|layer|stack|structure|system design|"
            r"data flow|module|package layout)\b",
            re.I,
        ),
    ),
    (
        "open_work",
        re.compile(r"\b(todo|to-do|leftover|unfinished|next step|backlog)\b", re.I),
    ),
]

# Kind inference from free text when the agent omits `kind`
_KIND_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("file_map", re.compile(r"\b(lives in|located in|file(s)?|path|src/|lib/|\.tsx?\b|\.py\b|\.go\b)", re.I)),
    ("bug", re.compile(r"\b(bug|fix|root cause|regression|crash|broke)\b", re.I)),
    ("decision", re.compile(r"\b(decided|decision|we chose|instead of|trade-?off)\b", re.I)),
    ("convention", re.compile(r"\b(convention|always|never|must|style|prefer)\b", re.I)),
    ("preference", re.compile(r"\b(user prefers|prefer(s|ence)?|don't|do not)\b", re.I)),
    ("gotcha", re.compile(r"\b(gotcha|pitfall|footgun|watch out|careful|non-obvious)\b", re.I)),
    ("architecture", re.compile(r"\b(architecture|layer|stack|pipeline|data flow)\b", re.I)),
    ("api", re.compile(r"\b(endpoint|route|API|handler|public)\b", re.I)),
    ("todo", re.compile(r"\b(TODO|FIXME|still need|follow-?up|later:)\b", re.I)),
]

# Paths like src/foo/bar.ts, server.py, .\lib\x.py, packages/a/b
_CODE_EXTS = (
    r"py|ts|tsx|js|jsx|go|rs|java|kt|cs|rb|php|c|cc|cpp|h|hpp|md|sql|"
    r"json|ya?ml|toml|ini|cfg|sh|bash|zsh|ps1|css|scss|html|vue|svelte|"
    r"swift|m|mm|r|scala|lua|dart|proto|graphql|env|lock"
)
_PATH_RE = re.compile(
    rf"(?![0-9]+(?:\.[0-9]+)+\b)"  # skip version numbers
    rf"(?:"
    # path with directory + file (with or without extension)
    rf"(?:(?:[A-Za-z]:)?(?:\.\.?[/\\])?)?"
    rf"(?:[\w.-]+[/\\])+[\w.-]+(?:\.({_CODE_EXTS}))?"
    # bare filename with a known code extension
    rf"|[\w.-]+\.({_CODE_EXTS})"
    rf")\b",
    re.I,
)
# Extensionless dir paths only if they start with common roots
_DIR_PATH_RE = re.compile(
    r"\b(?:src|lib|app|pkg|packages|internal|cmd|scripts|tests?|docs?|"
    r"components|hooks|utils|services|middleware|api|core|modules)"
    r"(?:/[\w.-]+){1,6}\b",
    re.I,
)

_SYMBOL_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)+)\b"  # PascalCase
    r"|\b([a-z_][a-z0-9_]{2,})\s*\("  # call-ish snake/camel
    r"|\b([a-z]+[A-Z][a-zA-Z0-9]+)\b"  # camelCase
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-./\\]*")

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "shall", "can", "this", "that", "these", "those", "it",
        "its", "we", "you", "they", "i", "me", "my", "our", "your", "with",
        "from", "by", "as", "into", "about", "what", "which", "who", "how",
        "when", "where", "why", "not", "no", "yes", "if", "then", "than", "so",
        "just", "also", "very", "any", "all", "some", "more", "most", "other",
        "only", "own", "same", "such", "too", "use", "using", "used", "please",
        "find", "get", "show", "tell", "know", "need", "want", "like", "look",
        "looking", "related", "regarding", "memory", "memories", "recall",
        "remember", "code", "codebase", "project", "file", "files", "thing",
        "things", "stuff", "there", "here", "does", "work", "works", "working",
    }
)

# Session pack priorities for small codebases (dense bootstrap context)
_PACK_KIND_ORDER: list[str] = [
    "architecture",
    "decision",
    "convention",
    "file_map",
    "preference",
    "gotcha",
    "bug",
    "api",
    "todo",
    "context",
]

# Soft cap for context packs — small repos don't need huge dumps
DEFAULT_PACK_LIMIT = 14
MAX_PER_KIND_IN_PACK = 3


# ---------------------------------------------------------------------------
# Extraction / normalization
# ---------------------------------------------------------------------------


def normalize_kind(kind: str | None) -> str:
    if not kind:
        return "context"
    k = kind.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "arch": "architecture",
        "architecture_decision": "decision",
        "adr": "decision",
        "pref": "preference",
        "prefs": "preference",
        "map": "file_map",
        "files": "file_map",
        "filepath": "file_map",
        "location": "file_map",
        "fix": "bug",
        "issue": "bug",
        "trap": "gotcha",
        "pitfall": "gotcha",
        "style": "convention",
        "rule": "convention",
        "endpoint": "api",
        "note": "context",
        "general": "context",
    }
    k = aliases.get(k, k)
    return k if k in MEMORY_KINDS else "context"


def infer_kind(content: str, tags: Iterable[str] | None = None) -> str:
    """Best-effort kind when the agent doesn't specify one."""
    tag_set = {t.lower() for t in (tags or [])}
    for kind in MEMORY_KINDS:
        if kind in tag_set or kind.replace("_", "") in tag_set:
            return kind
    for kind, pat in _KIND_HINTS:
        if pat.search(content):
            return kind
    return "context"


def extract_paths(text: str) -> list[str]:
    """Pull likely file/module paths from free text."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        raw = raw.replace("\\", "/").rstrip(".,;:)'\"")
        key = raw.lower()
        if key in seen or len(raw) < 3:
            return
        # Skip URLs / domains
        if raw.count(".") == 1 and "/" not in raw:
            ext = raw.split(".")[-1].lower()
            if ext in {"com", "org", "io", "net", "dev", "app"}:
                return
        # Skip tool-call prose like start/remember/recall (no code ext, no src root)
        if "/" in raw and "." not in raw.rsplit("/", 1)[-1]:
            first = raw.split("/", 1)[0].lower()
            if first not in {
                "src", "lib", "app", "pkg", "packages", "internal", "cmd",
                "scripts", "test", "tests", "docs", "doc", "components",
                "hooks", "utils", "services", "middleware", "api", "core",
                "modules", ".", "..",
            } and not first.endswith(":"):
                return
        seen.add(key)
        found.append(raw)

    for m in _PATH_RE.finditer(text):
        _add(m.group(0))
    for m in _DIR_PATH_RE.finditer(text):
        _add(m.group(0))
    return found


def extract_symbols(text: str) -> list[str]:
    """Pull likely code identifiers (PascalCase, camelCase, call names)."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _SYMBOL_RE.finditer(text):
        sym = next(g for g in m.groups() if g)
        if sym.lower() in _STOPWORDS or len(sym) < 3:
            continue
        key = sym.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(sym)
    return found[:24]


def build_search_doc(
    *,
    content: str,
    kind: str,
    tags: list[str],
    paths: list[str],
    symbols: list[str],
) -> str:
    """Denormalized document for FTS — optimized for small-repo keyword recall."""
    parts = [
        content,
        f"kind:{kind}",
        kind.replace("_", " "),
    ]
    if tags:
        parts.append("tags: " + " ".join(tags))
        parts.extend(tags)
    if paths:
        parts.append("paths: " + " ".join(paths))
        for p in paths:
            parts.append(p.replace("/", " ").replace("\\", " ").replace(".", " "))
            parts.append(p.split("/")[-1])
    if symbols:
        parts.append("symbols: " + " ".join(symbols))
        parts.extend(symbols)
    return "\n".join(parts)


def tokenize_query(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(query):
        tok = match.group(0).lower().replace("\\", "/")
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        candidates = [tok]
        if "/" in tok or "." in tok:
            # Also index path segments and basenames
            for seg in re.split(r"[/.]", tok):
                if seg and seg not in _STOPWORDS and len(seg) >= 2:
                    candidates.append(seg)
        if tok.endswith("ies") and len(tok) > 4:
            candidates.append(tok[:-3] + "y")
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            candidates.append(tok[:-1])
        for c in candidates:
            if c not in seen and c not in _STOPWORDS and len(c) >= 2:
                seen.add(c)
                tokens.append(c)
    return tokens


def detect_intent(query: str) -> str:
    for name, pat in _INTENT_PATTERNS:
        if pat.search(query):
            return name
    return "general"


def build_fts_match(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    parts: list[str] = []
    for t in tokens:
        safe = t.replace('"', '""')
        parts.append(f'"{safe}"*')
    return " OR ".join(parts)


# ---------------------------------------------------------------------------
# Scoring (multi-signal, small-codebase)
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    fts: float = 0.0
    path: float = 0.0
    symbol: float = 0.0
    tag: float = 0.0
    kind: float = 0.0
    importance: float = 0.0
    usage: float = 0.0
    total: float = 0.0
    intent: str = "general"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "fts": round(self.fts, 4),
            "path": round(self.path, 4),
            "symbol": round(self.symbol, 4),
            "tag": round(self.tag, 4),
            "kind": round(self.kind, 4),
            "importance": round(self.importance, 4),
            "usage": round(self.usage, 4),
            "intent": self.intent,
            "reasons": self.reasons,
        }


def score_memory(
    *,
    content: str,
    kind: str,
    tags: list[str],
    paths: list[str],
    symbols: list[str],
    importance: int,
    access_count: int,
    fts_raw: float,
    fts_norm: float,
    query: str,
    tokens: list[str],
    intent: str,
    query_paths: list[str],
) -> ScoreBreakdown:
    """Composite score for small-repo coding memory.

    Weights favor precise path/symbol/tag hits over fuzzy prose — agents
    working in <50k LOC care most about *where* and *what kind*.
    """
    bd = ScoreBreakdown(intent=intent)
    reasons: list[str] = []
    q_lower = query.lower()
    content_l = content.lower()
    tags_l = [t.lower() for t in tags]
    paths_l = [p.lower().replace("\\", "/") for p in paths]
    symbols_l = [s.lower() for s in symbols]

    # --- FTS / lexical ---
    bd.fts = 0.42 * fts_norm
    if fts_norm >= 0.75:
        reasons.append("strong text match")
    elif fts_norm >= 0.35:
        reasons.append("text match")

    # Token overlap fallback signal folded into fts channel
    if tokens:
        hay = " ".join([content_l, " ".join(tags_l), " ".join(paths_l), " ".join(symbols_l)])
        hits = sum(1 for t in tokens if t in hay)
        overlap = hits / len(tokens)
        if overlap > fts_norm:
            bd.fts = max(bd.fts, 0.42 * overlap)
            if overlap >= 0.5:
                reasons.append(f"token overlap {hits}/{len(tokens)}")

    # --- Path hits (high value in small codebases) ---
    path_score = 0.0
    for qp in query_paths:
        qp_l = qp.lower().replace("\\", "/")
        for p in paths_l:
            if qp_l == p or qp_l in p or p in qp_l:
                path_score = 1.0
                reasons.append(f"path:{p}")
                break
        if path_score >= 1.0:
            break
        # basename mention in query text
        base = qp_l.rsplit("/", 1)[-1]
        if base and (base in content_l or any(base in p for p in paths_l)):
            path_score = max(path_score, 0.7)
    if path_score == 0.0 and tokens:
        for p in paths_l:
            base = p.rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0]
            if any(t in p or t == stem or t == base for t in tokens):
                path_score = max(path_score, 0.55)
                reasons.append(f"path-token:{base}")
                break
    bd.path = 0.22 * path_score

    # --- Symbol hits ---
    sym_score = 0.0
    for s in symbols_l:
        if s in q_lower or any(t == s for t in tokens):
            sym_score = 1.0
            reasons.append(f"symbol:{s}")
            break
        if any(t in s or s in t for t in tokens if len(t) >= 3):
            sym_score = max(sym_score, 0.6)
    bd.symbol = 0.12 * sym_score

    # --- Tag hits ---
    tag_hits = sum(1 for t in tags_l if t in q_lower or t in tokens)
    if tags_l:
        bd.tag = 0.08 * min(1.0, tag_hits / max(1, min(3, len(tags_l))))
        if tag_hits:
            reasons.append(f"tags×{tag_hits}")

    # --- Kind / intent ---
    kind_boosts = _INTENT_KIND_BOOSTS.get(intent, {})
    bd.kind = kind_boosts.get(kind, 0.0)
    if bd.kind > 0:
        reasons.append(f"intent:{intent}→{kind}")

    # --- Importance ---
    bd.importance = 0.12 * (importance / 5.0)
    if importance >= 5:
        reasons.append("critical")
    elif importance >= 4:
        reasons.append("high importance")

    # --- Usage (soft) — frequently recalled memories are sticky ---
    bd.usage = 0.04 * min(1.0, access_count / 10.0)

    bd.total = (
        bd.fts + bd.path + bd.symbol + bd.tag + bd.kind + bd.importance + bd.usage
    )
    # Tiny stable tie-break from fts raw magnitude
    bd.total += min(abs(fts_raw), 10.0) * 1e-5
    bd.reasons = reasons[:6]
    return bd


def diversify(
    items: list[tuple[Any, ScoreBreakdown]],
    *,
    limit: int,
    max_per_kind: int = 3,
) -> list[tuple[Any, ScoreBreakdown]]:
    """Keep top scores but avoid flooding the agent with one kind."""
    if limit < 1:
        return []
    by_kind: dict[str, int] = defaultdict(int)
    out: list[tuple[Any, ScoreBreakdown]] = []
    deferred: list[tuple[Any, ScoreBreakdown]] = []

    for item, bd in items:
        kind = getattr(item, "kind", None) or "context"
        if by_kind[kind] < max_per_kind:
            out.append((item, bd))
            by_kind[kind] += 1
            if len(out) >= limit:
                return out
        else:
            deferred.append((item, bd))

    for item, bd in deferred:
        out.append((item, bd))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Context pack (session bootstrap for small codebases)
# ---------------------------------------------------------------------------


@dataclass
class ContextPack:
    """Dense working context for a coding agent at session start."""

    agent_id: str
    memory_count: int
    summary: str
    by_kind: dict[str, int]
    anchors: list[dict[str, Any]]  # high-signal memories to load into context
    guidance: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "memory_count": self.memory_count,
            "summary": self.summary,
            "by_kind": self.by_kind,
            "anchors": self.anchors,
            "guidance": self.guidance,
        }


def build_context_pack(
    agent_id: str,
    memories: list[Any],
    *,
    is_new_agent: bool,
    limit: int = DEFAULT_PACK_LIMIT,
) -> ContextPack:
    """Build a session-start pack for a small codebase.

    Prefers architecture → decisions → conventions → file maps so agents
    recover project shape quickly without stuffing the whole vault.
    """
    by_kind: dict[str, int] = defaultdict(int)
    for m in memories:
        by_kind[getattr(m, "kind", "context")] += 1

    if is_new_agent or not memories:
        summary = (
            f"New coding-agent vault for {agent_id!r} (local SQLite). "
            "No prior memories. Use remember() for decisions, file maps, "
            "bugs, conventions, and gotchas as you learn the codebase."
        )
        guidance = [
            "Prefer kind=file_map when noting where code lives (paths matter).",
            "Prefer kind=decision for architectural choices and trade-offs.",
            "Prefer kind=bug/gotcha for root causes that should never be relearned.",
            "Call recall() before non-trivial edits to load prior context.",
            "This vault is local-only — safe for project-private notes.",
        ]
        return ContextPack(
            agent_id=agent_id,
            memory_count=0,
            summary=summary,
            by_kind={},
            anchors=[],
            guidance=guidance,
        )

    # Score each memory for pack inclusion (importance + kind priority + usage)
    ranked: list[tuple[float, Any]] = []
    for m in memories:
        kind = getattr(m, "kind", "context")
        try:
            kind_rank = _PACK_KIND_ORDER.index(kind)
        except ValueError:
            kind_rank = len(_PACK_KIND_ORDER)
        imp = int(getattr(m, "importance", 3))
        access = int(getattr(m, "access_count", 0))
        # Higher is better
        score = (6 - min(kind_rank, 5)) * 1.2 + imp * 1.0 + min(access, 8) * 0.15
        ranked.append((score, m))
    ranked.sort(key=lambda x: x[0], reverse=True)

    # Diversify by kind for a balanced pack
    picked: list[Any] = []
    kind_counts: dict[str, int] = defaultdict(int)
    for _score, m in ranked:
        kind = getattr(m, "kind", "context")
        if kind_counts[kind] >= MAX_PER_KIND_IN_PACK:
            continue
        picked.append(m)
        kind_counts[kind] += 1
        if len(picked) >= limit:
            break

    anchors: list[dict[str, Any]] = []
    for m in picked:
        if hasattr(m, "to_dict"):
            d = m.to_dict()
        else:
            d = dict(m)
        # Compact anchor for agent context windows
        anchors.append(
            {
                "id": d.get("id"),
                "kind": d.get("kind", "context"),
                "importance": d.get("importance", 3),
                "paths": d.get("paths") or [],
                "symbols": d.get("symbols") or [],
                "tags": d.get("tags") or [],
                "content": d.get("content", ""),
            }
        )

    total = len(memories)
    kind_bits = ", ".join(f"{k}={by_kind[k]}" for k in _PACK_KIND_ORDER if by_kind.get(k))
    summary = (
        f"Resuming coding-agent vault {agent_id!r}: {total} local memories"
        f"{f' ({kind_bits})' if kind_bits else ''}. "
        f"Loaded {len(anchors)} high-signal anchors for this small-codebase session."
    )
    guidance = [
        "Use anchors below as durable project truth; call recall() for task-specific hits.",
        "Update the vault with remember() when you discover new decisions, paths, or bugs.",
        "Path-tagged memories rank higher for 'where is…' questions — keep paths filled in.",
    ]
    return ContextPack(
        agent_id=agent_id,
        memory_count=total,
        summary=summary,
        by_kind=dict(by_kind),
        anchors=anchors,
        guidance=guidance,
    )


def enrich_memory_fields(
    content: str,
    *,
    kind: str | None = None,
    tags: list[str] | None = None,
    paths: list[str] | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Normalize + auto-fill kind/paths/symbols for remember()."""
    tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
    path_list = [str(p).replace("\\", "/").strip() for p in (paths or []) if str(p).strip()]
    sym_list = [str(s).strip() for s in (symbols or []) if str(s).strip()]

    # Auto-extract from content when agent omits structured fields
    for p in extract_paths(content):
        if p not in path_list:
            path_list.append(p)
    for s in extract_symbols(content):
        if s not in sym_list:
            sym_list.append(s)

    resolved_kind = normalize_kind(kind) if kind else infer_kind(content, tag_list)
    search_doc = build_search_doc(
        content=content,
        kind=resolved_kind,
        tags=tag_list,
        paths=path_list,
        symbols=sym_list,
    )
    return {
        "kind": resolved_kind,
        "tags": tag_list,
        "paths": path_list,
        "symbols": sym_list,
        "search_doc": search_doc,
    }
