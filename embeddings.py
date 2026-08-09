"""Optional semantic re-rank for recall — the 'integrations / API key' path.

Off by default: recall stays purely lexical (FTS5 + multi-signal), with zero
keys and zero network. Enable with environment:

    CONTEXTVAULT_EMBEDDINGS=openai   # OpenAI-compatible /embeddings API
        OPENAI_API_KEY=...            # required
        EMBEDDING_MODEL=text-embedding-3-small
        EMBEDDING_BASE_URL=https://api.openai.com/v1   # OpenAI, Azure, local
    CONTEXTVAULT_EMBEDDINGS=local    # sentence-transformers on this machine
        EMBEDDING_MODEL=all-MiniLM-L6-v2

Everything degrades gracefully: if the embedder is unconfigured, missing, or
fails, recall silently falls back to the lexical engine.

**Privacy note:** `openai` mode sends memory content to the embedding provider
(their servers see your notes). `local` mode keeps vectors on this machine but
downloads a model once. Both are opt-in — the default vault never sends data
anywhere.
"""

from __future__ import annotations

import os
from typing import Any


class EmbedderError(Exception):
    """Raised when an embedder is configured but cannot be used."""


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1] (clamped; vectors are non-negative-ish)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


class _BaseEmbedder:
    """Embedding backend contract."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(_BaseEmbedder):
    """OpenAI-compatible /embeddings endpoint (stdlib only, no extra deps)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        batch_size: int = 64,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.request

        url = f"{self.base_url}/embeddings"
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            body = json.dumps({"model": self.model, "input": batch}).encode()
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - user opted in
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise EmbedderError(f"embedding request failed: {exc}") from exc
            if "data" not in data:
                raise EmbedderError(f"embedding response missing 'data': {data}")
            for item in data["data"]:
                vec = item.get("embedding")
                if not isinstance(vec, list):
                    raise EmbedderError(f"bad embedding in response: {item}")
                out.append([float(v) for v in vec])
        return out


class LocalEmbedder(_BaseEmbedder):
    """sentence-transformers on this machine (model downloaded once)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.model = model
        self._enc: Any = None

    def _get_enc(self) -> Any:
        if self._enc is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise EmbedderError(
                    "local embeddings need 'sentence-transformers' "
                    "(pip install 'contextvault[embed]')"
                ) from exc
            try:
                self._enc = SentenceTransformer(self.model)
            except Exception as exc:  # noqa: BLE001 - model download/load
                raise EmbedderError(f"failed to load model {self.model!r}: {exc}") from exc
        return self._enc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        enc = self._get_enc()
        try:
            vecs = enc.encode(texts, normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001
            raise EmbedderError(f"local embedding failed: {exc}") from exc
        return [[float(v) for v in row] for row in vecs]


def get_embedder() -> _BaseEmbedder | None:
    """Build the embedder from environment; None when disabled/unconfigured.

    Raises EmbedderError only when the user explicitly enabled a mode that
    cannot be built (e.g. openai without a key) — callers degrade to lexical.
    """
    mode = os.environ.get("CONTEXTVAULT_EMBEDDINGS", "").strip().lower()
    if mode in ("", "off", "0", "none", "false", "disabled"):
        return None
    if mode in ("openai", "api"):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EmbedderError(
                "CONTEXTVAULT_EMBEDDINGS=openai requires OPENAI_API_KEY"
            )
        return OpenAIEmbedder(
            api_key,
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.environ.get(
                "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
            ),
        )
    if mode == "local":
        return LocalEmbedder(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    raise EmbedderError(f"unknown CONTEXTVAULT_EMBEDDINGS mode {mode!r}")
