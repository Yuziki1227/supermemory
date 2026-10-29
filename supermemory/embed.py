"""Pluggable embedding backends.

Default: a deterministic, offline *hashed* embedder. It hashes word, identifier
and CJK-ngram features into a fixed-width vector with sign hashing and
sublinear term weighting. No downloads, no API keys, no network — the memory
layer works the moment it is installed, and vectors stay comparable across
processes because blake2b is deterministic.

Optional: any OpenAI-compatible ``/v1/embeddings`` endpoint for real semantic
vectors. Set ``embedder.provider = "openai"`` in ``config.json`` or export
``SUPERMEMORY_EMBED_PROVIDER=openai`` plus an API key.

If the remote provider is unreachable the process permanently falls back to the
local embedder and says so on stderr — a memory server that refuses to start
because an API key expired is worse than one with slightly weaker recall.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import lru_cache

import numpy as np

from .text import tokenize


def _log(message: str) -> None:
    print(f"[supermemory] {message}", file=sys.stderr, flush=True)


class Embedder:
    name: str = "none"
    dim: int = 0
    provider: str = "none"

    def embed_many(self, texts: list[str]) -> np.ndarray | None:
        return None

    def embed_one(self, text: str) -> np.ndarray | None:
        many = self.embed_many([text])
        return None if many is None else many[0]

    def available(self) -> bool:
        return self.dim > 0


class NullEmbedder(Embedder):
    """Vector search disabled; lexical + heuristic signals only."""


class LocalEmbedder(Embedder):
    provider = "local"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = int(dim)
        self.name = f"local-hash-v1-{self.dim}"

    @staticmethod
    @lru_cache(maxsize=262_144)
    def _bucket(token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8", "replace"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value & 0xFFFFFFFF, 1.0 if value >> 63 else -1.0

    def _features(self, text: str) -> Counter:
        counts: Counter = Counter()
        for tok in tokenize(text):
            counts[tok] += 1.0
            if tok.isascii() and len(tok) >= 5:
                # character trigrams give typo/morphology tolerance
                for i in range(len(tok) - 2):
                    counts["##" + tok[i : i + 3]] += 0.35
        return counts

    def embed_many(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok, tf in self._features(text or "").items():
                index, sign = self._bucket(tok)
                matrix[row, index % self.dim] += sign * (1.0 + math.log(tf))
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


class OpenAIEmbedder(Embedder):
    provider = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        dim: int = 0,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or "text-embedding-3-small"
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.requested_dim = int(dim or 0)
        self.dim = self.requested_dim
        self.timeout = timeout
        self.name = f"openai-{self.model}"

    def _post(self, batch: list[str]) -> np.ndarray:
        payload: dict = {"model": self.model, "input": batch}
        if self.requested_dim:
            payload["dimensions"] = self.requested_dim
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                vectors = [item["embedding"] for item in data["data"]]
                matrix = np.asarray(vectors, dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return matrix / norms
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last_error = exc
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError(f"embedding request failed: {last_error}")

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), 64):
            out.append(self._post(texts[start : start + 64]))
        matrix = np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)
        if not self.dim:
            self.dim = int(matrix.shape[1])
            self.name = f"openai-{self.model}"
        return matrix

    def probe(self) -> bool:
        try:
            vector = self.embed_one("supermemory probe")
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            _log(f"remote embedder unavailable ({exc}); using local hash embedder")
            return False
        if vector is None:
            return False
        self.dim = int(vector.shape[0])
        return True


def get_embedder(config: dict) -> Embedder:
    """Build the configured embedder, degrading to local on any failure."""
    cfg = (config or {}).get("embedder", {})
    provider = str(cfg.get("provider", "local")).lower()

    if provider == "none":
        return NullEmbedder()

    if provider == "openai":
        key_env = cfg.get("api_key_env") or "SUPERMEMORY_EMBED_API_KEY"
        api_key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            _log(f"embedder.provider=openai but ${key_env} is empty; using local")
            return LocalEmbedder(int(cfg.get("dim") or 1024))
        candidate = OpenAIEmbedder(
            model=cfg.get("model") or "text-embedding-3-small",
            base_url=cfg.get("base_url") or "https://api.openai.com/v1",
            api_key=api_key,
            dim=int(cfg.get("dim") or 0),
        )
        if candidate.probe():
            return candidate
        return LocalEmbedder(int(cfg.get("dim") or 1024))

    return LocalEmbedder(int(cfg.get("dim") or 1024))
