"""Configuration and filesystem layout.

Everything lives under ``SUPERMEMORY_HOME`` (default ``~/.supermemory``)::

    ~/.supermemory/
        memory.db        SQLite database (WAL)
        config.json      user overrides (optional, merged over DEFAULTS)
        backups/         automatic backups written by the installer
        bin/             launcher scripts
        app/             installed copy of the package (install.py)
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

DEFAULT_CONFIG: dict = {
    "version": 1,
    # ---------- embedding ----------
    "embedder": {
        # "local" | "openai" | "none"
        # local = deterministic hashed n-gram vectors, zero setup, offline.
        # openai = any OpenAI-compatible /v1/embeddings endpoint.
        "provider": "local",
        "dim": 1024,
        "model": "",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "SUPERMEMORY_EMBED_API_KEY",
    },
    # ---------- ranking (the "attention" layer) ----------
    "ranking": {
        "weights": {
            "lexical": 0.36,        # BM25 / FTS5
            "vector": 0.28,         # cosine
            "importance": 0.13,     # author-declared importance
            "recency": 0.09,        # exponential decay on last touch
            "reinforcement": 0.06,  # log(1 + times recalled)
            "pinned": 0.08,         # explicitly pinned
        },
        "temperature": 0.35,        # softmax temperature for attention weights
        "half_life_days": 45.0,     # recency decay half-life
        "mmr_lambda": 0.22,         # diversity penalty (0 = off, 1 = maximal)
        "confidence_floor": 0.25,   # memories below this are down-ranked
    },
    # ---------- recall ----------
    "recall": {
        "default_limit": 8,
        "max_tokens": 1400,
        "min_score": 0.04,
        # a candidate must clear one of these to be relevant at all; without a
        # gate, recency/importance alone drags unrelated memories into the top-k
        "vector_floor": 0.15,
    },
    # ---------- briefing (session hand-off) ----------
    "briefing": {
        "max_tokens": 900,
        "pinned": 6,
        "tasks": 8,
        "decisions": 5,
        "preferences": 5,
        "recent": 4,
        "relevant": 5,
    },
    # ---------- behaviour ----------
    "store": {
        "namespace_mode": "cwd",   # "cwd" | "fixed" | "none"
        "fixed_namespace": "default",
        "shared_namespace": "global",
        "include_shared": True,
        "chunk_max_tokens": 260,
        "chunk_overlap_tokens": 48,
        "summary_chars": 180,
        "dedupe_exact": True,
        "near_duplicate_hint": 0.90,
        "max_content_bytes": 512_000,
    },
    "maintenance": {
        "soft_delete_ttl_days": 30,
        "keep_events": 5000,
    },
}

KINDS = (
    "fact",
    "decision",
    "preference",
    "task",
    "procedure",
    "gotcha",
    "episode",
    "artifact",
    "reference",
)

_CONFIG_CACHE: dict = {"key": None, "config": None}


def home_dir() -> Path:
    """Return the SuperMemory home directory, creating it if needed."""
    raw = os.environ.get("SUPERMEMORY_HOME")
    path = Path(raw).expanduser() if raw else Path.home() / ".supermemory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return home_dir() / "memory.db"


def config_path() -> Path:
    return home_dir() / "config.json"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(reload: bool = False) -> dict:
    """Load defaults merged with ``config.json`` (cached on mtime)."""
    path = config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = (str(path), mtime)
    if not reload and _CONFIG_CACHE["key"] == key:
        return _CONFIG_CACHE["config"]

    override: dict = {}
    if mtime:
        try:
            override = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            override = {}
    merged = _deep_merge(DEFAULT_CONFIG, override if isinstance(override, dict) else {})

    # environment overrides for the embedder
    env = os.environ
    if env.get("SUPERMEMORY_EMBED_PROVIDER"):
        merged["embedder"]["provider"] = env["SUPERMEMORY_EMBED_PROVIDER"]
    if env.get("SUPERMEMORY_EMBED_MODEL"):
        merged["embedder"]["model"] = env["SUPERMEMORY_EMBED_MODEL"]
    if env.get("SUPERMEMORY_EMBED_BASE_URL"):
        merged["embedder"]["base_url"] = env["SUPERMEMORY_EMBED_BASE_URL"]
    if env.get("SUPERMEMORY_EMBED_DIM"):
        try:
            merged["embedder"]["dim"] = int(env["SUPERMEMORY_EMBED_DIM"])
        except ValueError:
            pass

    _CONFIG_CACHE["key"] = key
    _CONFIG_CACHE["config"] = merged
    return merged


def load_config_file(path: str | Path) -> dict:
    """Load a config file merged over ``DEFAULT_CONFIG``.

    Unlike :func:`load_config`, this ignores environment overrides and the
    default home directory, so offline tools (the benchmark) get a reproducible
    configuration.
    """
    target = Path(path).expanduser()
    try:
        override = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read config {target}: {exc}") from exc
    return _deep_merge(DEFAULT_CONFIG, override if isinstance(override, dict) else {})


def save_config(config: dict) -> None:
    path = config_path()
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _CONFIG_CACHE["key"] = None


def sanitize_namespace(name: str) -> str:
    """Normalise a namespace: lowercase, no path separators, bounded length."""
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_.") else "-" for ch in (name or "").strip()
    ).strip("-.")
    cleaned = cleaned.lower()[:64]
    return cleaned or "default"


def current_namespace(config: dict | None = None) -> str:
    """Namespace for this process: explicit env > configured mode > cwd."""
    explicit = os.environ.get("SUPERMEMORY_NAMESPACE")
    if explicit and explicit != "*":
        return sanitize_namespace(explicit)
    cfg = config or load_config()
    mode = cfg["store"].get("namespace_mode", "cwd")
    if mode == "fixed":
        return sanitize_namespace(cfg["store"].get("fixed_namespace", "default"))
    if mode == "none":
        return "global"
    # stdio MCP servers inherit the launch directory, which is not necessarily
    # the project root. CLAUDE_PROJECT_DIR is the reliable signal in Claude Code;
    # SUPERMEMORY_PROJECT_CWD is set by our own hooks.
    cwd = (
        os.environ.get("SUPERMEMORY_PROJECT_CWD")
        or os.environ.get("CLAUDE_PROJECT_DIR")
        or os.getcwd()
    )
    return sanitize_namespace(Path(cwd.rstrip("/\\")).name or "default")


def resolve_namespaces(namespace: str | None, config: dict | None = None) -> list[str] | None:
    """Return the namespaces to search, or ``None`` meaning "all".

    ``*`` (or the env value) searches everything; an explicit name searches only
    that name; the default searches the current project plus the shared
    ``global`` namespace.
    """
    cfg = config or load_config()
    if namespace is not None:
        if namespace.strip() in ("*", "all"):
            return None
        return [sanitize_namespace(namespace)]
    if os.environ.get("SUPERMEMORY_NAMESPACE", "").strip() == "*":
        return None
    current = current_namespace(cfg)
    shared = sanitize_namespace(cfg["store"].get("shared_namespace", "global"))
    if cfg["store"].get("include_shared", True) and current != shared:
        return [current, shared]
    return [current]
