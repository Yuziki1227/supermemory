"""MemoryStore — the domain layer shared by the MCP server, CLI and hooks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import config as config_mod
from . import db as db_mod
from .embed import Embedder, get_embedder
from .ranking import mmr_select, score_candidates
from .text import (
    chunk_text,
    estimate_tokens,
    fts_expression,
    index_blob,
    query_tokens,
    summarize,
)

# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class Memory:
    id: int
    namespace: str
    kind: str
    title: str
    content: str
    summary: str
    tags: list[str]
    source: str
    source_ref: str
    importance: float
    confidence: float
    pinned: bool
    created_at: float
    updated_at: float
    accessed_at: float | None
    access_count: int
    valid_to: float | None
    superseded_by: int | None
    tokens: int

    @property
    def label(self) -> str:
        return f"M{self.id}"

    def to_dict(self, include_content: bool = True) -> dict:
        data = {
            "id": self.id,
            "label": self.label,
            "namespace": self.namespace,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "source": self.source,
            "source_ref": self.source_ref,
            "importance": round(self.importance, 3),
            "confidence": round(self.confidence, 3),
            "pinned": self.pinned,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "access_count": self.access_count,
            "tokens": self.tokens,
        }
        if include_content:
            data["content"] = self.content
        return data


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    attention: float
    features: dict = field(default_factory=dict)

    def to_dict(self, include_content: bool = False) -> dict:
        data = self.memory.to_dict(include_content=include_content)
        data["score"] = round(self.score, 4)
        data["attention"] = round(self.attention, 4)
        data["features"] = self.features
        return data


@dataclass
class RememberResult:
    memories: list[Memory]
    duplicates: list[Memory] = field(default_factory=list)
    near_duplicates: list[dict] = field(default_factory=list)
    chunked: bool = False


def _iso(ts: float | None) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def parse_id(value: Any) -> int:
    """Accept 12, "12", "M12", "#12", "m-12"."""
    if isinstance(value, int):
        return value
    text = str(value).strip().lower().lstrip("#")
    if text.startswith("m"):
        text = text[1:].lstrip("-_ ")
    match = re.search(r"\d+", text)
    if not match:
        raise ValueError(f"not a memory id: {value!r}")
    return int(match.group())


def _normalize_hash(namespace: str, kind: str, content: str) -> str:
    payload = f"{namespace}\x00{kind}\x00{content.strip()}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _row_to_memory(row: sqlite3.Row) -> Memory:
    try:
        tags = json.loads(row["tags"] or "[]")
    except (TypeError, ValueError):
        tags = []
    return Memory(
        id=int(row["id"]),
        namespace=row["namespace"],
        kind=row["kind"],
        title=row["title"] or "",
        content=row["content"],
        summary=row["summary"] or "",
        tags=tags if isinstance(tags, list) else [],
        source=row["source"] or "",
        source_ref=row["source_ref"] or "",
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        pinned=bool(row["pinned"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        accessed_at=row["accessed_at"],
        access_count=int(row["access_count"]),
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
        tokens=int(row["tokens"] or 0),
    )


# --------------------------------------------------------------------------- #
# Vector index cache
# --------------------------------------------------------------------------- #

_VEC_CACHE: dict = {"key": None, "ids": np.empty(0, dtype=np.int64), "matrix": None,
                    "kinds": [], "vectors": {}}


class MemoryStore:
    def __init__(
        self,
        namespace: str | None = None,
        con: sqlite3.Connection | None = None,
        config: dict | None = None,
        embedder: Embedder | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.config = config or config_mod.load_config()
        self.namespace = config_mod.sanitize_namespace(
            namespace or config_mod.current_namespace(self.config)
        )
        self.con = con or db_mod.connect(path)
        self.embedder = embedder or get_embedder(self.config)
        self._db_id = str(path) if path else str(config_mod.db_path())

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        try:
            self.con.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @contextmanager
    def _tx(self):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        else:
            self.con.execute("COMMIT")

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #

    def remember(
        self,
        content: str,
        *,
        title: str = "",
        kind: str = "fact",
        tags: Iterable[str] | None = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        source: str = "",
        source_ref: str = "",
        pinned: bool = False,
        namespace: str | None = None,
        chunk: bool | None = None,
        dedupe: bool | None = None,
        ttl_days: float | None = None,
    ) -> RememberResult:
        """Store one or more memory chunks. Long content is chunked."""
        cfg = self.config
        store_cfg = cfg["store"]
        content = (content or "").strip()
        if not content:
            raise ValueError("content is empty")
        limit = int(store_cfg.get("max_content_bytes", 512_000))
        if len(content.encode("utf-8", "replace")) > limit:
            raise ValueError(f"content too large (> {limit} bytes); split it first")

        ns = config_mod.sanitize_namespace(namespace or self.namespace)
        kind = (kind or "fact").strip().lower()[:32] or "fact"
        tag_list = _clean_tags(tags)
        do_dedupe = store_cfg.get("dedupe_exact", True) if dedupe is None else dedupe

        max_tokens = int(store_cfg.get("chunk_max_tokens", 260))
        overlap = int(store_cfg.get("chunk_overlap_tokens", 48))
        do_chunk = chunk if chunk is not None else True
        pieces = (
            chunk_text(content, max_tokens, overlap) if do_chunk else [content]
        ) or [content]

        result = RememberResult(memories=[], chunked=len(pieces) > 1)
        for index, piece in enumerate(pieces):
            part_title = title
            if len(pieces) > 1:
                part_title = f"{title or summarize(content, 48)} [{index + 1}/{len(pieces)}]"
            memory, duplicate = self._insert(
                content=piece,
                title=part_title,
                kind=kind,
                tags=tag_list,
                importance=importance,
                confidence=confidence,
                source=source,
                source_ref=source_ref,
                pinned=pinned,
                namespace=ns,
                dedupe=do_dedupe,
                ttl_days=ttl_days,
            )
            if duplicate is not None:
                result.duplicates.append(memory)
                if importance > memory.importance:
                    self.con.execute(
                        "UPDATE memories SET importance=? WHERE id=?",
                        (_clamp(importance, 0.0, 1.0), memory.id),
                    )
            else:
                result.memories.append(memory)
                # Only hint on the first chunk of a chunked document: checking
                # every chunk would rebuild the vector index once per insert.
                if index == 0:
                    hints = self._near_duplicate_hints(memory)
                    if hints:
                        result.near_duplicates.extend(hints)
        return result

    def _insert(
        self,
        *,
        content: str,
        title: str,
        kind: str,
        tags: list[str],
        importance: float,
        confidence: float,
        source: str,
        source_ref: str,
        pinned: bool,
        namespace: str,
        dedupe: bool,
        ttl_days: float | None,
        supersedes: int | None = None,
    ) -> tuple[Memory, Memory | None]:
        now = time.time()
        content_hash = _normalize_hash(namespace, kind, content)

        if dedupe and supersedes is None:
            existing = self.con.execute(
                "SELECT * FROM memories WHERE namespace=? AND kind=? AND content_hash=?"
                " AND deleted=0 AND superseded_by IS NULL LIMIT 1",
                (namespace, kind, content_hash),
            ).fetchone()
            if existing is not None:
                return _row_to_memory(existing), _row_to_memory(existing)

        title = (title or "").strip() or _derive_title(content)
        summary = summarize(content, int(self.config["store"].get("summary_chars", 180)))
        token_count = estimate_tokens(content)
        search_text = index_blob(title, title, summary, content,
                                 " ".join(tags), " ".join(tags), kind, kind)
        importance = _clamp(importance, 0.0, 1.0)
        confidence = _clamp(confidence, 0.0, 1.0)
        valid_to = now + ttl_days * 86400 if ttl_days else None

        vector = self.embedder.embed_one(f"{title}\n{content}\n{' '.join(tags)}")
        blob = None
        model_name = ""
        if vector is not None:
            blob = np.asarray(vector, dtype=np.float32).tobytes()
            model_name = self.embedder.name

        with self._tx():
            cursor = self.con.execute(
                """INSERT INTO memories(
                    namespace, kind, title, content, summary, tags, source, source_ref,
                    importance, confidence, pinned, created_at, updated_at,
                    valid_from, valid_to, content_hash, tokens, embedding,
                    embedding_model, search_text)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    namespace, kind, title, content, summary, json.dumps(tags, ensure_ascii=False),
                    source, source_ref, importance, confidence, int(bool(pinned)),
                    now, now, now, valid_to, content_hash, token_count, blob,
                    model_name, search_text,
                ),
            )
            memory_id = int(cursor.lastrowid)
            db_mod.fts_upsert(self.con, memory_id, search_text)

            if supersedes is not None:
                self.con.execute(
                    "UPDATE memories SET superseded_by=?, valid_to=?, updated_at=? WHERE id=?",
                    (memory_id, now, now, supersedes),
                )
                self.con.execute(
                    "INSERT OR REPLACE INTO links(src, dst, relation, weight, created_at)"
                    " VALUES(?,?,?,?,?)",
                    (supersedes, memory_id, "revises", 1.0, now),
                )
            self._event("remember", memory_id, f"{namespace}/{kind}")

        self._invalidate_vectors()
        row = self.con.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        assert row is not None
        return _row_to_memory(row), None

    def update(self, memory_id: Any, **changes: Any) -> Memory:
        """Revise a memory: writes a new version, supersedes the old one."""
        old = self.get(memory_id, include_superseded=True)
        if old is None:
            raise KeyError(f"memory {memory_id} not found")
        allowed = {
            "content", "title", "kind", "tags", "importance", "confidence",
            "pinned", "source", "source_ref", "namespace", "ttl_days",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
        if not changes:
            return old

        new, _ = self._insert(
            content=str(changes.get("content") or old.content),
            title=str(changes.get("title") if changes.get("title") is not None else old.title),
            kind=str(changes.get("kind") or old.kind),
            tags=_clean_tags(changes.get("tags", old.tags)),
            importance=float(changes.get("importance", old.importance)),
            confidence=float(changes.get("confidence", old.confidence)),
            source=str(changes.get("source", old.source)),
            source_ref=str(changes.get("source_ref", old.source_ref)),
            pinned=bool(changes.get("pinned", old.pinned)),
            namespace=str(changes.get("namespace") or old.namespace),
            dedupe=False,
            ttl_days=changes.get("ttl_days"),
            supersedes=old.id,
        )
        return new

    def forget(self, ids: Iterable[Any], hard: bool = False) -> int:
        target = [parse_id(i) for i in ids]
        if not target:
            return 0
        existing = self.con.execute(
            f"SELECT COUNT(*) FROM memories WHERE id IN ({','.join('?' * len(target))})"
            " AND deleted=0",
            target,
        ).fetchone()[0]
        now = time.time()
        placeholders = ",".join("?" * len(target))
        with self._tx():
            if hard:
                self.con.execute(
                    f"DELETE FROM links WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
                    (*target, *target),
                )
                for memory_id in target:
                    db_mod.fts_delete(self.con, memory_id)
                self.con.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", target)
                op = "forget-hard"
            else:
                self.con.execute(
                    f"UPDATE memories SET deleted=1, valid_to=?, updated_at=?"
                    f" WHERE id IN ({placeholders})",
                    (now, now, *target),
                )
                for memory_id in target:
                    db_mod.fts_delete(self.con, memory_id)
                op = "forget"
            self._event(op, None, ",".join(map(str, target)))
        self._invalidate_vectors()
        return int(existing)

    def link(self, src: Any, dst: Any, relation: str = "related", weight: float = 1.0) -> None:
        src_id, dst_id = parse_id(src), parse_id(dst)
        now = time.time()
        with self._tx():
            self.con.execute(
                "INSERT OR REPLACE INTO links(src, dst, relation, weight, created_at)"
                " VALUES(?,?,?,?,?)",
                (src_id, dst_id, (relation or "related").strip()[:32], float(weight), now),
            )
            self._event("link", src_id, f"->{dst_id}:{relation}")

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #

    def get(
        self,
        ids: Any,
        include_superseded: bool = True,
        include_content: bool = True,
    ) -> Memory | list[Memory] | None:
        single = not isinstance(ids, (list, tuple, set))
        wanted = [parse_id(ids)] if single else [parse_id(i) for i in ids]
        if not wanted:
            return []
        placeholders = ",".join("?" * len(wanted))
        rows = self.con.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND deleted=0",
            wanted,
        ).fetchall()
        by_id = {int(row["id"]): _row_to_memory(row) for row in rows}
        ordered = [by_id[i] for i in wanted if i in by_id]
        for memory in ordered:
            if not include_content:
                memory.content = ""
        if single:
            return ordered[0] if ordered else None
        return ordered

    def list_memories(
        self,
        *,
        kind: str | None = None,
        namespace: str | None = None,
        limit: int = 50,
        include_superseded: bool = False,
        pinned_only: bool = False,
    ) -> list[Memory]:
        nss = self._scope(namespace)
        ns_sql, ns_params = _namespace_sql(nss, alias="m")
        clauses = ["m.deleted=0", "(m.valid_to IS NULL OR m.valid_to > ?)"]
        params: list[Any] = [time.time()]
        if not include_superseded:
            clauses.append("m.superseded_by IS NULL")
        if kind:
            clauses.append("m.kind=?")
            params.append(kind.strip().lower())
        if pinned_only:
            clauses.append("m.pinned=1")
        rows = self.con.execute(
            "SELECT m.* FROM memories m WHERE " + " AND ".join(clauses) + ns_sql
            + " ORDER BY m.pinned DESC, m.updated_at DESC LIMIT ?",
            (*params, *ns_params, int(limit)),
        ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def recall(
        self,
        query: str = "",
        *,
        limit: int | None = None,
        kinds: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        namespace: str | None = None,
        max_tokens: int | None = None,
        include_superseded: bool = False,
        mmr: bool = True,
        touch: bool = True,
    ) -> list[ScoredMemory]:
        cfg = self.config
        recall_cfg = cfg["recall"]
        limit = int(limit or recall_cfg.get("default_limit", 8))
        max_tokens = max_tokens if max_tokens is not None else int(recall_cfg.get("max_tokens", 1400))
        now = time.time()
        query = (query or "").strip()
        nss = self._scope(namespace)
        kind_filter = {k.strip().lower() for k in (kinds or []) if str(k).strip()}
        tag_filter = {t.strip().lower() for t in (tags or []) if str(t).strip()}
        pool = max(limit * 8, 64)

        lexical: dict[int, float] = {}
        if query:
            lexical = self._lexical_search(
                query, nss, kind_filter, include_superseded, pool
            )

        cosine: dict[int, float] = {}
        vectors: dict[int, np.ndarray] = {}
        query_vector = None
        if query and self.embedder.available():
            query_vector = self.embedder.embed_one(query)
        if query_vector is not None:
            ids, matrix, row_kinds = self._vector_index(nss, include_superseded)
            if matrix is not None and len(ids):
                sims = matrix @ query_vector
                order = np.argsort(-sims)[:pool]
                for position in order:
                    score = float(sims[position])
                    if score <= 0.0:
                        continue
                    memory_id = int(ids[position])
                    if kind_filter and row_kinds[position] not in kind_filter:
                        continue
                    cosine[memory_id] = score
                    vectors[memory_id] = matrix[position]

        candidate_ids = set(lexical) | set(cosine)
        if not query:
            candidate_ids = self._fallback_ids(nss, kind_filter, include_superseded, pool)

        if not candidate_ids:
            return []

        rows = self._fetch_rows(candidate_ids, nss, include_superseded, now)
        vector_floor = float(self.config["recall"].get("vector_floor", 0.15))
        candidates: list[dict] = []
        for row in rows:
            memory = _row_to_memory(row)
            if tag_filter and not tag_filter.issubset({t.lower() for t in memory.tags}):
                continue
            if kind_filter and memory.kind not in kind_filter:
                continue
            if query:
                # relevance gate: lexical hit or a genuinely close vector
                if lexical.get(memory.id, 0.0) <= 0.0 and cosine.get(memory.id, 0.0) < vector_floor:
                    continue
            candidates.append(
                {
                    "id": memory.id,
                    "memory": memory,
                    "lex": lexical.get(memory.id, 0.0),
                    "cosine": cosine.get(memory.id, 0.0),
                    "importance": memory.importance,
                    "confidence": memory.confidence,
                    "pinned": memory.pinned,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                    "accessed_at": memory.accessed_at,
                    "access_count": memory.access_count,
                }
            )
        if not candidates:
            return []

        scored = score_candidates(candidates, cfg, now)
        costs = {
            cand["id"]: estimate_tokens(cand["memory"].summary) + 28
            for cand in scored
        }
        if mmr:
            picked = mmr_select(
                scored, limit, max_tokens, cfg,
                vectors={mid: v for mid, v in vectors.items()},
                costs=costs,
            )
        else:
            picked = scored[:limit]

        results = [
            ScoredMemory(
                memory=cand["memory"],
                score=cand["score"],
                attention=cand["attention"],
                features=cand["features"],
            )
            for cand in picked
        ]
        if touch and results:
            self._touch([r.memory.id for r in results])
        return results

    def _lexical_search(
        self,
        query: str,
        namespaces: list[str] | None,
        kinds: set[str],
        include_superseded: bool,
        pool: int,
    ) -> dict[int, float]:
        tokens = query_tokens(query)
        if not tokens:
            return {}
        ns_sql, ns_params = _namespace_sql(namespaces)
        kind_sql = ""
        kind_params: list[str] = []
        if kinds:
            kind_sql = f" AND m.kind IN ({','.join('?' * len(kinds))})"
            kind_params = sorted(kinds)

        base = (
            "SELECT m.id AS id, bm25(memories_fts) AS rank"
            " FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid"
            " WHERE memories_fts MATCH ? AND m.deleted=0"
            + ("" if include_superseded else " AND m.superseded_by IS NULL AND (m.valid_to IS NULL OR m.valid_to > ?)")
            + ns_sql
            + kind_sql
            + " ORDER BY rank LIMIT ?"
        )

        attempts = []
        if len(tokens) <= 4:
            attempts.append(fts_expression(tokens, "AND"))
        attempts.append(fts_expression(tokens, "OR"))

        now = time.time()
        tail_params: list = []
        if not include_superseded:
            tail_params.append(now)
        tail_params.extend(ns_params)
        tail_params.extend(kind_params)
        tail_params.append(pool)

        for expression in attempts:
            params = (expression, *tail_params)
            try:
                rows = self.con.execute(base, params).fetchall()
            except sqlite3.OperationalError:
                continue
            if rows:
                return {int(row["id"]): max(0.0, -float(row["rank"])) for row in rows}
        return {}

    def _vector_index(
        self, namespaces: list[str] | None, include_superseded: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        model = self.embedder.name if self.embedder.available() else ""
        if not model:
            return np.empty(0, dtype=np.int64), None, []
        signature = self.con.execute(
            "SELECT COUNT(*), COALESCE(MAX(id),0), COALESCE(SUM(updated_at),0)"
            " FROM memories"
        ).fetchone()
        key = (
            self._db_id,
            model,
            tuple(namespaces) if namespaces is not None else None,
            include_superseded,
            int(signature[0]),
            int(signature[1]),
            round(float(signature[2]), 3),
        )
        if _VEC_CACHE["key"] == key and _VEC_CACHE["matrix"] is not None:
            return _VEC_CACHE["ids"], _VEC_CACHE["matrix"], _VEC_CACHE["kinds"]

        ns_sql, ns_params = _namespace_sql(namespaces, alias="")
        rows = self.con.execute(
            "SELECT id, kind, embedding FROM memories"
            " WHERE embedding IS NOT NULL AND embedding_model=? AND deleted=0"
            + ("" if include_superseded else " AND superseded_by IS NULL")
            + ns_sql,
            (model, *ns_params),
        ).fetchall()
        if not rows:
            _VEC_CACHE.update({"key": key, "ids": np.empty(0, dtype=np.int64),
                               "matrix": None, "kinds": [], "vectors": {}})
            return np.empty(0, dtype=np.int64), None, []

        ids = np.empty(len(rows), dtype=np.int64)
        kinds: list[str] = []
        vectors = []
        for index, row in enumerate(rows):
            ids[index] = int(row["id"])
            kinds.append(row["kind"])
            vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
        matrix = np.vstack(vectors)
        _VEC_CACHE.update({"key": key, "ids": ids, "matrix": matrix, "kinds": kinds})
        return ids, matrix, kinds

    def _fallback_ids(
        self,
        namespaces: list[str] | None,
        kinds: set[str],
        include_superseded: bool,
        pool: int,
    ) -> set[int]:
        ns_sql, ns_params = _namespace_sql(namespaces, alias="")
        kind_sql = ""
        kind_params: list[str] = []
        if kinds:
            kind_sql = f" AND kind IN ({','.join('?' * len(kinds))})"
            kind_params = sorted(kinds)
        params: list = []
        if not include_superseded:
            params.append(time.time())
        params.extend(ns_params)
        params.extend(kind_params)
        params.append(pool)
        rows = self.con.execute(
            "SELECT id FROM memories WHERE deleted=0"
            + ("" if include_superseded else " AND superseded_by IS NULL AND (valid_to IS NULL OR valid_to > ?)")
            + ns_sql
            + kind_sql
            + " ORDER BY pinned DESC, importance DESC, updated_at DESC LIMIT ?",
            tuple(params),
        ).fetchall()
        return {int(row["id"]) for row in rows}

    def _fetch_rows(
        self,
        ids: set[int],
        namespaces: list[str] | None,
        include_superseded: bool,
        now: float,
    ) -> list[sqlite3.Row]:
        if not ids:
            return []
        id_list = sorted(ids)
        ns_sql, ns_params = _namespace_sql(namespaces, alias="")
        rows = self.con.execute(
            f"SELECT * FROM memories WHERE id IN ({','.join('?' * len(id_list))})"
            " AND deleted=0"
            + ("" if include_superseded else " AND superseded_by IS NULL AND (valid_to IS NULL OR valid_to > ?)")
            + ns_sql,
            (*id_list, *( [] if include_superseded else [now]), *ns_params),
        ).fetchall()
        return rows

    def _touch(self, ids: list[int]) -> None:
        if not ids:
            return
        now = time.time()
        placeholders = ",".join("?" * len(ids))
        try:
            self.con.execute(
                f"UPDATE memories SET access_count=access_count+1, accessed_at=?"
                f" WHERE id IN ({placeholders})",
                (now, *ids),
            )
        except sqlite3.Error:
            pass

    def _near_duplicate_hints(self, memory: Memory, threshold: float | None = None) -> list[dict]:
        threshold = threshold if threshold is not None else float(
            self.config["store"].get("near_duplicate_hint", 0.90)
        )
        if not self.embedder.available():
            return []
        ids, matrix, kinds = self._vector_index([memory.namespace], include_superseded=False)
        if matrix is None or not len(ids):
            return []
        vector = self.embedder.embed_one(f"{memory.title}\n{memory.content}")
        if vector is None:
            return []
        sims = matrix @ vector
        order = np.argsort(-sims)[:8]
        hints: list[dict] = []
        for position in order:
            memory_id = int(ids[position])
            if memory_id == memory.id:
                continue
            score = float(sims[position])
            if score < threshold:
                continue
            row = self.con.execute(
                "SELECT id, title, kind, summary FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None or row["kind"] != memory.kind:
                continue
            hints.append(
                {
                    "id": int(row["id"]),
                    "label": f"M{int(row['id'])}",
                    "title": row["title"],
                    "summary": row["summary"],
                    "similarity": round(score, 3),
                }
            )
            if len(hints) >= 3:
                break
        return hints

    # ------------------------------------------------------------------ #
    # Briefing — the session hand-off document
    # ------------------------------------------------------------------ #

    def briefing(
        self,
        query: str = "",
        max_tokens: int | None = None,
        namespace: str | None = None,
    ) -> str:
        from .render import render_briefing

        cfg = self.config
        budget = int(max_tokens or cfg["briefing"].get("max_tokens", 900))
        nss = self._scope(namespace)
        scope_label = (
            config_mod.sanitize_namespace(namespace) if namespace else self.namespace
        )
        now = time.time()

        def pick(sql_where: str, params: tuple, limit: int) -> list[Memory]:
            ns_sql, ns_params = _namespace_sql(nss, alias="m")
            rows = self.con.execute(
                "SELECT m.* FROM memories m WHERE m.deleted=0 AND m.superseded_by IS NULL"
                " AND (m.valid_to IS NULL OR m.valid_to > ?)"
                + ns_sql + sql_where
                + " ORDER BY m.importance DESC, m.updated_at DESC LIMIT ?",
                (now, *ns_params, *params, limit),
            ).fetchall()
            return [_row_to_memory(row) for row in rows]

        sections: list[tuple[str, list[Memory]]] = []
        sections.append(("固定记忆", pick(" AND m.pinned=1", (), int(cfg["briefing"]["pinned"]))))
        sections.append(("进行中任务", pick(" AND m.kind='task'", (), int(cfg["briefing"]["tasks"]))))
        sections.append(("关键决策", pick(" AND m.kind='decision'", (), int(cfg["briefing"]["decisions"]))))
        sections.append(("偏好与约束", pick(" AND m.kind IN ('preference','gotcha')", (),
                                            int(cfg["briefing"]["preferences"]))))
        sections.append(("最近更新", pick("", (), int(cfg["briefing"]["recent"]))))

        relevant: list[ScoredMemory] = []
        if query.strip():
            relevant = self.recall(
                query, limit=int(cfg["briefing"].get("relevant", 5)),
                max_tokens=max(240, budget // 3), touch=False,
            )

        return render_briefing(
            namespace=scope_label,
            namespaces=nss,
            sections=sections,
            relevant=relevant,
            stats=self.stats(),
            budget=budget,
        )

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        total = self.con.execute("SELECT COUNT(*) FROM memories WHERE deleted=0").fetchone()[0]
        current = self.con.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted=0 AND superseded_by IS NULL"
        ).fetchone()[0]
        superseded = self.con.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL"
        ).fetchone()[0]
        deleted = self.con.execute("SELECT COUNT(*) FROM memories WHERE deleted=1").fetchone()[0]
        by_kind = {
            row[0]: row[1]
            for row in self.con.execute(
                "SELECT kind, COUNT(*) FROM memories WHERE deleted=0"
                " AND superseded_by IS NULL GROUP BY kind ORDER BY 2 DESC"
            )
        }
        by_ns = {
            row[0]: row[1]
            for row in self.con.execute(
                "SELECT namespace, COUNT(*) FROM memories WHERE deleted=0"
                " AND superseded_by IS NULL GROUP BY namespace ORDER BY 2 DESC LIMIT 12"
            )
        }
        events = self.con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "total_rows": total,
            "current": current,
            "superseded": superseded,
            "soft_deleted": deleted,
            "kinds": by_kind,
            "namespaces": by_ns,
            "events": events,
            "db_bytes": db_mod.db_size_bytes(),
            "embedder": self.embedder.name if self.embedder.available() else "none",
            "namespace": self.namespace,
            "schema_version": db_mod.SCHEMA_VERSION,
        }

    def maintenance(self, hard_delete_after_days: float | None = None, keep_events: int | None = None) -> dict:
        cfg = self.config
        ttl = hard_delete_after_days
        if ttl is None:
            ttl = float(cfg["maintenance"].get("soft_delete_ttl_days", 30))
        keep = int(keep_events or cfg["maintenance"].get("keep_events", 5000))
        cutoff = time.time() - ttl * 86400

        purged = self.con.execute(
            "SELECT id FROM memories WHERE deleted=1 AND updated_at < ?", (cutoff,)
        ).fetchall()
        ids = [int(row[0]) for row in purged]
        if ids:
            placeholders = ",".join("?" * len(ids))
            with self._tx():
                self.con.execute(
                    f"DELETE FROM links WHERE src IN ({placeholders}) OR dst IN ({placeholders})",
                    (*ids, *ids),
                )
                self.con.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
        self.con.execute(
            "DELETE FROM events WHERE id <= (SELECT MAX(id) - ? FROM events)", (keep,)
        )
        self.con.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')")
        self._invalidate_vectors()
        return {"purged": len(ids), "events_kept": keep}

    def reembed(self) -> int:
        if not self.embedder.available():
            raise RuntimeError("no embedder available")
        rows = self.con.execute("SELECT id, title, content, tags FROM memories").fetchall()
        processed = 0
        batch_size = 64
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [
                f"{row['title']}\n{row['content']}\n{' '.join(_safe_tags(row['tags']))}"
                for row in batch
            ]
            matrix = self.embedder.embed_many(texts)
            if matrix is None:
                break
            with self._tx():
                for offset, row in enumerate(batch):
                    self.con.execute(
                        "UPDATE memories SET embedding=?, embedding_model=? WHERE id=?",
                        (
                            np.asarray(matrix[offset], dtype=np.float32).tobytes(),
                            self.embedder.name,
                            int(row["id"]),
                        ),
                    )
            processed += len(batch)
        self._invalidate_vectors()
        return processed

    def export_json(self, path: str | Path | None = None) -> dict:
        memories = self.con.execute("SELECT * FROM memories").fetchall()
        links = self.con.execute("SELECT * FROM links").fetchall()
        payload = {
            "format": "supermemory-export",
            "version": 1,
            "exported_at": time.time(),
            "memories": [
                {k: row[k] for k in row.keys() if k not in ("embedding", "search_text")}
                for row in memories
            ],
            "links": [{k: row[k] for k in row.keys()} for row in links],
        }
        if path:
            Path(path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return payload

    def import_json(self, payload: dict) -> int:
        """Insert memories that are not already present (matched by id).

        Search text and embeddings are rebuilt locally rather than trusted from
        the file, so an export stays small and an import is always consistent
        with the *current* tokenizer and embedder.
        """
        rows = payload.get("memories") or []
        inserted_rows: list[dict] = []
        with self._tx():
            for row in rows:
                exists = self.con.execute(
                    "SELECT 1 FROM memories WHERE id=?", (row.get("id"),)
                ).fetchone()
                if exists:
                    continue
                columns = [
                    c for c in row.keys()
                    if c not in ("embedding", "embedding_model", "search_text")
                ]
                values = [row[c] for c in columns]
                self.con.execute(
                    f"INSERT INTO memories({','.join(columns)})"
                    f" VALUES({','.join('?' * len(columns))})",
                    values,
                )
                tags = _safe_tags(row.get("tags") or "[]")
                search_text = index_blob(
                    row.get("title") or "", row.get("title") or "",
                    row.get("summary") or "", row.get("content") or "",
                    " ".join(tags), " ".join(tags),
                    row.get("kind") or "", row.get("kind") or "",
                )
                self.con.execute(
                    "UPDATE memories SET search_text=? WHERE id=?", (search_text, row["id"])
                )
                db_mod.fts_upsert(self.con, int(row["id"]), search_text)
                inserted_rows.append(row)

        if inserted_rows and self.embedder.available():
            texts = [
                f"{row.get('title') or ''}\n{row.get('content') or ''}\n"
                f"{' '.join(_safe_tags(row.get('tags') or '[]'))}"
                for row in inserted_rows
            ]
            matrix = self.embedder.embed_many(texts)
            if matrix is not None:
                with self._tx():
                    for offset, row in enumerate(inserted_rows):
                        self.con.execute(
                            "UPDATE memories SET embedding=?, embedding_model=? WHERE id=?",
                            (
                                np.asarray(matrix[offset], dtype=np.float32).tobytes(),
                                self.embedder.name,
                                int(row["id"]),
                            ),
                        )
        for link in payload.get("links") or []:
            try:
                self.con.execute(
                    "INSERT OR IGNORE INTO links(src, dst, relation, weight, created_at)"
                    " VALUES(?,?,?,?,?)",
                    (
                        int(link["src"]), int(link["dst"]),
                        link.get("relation") or "related",
                        float(link.get("weight") or 1.0),
                        float(link.get("created_at") or time.time()),
                    ),
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._invalidate_vectors()
        return len(inserted_rows)

    # -- internals ----------------------------------------------------------

    def _scope(self, namespace: str | None) -> list[str] | None:
        """Namespaces visible from this store: explicit, or project + shared."""
        if namespace is not None:
            return config_mod.resolve_namespaces(namespace, self.config)
        if os.environ.get("SUPERMEMORY_NAMESPACE", "").strip() == "*":
            return None
        shared = config_mod.sanitize_namespace(
            self.config["store"].get("shared_namespace", "global")
        )
        if self.config["store"].get("include_shared", True) and self.namespace != shared:
            return [self.namespace, shared]
        return [self.namespace]

    def _event(self, op: str, memory_id: int | None, detail: str) -> None:
        try:
            self.con.execute(
                "INSERT INTO events(ts, op, memory_id, detail) VALUES(?,?,?,?)",
                (time.time(), op, memory_id, detail[:500]),
            )
        except sqlite3.Error:
            pass

    @staticmethod
    def _invalidate_vectors() -> None:
        _VEC_CACHE["key"] = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _namespace_sql(namespaces: list[str] | None, alias: str = "m") -> tuple[str, tuple]:
    if namespaces is None or not namespaces:
        return "", ()
    placeholders = ",".join("?" * len(namespaces))
    prefix = f"{alias}." if alias else ""
    return f" AND {prefix}namespace IN ({placeholders})", tuple(namespaces)


def _clean_tags(tags: Iterable[str] | None) -> list[str]:
    if not tags:
        return []
    out: list[str] = []
    for tag in tags:
        text = str(tag).strip().lower()[:48]
        if text and text not in out:
            out.append(text)
        if len(out) >= 16:
            break
    return out


def _safe_tags(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
        return [str(t) for t in value] if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _clamp(value: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _derive_title(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    first_line = re.sub(r"^[#>*\-\s]+", "", first_line).strip()
    if not first_line:
        first_line = summarize(content, 60)
    return first_line[:120] or "untitled"
