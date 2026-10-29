"""SQLite storage: schema, connections, and FTS5 bookkeeping.

Design notes
------------
* One row per *memory chunk*. Long text is chunked before it ever reaches SQL,
  so every row is a self-contained, retrievable atom.
* ``memories_fts`` is a plain FTS5 table (not external-content) holding a
  pre-tokenised blob. Duplicating the tokens costs little and makes CJK search
  actually work; the alternative is a tokenizer extension.
* History is never destroyed by an update: a revision is a new row and the old
  row gets ``superseded_by`` + ``valid_to``. Recalls skip superseded rows unless
  asked, so the agent sees current truth while the audit trail survives.
* WAL + a generous busy timeout means the MCP server and short-lived CLI hook
  processes can share the file without coordination.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config as config_mod

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace       TEXT    NOT NULL DEFAULT 'default',
    kind            TEXT    NOT NULL DEFAULT 'fact',
    title           TEXT    NOT NULL DEFAULT '',
    content         TEXT    NOT NULL,
    summary         TEXT    NOT NULL DEFAULT '',
    tags            TEXT    NOT NULL DEFAULT '[]',
    source          TEXT    NOT NULL DEFAULT '',
    source_ref      TEXT    NOT NULL DEFAULT '',
    importance      REAL    NOT NULL DEFAULT 0.5,
    confidence      REAL    NOT NULL DEFAULT 0.8,
    pinned          INTEGER NOT NULL DEFAULT 0,
    deleted         INTEGER NOT NULL DEFAULT 0,
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    accessed_at     REAL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    valid_from      REAL,
    valid_to        REAL,
    superseded_by   INTEGER,
    content_hash    TEXT    NOT NULL,
    tokens          INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB,
    embedding_model TEXT    NOT NULL DEFAULT '',
    search_text     TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_memories_ns      ON memories(namespace, deleted, superseded_by);
CREATE INDEX IF NOT EXISTS idx_memories_hash    ON memories(namespace, content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_kind    ON memories(namespace, kind, updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_pinned  ON memories(pinned, updated_at);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    search_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS links (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src        INTEGER NOT NULL,
    dst        INTEGER NOT NULL,
    relation   TEXT    NOT NULL DEFAULT 'related',
    weight     REAL    NOT NULL DEFAULT 1.0,
    created_at REAL    NOT NULL,
    UNIQUE(src, dst, relation)
);

CREATE INDEX IF NOT EXISTS idx_links_src ON links(src);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);

CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    op        TEXT NOT NULL,
    memory_id INTEGER,
    detail    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open (and initialise) the memory database."""
    target = Path(path) if path else config_mod.db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target), timeout=30.0, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-16000")
    init_schema(con)
    return con


def init_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        con.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
    con.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES('created_at', ?)",
        (str(__import__("time").time()),),
    )


# ---------------------------------------------------------------- FTS helpers


def fts_upsert(con: sqlite3.Connection, memory_id: int, search_text: str) -> None:
    # FTS5 is a virtual table: no UPSERT support, so replace explicitly.
    con.execute("DELETE FROM memories_fts WHERE rowid=?", (memory_id,))
    con.execute(
        "INSERT INTO memories_fts(rowid, search_text) VALUES(?, ?)",
        (memory_id, search_text),
    )


def fts_delete(con: sqlite3.Connection, memory_id: int) -> None:
    con.execute("DELETE FROM memories_fts WHERE rowid=?", (memory_id,))


def fts_clear(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM memories_fts")


def db_size_bytes(path: Path | str | None = None) -> int:
    target = Path(path) if path else config_mod.db_path()
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(target) + suffix)
        try:
            total += candidate.stat().st_size
        except OSError:
            pass
    return total
