# AGENTS.md

Guidance for AI coding agents working in this repository. Read this before
changing code. `README.md` explains the product; this file explains how the
repository is built and which behaviours must not regress.

## What this is

SuperMemory is a local long-term memory server for coding agents. It stores
memory chunks in one SQLite database (WAL + FTS5), ranks recall with a hybrid
attention score, and exposes the store over a hand-rolled MCP stdio server.
Claude Code and Codex are the primary clients; the CLI works standalone.

- Language: Python ≥ 3.10 (the suite passes on 3.14; avoid 3.11+ only syntax).
- Runtime dependencies: **numpy only**. Everything else is the standard library.
- No build step, no third-party test framework, no network in the default path.
- User-facing CLI and hook text is Chinese today; code comments, docstrings and
  agent-facing docs are English. Both READMEs must stay in sync.

## First commands

```bash
python -m unittest discover -s tests -v    # 21 tests, includes a real MCP subprocess
python -m supermemory doctor               # environment + FTS5 + numpy + DB health
python install.py --dry-run                # inspect installer changes without writing
python run.py mcp                          # run the MCP server by hand (JSON-RPC on stdio)
```

Use `SUPERMEMORY_HOME=/tmp/sm-test` (or any temp directory) when experimenting
by hand, so you never touch a real memory database:

```bash
SUPERMEMORY_HOME=/tmp/sm-test python -m supermemory remember "hello" -t test
```

## Repository map

| Path | Responsibility |
|---|---|
| `supermemory/config.py` | `DEFAULT_CONFIG`, paths under `SUPERMEMORY_HOME`, namespace resolution and sanitising |
| `supermemory/text.py` | CJK/Latin tokenizer, `index_blob`, `query_tokens`, token estimate, summary, chunking |
| `supermemory/embed.py` | Local blake2b hashed embedder, OpenAI-compatible embedder, automatic fallback |
| `supermemory/db.py` | Schema (`SCHEMA_VERSION = 1`), connection PRAGMAs, FTS5 helpers |
| `supermemory/ranking.py` | Six-feature scoring, softmax attention, MMR selection |
| `supermemory/core.py` | `MemoryStore`: write path, versioning, recall, briefing, stats, export/import, maintenance |
| `supermemory/render.py` | Token-aware rendering of recall results, records, briefing, stats |
| `supermemory/mcp_server.py` | JSON-RPC 2.0 over stdio, tool schemas, handlers, `initialize` instructions |
| `supermemory/cli.py` | `argparse` CLI; `mcp` and `hook` entry points |
| `supermemory/hooks.py` | `SessionStart` / `PreCompact` / `SessionEnd` handlers for Claude Code |
| `install.py` | Idempotent installer: copy app, register MCP, install hooks, instructions block, doctor |
| `run.py` | Single launcher used by MCP clients and hooks; fixes `sys.path` |
| `tests/test_core.py` | Text, store, versioning, dedupe, namespace, budget, briefing, export tests |
| `tests/test_mcp.py` | Spawns `run.py mcp` and speaks real JSON-RPC to it |

## Non-negotiable invariants

These are load-bearing. If a change needs to break one, stop and surface the
trade-off instead of silently changing it.

1. **MCP stdout is the protocol channel.** In `mcp_server.py`, only `_write()`
   may write to stdout. Diagnostics go to stderr. A stray `print()` corrupts the
   stream and breaks every client.
2. **Hooks fail open.** `hooks.run()` catches every exception, prints the
   traceback to stderr and returns `0`. `SessionStart` emits exactly one JSON
   object; `PreCompact` emits plain text only (Claude Code rejects its
   `hookSpecificOutput`); `SessionEnd` stays silent.
3. **No new runtime dependencies.** `numpy` is the ceiling. Use `urllib` rather
   than `requests`; use `sqlite3` rather than an ORM. Development-only tooling is
   acceptable only if it is optional.
4. **Tokenizers stay symmetric.** Indexing uses `text.index_blob()`; querying
   uses `text.query_tokens()` + `text.fts_expression()`. Changing one without
   the other silently zeroes recall.
5. **Updates never overwrite.** `MemoryStore.update()` inserts a new row and
   marks the old one `superseded_by` / `valid_to`. Do not add an in-place
   content update.
6. **Only exact duplicates are suppressed.** Never auto-merge near-duplicates.
   The system records a hint at cosine ≥ 0.90 and leaves the decision to the
   agent — negation semantics are too easy to destroy.
7. **Namespace resolution has a fixed precedence.** Explicit
   `SUPERMEMORY_NAMESPACE` → configured mode (`fixed` / `none`) → project
   directory from `SUPERMEMORY_PROJECT_CWD` → `CLAUDE_PROJECT_DIR` → process
   cwd. Always pass values through `config.sanitize_namespace()`.
8. **Writes use `MemoryStore._tx()`** (`BEGIN IMMEDIATE`). Do not nest `_tx()`
   calls or switch to autocommit writes; MCP server, hooks and CLI share one
   file over WAL.
9. **Token budgets are contracts.** `recall` and `briefing` must respect
   `max_tokens`. If you change `render.py` formatting, re-check the cost estimate
   in `MemoryStore.recall` (summary tokens + fixed overhead).
10. **The installer is idempotent and backs up.** Every config merge must be
    repeatable without duplicates, and every modified file must be copied to
    `~/.supermemory/backups/` first. On Windows, hooks call the quoted `.cmd`
    launcher; do not replace it with a bare `python` path.
11. **Tests never touch a real database.** Set `SUPERMEMORY_HOME` and
    `SUPERMEMORY_NAMESPACE` before creating a store. `tests/test_core.py` uses
    `IsolatedStore` for this; `tests/test_mcp.py` passes the environment to the
    subprocess.
12. **Python 3.10 compatibility.** Keep `from __future__ import annotations` and
    avoid 3.11+ only features (`typing.Self`, `tomllib`, `datetime.UTC`, exception
    groups). `X | Y` in annotations is fine.

## Data model

Tables live in `db.py`:

- `memories` — one row per chunk; `id` is exposed as `M<id>` (`Memory.label`);
  `parse_id()` accepts `"M12"`, `"12"` or `12`.
- `memories_fts` — standalone FTS5 table (`rowid` = memory id) storing the
  pre-tokenised `search_text` blob. Not external-content, so it must be kept in
  sync explicitly with `fts_upsert()` / `fts_delete()`.
- `links` — typed edges (`revises`, `related`, `refines`, `contradicts`, ...).
- `events` — append-only audit log, trimmed by `maintenance()`.
- `meta` — `schema_version`, `created_at`.

Soft delete is `deleted = 1` plus removal from FTS; `maintenance()` hard-deletes
after `maintenance.soft_delete_ttl_days`. Vectors are stored as `float32`
`BLOB` with `embedding_model`; changing the embedder requires `reembed` (the
CLI does this).

## How to extend

### Add an MCP tool

1. Add the schema to `TOOLS` in `mcp_server.py` (description must say *when* to
   use it; models rely on this).
2. Add a `_tool_*` handler and register it in `HANDLERS`.
3. Render output through `render.py`; keep results compact and token-aware.
4. Add CLI parity in `cli.py` when the operation is useful to humans.
5. Add a test in `tests/test_mcp.py`: at minimum extend `test_tools_list`, and
   exercise the call through the real subprocess.
6. Update the tool table in **both** READMEs.

### Add a CLI command

1. Add `cmd_*` in `cli.py`, then `add("<name>", help=...)` in `build_parser()`.
2. Return an `int` exit code; raise `ValueError` / `KeyError` for user errors
   (the top-level handler prints and returns 1).
3. Document it in both READMEs and in `supermemory/cli.py` usage examples.

### Add a config key

1. Add the default to `DEFAULT_CONFIG` in `config.py` (comment the units/semantics).
2. Read it through `self.config` (never hardcode), and keep the read tolerant of
   missing keys for forward compatibility.
3. Document it in both READMEs' configuration sections.
4. If it changes ranking or recall behaviour, add a regression test.

### Schema change

`SCHEMA_VERSION` is currently `1`. A schema change needs:

- an idempotent migration in `init_schema()` (old databases must open cleanly);
- `ALTER TABLE ... ADD COLUMN` with a default rather than a destructive rewrite;
- a test that opens a database created by the previous version;
- a note in the release/PR description.

## Style

- Type hints on public functions; `from __future__ import annotations` at the top.
- Prefer small, pure helpers in `text.py`, `ranking.py`, `render.py`; keep SQL
  in `core.py` / `db.py`.
- Parameterised SQL only. Build namespace filters with `_namespace_sql()`.
- Errors: `ValueError` for invalid input, `KeyError` for missing ids. The MCP
  layer converts both into tool errors (`isError: true`) without crashing.
- Comments explain *why*, not *what*; match the surrounding density.
- Keep lines readable (~100 columns); no formatter is enforced.
- Never log secrets. Never add telemetry.

## Testing expectations

- Run the full suite before finishing: `python -m unittest discover -s tests -v`.
- New behaviour needs a test. Bugs get a regression test that fails before the
  fix.
- Ranking changes: assert on ordering or relevance gate behaviour, not exact
  float scores (they depend on the candidate set).
- Chunking changes: assert content preservation and token ceilings, as in
  `test_chunking_respects_budget_and_keeps_content`.
- MCP changes: test through the subprocess (`McpSession`), not by calling
  handlers directly.
- Windows specifics (quoting, launcher, UTF-8) need manual reasoning; the CI
  target may be Linux only.

## Documentation rules

- `README.md` (English) and `README.zh-CN.md` (Chinese) are mirrors. A feature,
  CLI flag, config key or tool added to one must be added to the other.
- Keep the ASCII architecture diagram and feature tables accurate.
- `AGENTS.md` is agent-facing and may go deeper on internals than the READMEs.
- Update `pyproject.toml` when the version, entry points or classifiers change.

## Release checklist

1. `python -m unittest discover -s tests -v` passes.
2. `python install.py --dry-run` shows only intended changes.
3. `python -m supermemory doctor` passes on a clean `SUPERMEMORY_HOME`.
4. Both READMEs updated; `AGENTS.md` still matches the code.
5. Bump `__version__` in `supermemory/__init__.py` and `version` in
   `pyproject.toml` together.
6. `python -m build` (optional) and a smoke install in a clean virtualenv.
