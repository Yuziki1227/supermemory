<div align="center">

# SuperMemory

**Local, attention-weighted long-term memory for Claude Code and Codex.**

[![CI](https://github.com/Yuziki1227/supermemory/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuziki1227/supermemory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![MCP](https://img.shields.io/badge/MCP-stdio-6E56CF)
![Storage](https://img.shields.io/badge/storage-SQLite%20FTS5-003B57)
![Local first](https://img.shields.io/badge/cloud-none%20by%20default-2E7D32)

English · [简体中文](README.zh-CN.md)

</div>

SuperMemory gives coding agents a memory that survives session boundaries and
context compaction. Everything lives in one SQLite file. Recall is a hybrid
search — BM25, embeddings, recency, importance, reinforcement and pinning —
combined through a softmax "attention" layer, diversified with MMR, and cut to
an explicit token budget. Claude Code and Codex connect over MCP and read and
write the same database.

```text
conversation ──► memory_remember ──► SQLite (chunks · FTS5 · vectors · version chain)
                                            │
new session ◄── memory_briefing ◄── attention-weighted recall ◄── memory_recall
```

---

## Why

Coding agents start every session cold. `CLAUDE.md` and `AGENTS.md` carry
static project conventions, but not the decisions, constraints and half-finished
task state that accumulate while working. Context compaction then drops even
that. The result is the same questions, the same rediscovered gotchas, and the
same design debates every few hours.

SuperMemory closes that loop:

- write a durable fact **the moment you learn it** with `memory_remember`;
- get a budgeted hand-off document back at the **next SessionStart**;
- retrieve specifics on demand with `memory_recall`, without loading the whole
  store into the context window.

## Features

- **Local-first.** One SQLite file under `~/.supermemory/`. No server, account,
  API key or network call in the default configuration.
- **Two agents, one brain.** The MCP stdio server is shared by Claude Code and
  Codex; any other MCP client can use it too.
- **Retrieval that behaves.** Exact dedupe, near-duplicate hints, immutable
  version chains, soft delete, namespace isolation and a relevance gate.
- **Context-aware ranking.** Six features, a softmax attention distribution and
  MMR diversity selection. Every result is explainable and tunable.
- **Measurable retrieval.** A labelled benchmark scores Recall@k, MRR and nDCG
  and gates CI; `--explain` shows exactly why a result won.
- **Chinese-first tokenization.** SQLite's `unicode61` tokenizer cannot search
  CJK text. SuperMemory indexes CJK unigrams and bigrams itself and splits
  `camelCase` / `snake_case` identifiers.
- **Fail-open integration.** Hooks never block a session: any error goes to
  stderr and the agent keeps working.

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | the test suite passes on 3.14; avoid 3.11+ only syntax |
| `numpy` ≥ 1.24 | the only third-party runtime dependency |
| SQLite with FTS5 | bundled with official CPython builds |
| Claude Code and/or Codex | optional — the CLI works standalone |

Check everything at once:

```bash
python -m supermemory doctor
```

## Quick start

```bash
git clone https://github.com/Yuziki1227/supermemory.git
cd supermemory
python install.py            # install, register MCP servers and hooks, self-check
```

Then **restart Claude Code / Codex**. The first session will not show a briefing
because the database is empty; memories appear in the briefing once the agent
has written some.

Other installer modes:

```bash
python install.py --dry-run   # show every file that would change
python install.py --dev       # use this checkout in place (no copy)
python install.py --uninstall # remove config entries, keep the database
python install.py --purge     # uninstall and delete ~/.supermemory
```

### What the installer changes

| Action | Location | Notes |
|---|---|---|
| Copies the package | `~/.supermemory/app/` | the checkout can be moved afterwards |
| Registers MCP — Claude Code | `~/.claude.json` → `mcpServers.supermemory` | user scope |
| Installs hooks — Claude Code | `~/.claude/settings.json` → `hooks` | SessionStart, PreCompact, SessionEnd |
| Registers MCP — Codex | `~/.codex/config.toml` → `[mcp_servers.supermemory]` | |
| Adds usage instructions | `~/.claude/CLAUDE.md` (marked block) | skip with `--no-instructions` |

Every modified file is backed up to `~/.supermemory/backups/`. All merges are
idempotent: running the installer twice does not duplicate entries.

### Manual MCP registration

If you prefer not to run the installer, register the server yourself.

Claude Code (`~/.claude.json`, or a project-scoped `.mcp.json`):

```json
{
  "mcpServers": {
    "supermemory": {
      "type": "stdio",
      "command": "python",
      "args": ["C:/Users/you/.supermemory/app/run.py", "mcp"],
      "env": { "PYTHONUTF8": "1" }
    }
  }
}
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.supermemory]
command = "python"
args = ["C:/Users/you/.supermemory/app/run.py", "mcp"]
env = { PYTHONUTF8 = "1" }
startup_timeout_sec = 30
```

> Windows: point `command` at `python.exe` and keep the `.cmd` launcher only for
> human CLI use. The installer writes the paths for you.

`pip install .` also works for the CLI (`supermemory ...`); MCP registration is
still manual. In that case use `python -m supermemory mcp` as the server command.

## How it works

### 1. Storage: chunks, FTS index, version chain

Long input is chunked before it reaches SQL, in this order of preference:
blank line → line → sentence → hard split. Defaults are 260 tokens per chunk
with 48 tokens of overlap, so every row is a self-contained, retrievable atom.

Each row keeps title, body, summary, tags, source, importance, confidence,
timestamps, embedding, content hash and version pointers.

**Updates never overwrite.** `memory_update` inserts a new row and marks the old
one `superseded_by` with `valid_to` set. Normal recall only sees current rows;
`include_superseded=true` shows the full history. "This changed" and "nobody
maintains this" never get mixed up.

**Deduplication has two levels:**

- exact repeats (same namespace + kind + content hash) return the existing id
  instead of writing a new row;
- near-duplicates (cosine ≥ 0.90) are **not merged automatically** — the result
  carries a hint instead. "Use X" and "do not use X" can be vector neighbours;
  silently merging them is the fastest way to corrupt a memory store.

### 2. Retrieval: an attention distribution

Every candidate is described by six features:

| Feature | Source | Default weight |
|---|---|---|
| `lexical` | FTS5 BM25, normalised within the candidate set | 0.36 |
| `vector` | cosine similarity between query and memory | 0.28 |
| `importance` | author-declared, damped by confidence | 0.13 |
| `recency` | exponential decay on last write/recall, 45-day half-life | 0.09 |
| `reinforcement` | `log(1 + times recalled)` — proven-useful memories win ties | 0.06 |
| `pinned` | explicit, permanent boost | 0.08 |

The weighted sum is `score`. Then:

```text
alpha_i = softmax(score_i / τ)      τ = 0.35
```

`alpha` is an attention distribution: a fixed budget of probability mass is
split across candidates according to relevance, with `τ` controlling how sharp
the split is. This is not a neural attention head — local inference is not the
point — but the same mathematical shape, fully explainable and free to compute.

Selection uses **MMR** (maximal marginal relevance): each pick discounts
candidates that are too similar to what was already selected, so the top-k is
not five rephrasings of one memory. The result is then cut to a token budget.

A hard gate sits in front of all of this: a candidate must either match the
lexical index or have cosine ≥ 0.15. Without it, "recent + important" alone
drags unrelated memories into the top-k — the simplest source of recall
poisoning.

### 3. Why CJK needs its own tokenizer

SQLite's `unicode61` treats a whole Chinese sentence as one token, so
`MATCH "检索"` never hits. SuperMemory uses one tokenizer for both indexing and
querying: CJK runs are indexed as unigrams plus adjacent bigrams
(`检索层` → `检 索 层 检索 索层`), and Latin identifiers are split on camelCase
and snake_case (`MemoryStore` → `memorystore memory store`). Bigrams preserve
phrases; BM25's IDF automatically down-weights single characters.

### 4. Embeddings

The default is an **offline hashed embedding**: words, identifiers and CJK
n-grams are feature-hashed with blake2b into 1024 dimensions, sublinearly
weighted, and L2-normalised. Nothing to download, no API key, deterministic
across processes — it works immediately after install.

It correlates with BM25 but adds character-level similarity (typos, morphology,
mixed Chinese/English). For true semantic vectors, switch to any
OpenAI-compatible `/v1/embeddings` endpoint:

```bash
export SUPERMEMORY_EMBED_API_KEY=sk-...
python -m supermemory config --set embedder.provider=openai
python -m supermemory config --set embedder.model=text-embedding-3-small
python -m supermemory reembed
```

If the remote endpoint is unreachable, the process **falls back to local
embeddings** with a warning on stderr. A memory service should not refuse to
start because a key expired.

### 5. Namespaces

By default the namespace is the current project's directory name
(`SuperMemory/` → `supermemory`). Recall searches that namespace plus `global`;
cross-project knowledge should be written with `namespace="global"`.
`namespace="*"` searches everything.

Resolution order for the project directory:

1. `SUPERMEMORY_NAMESPACE` (explicit namespace, `*` = all)
2. `SUPERMEMORY_PROJECT_CWD` (set by the hooks from the hook payload)
3. `CLAUDE_PROJECT_DIR` (injected by Claude Code)
4. the process working directory

### 6. Session hand-off

The **SessionStart hook** injects a briefing: pinned rules, open tasks, key
decisions, preferences and gotchas, plus recent updates. It fires for every
session source, **including `source=compact`** — the only supported moment to
put text back in front of the model after a context window collapses.

The **PreCompact hook** cannot inject context (Claude Code rejects its
`hookSpecificOutput`), so it only prints a plain-text reminder to stdout.
Persisting state is still the agent's job; the SessionStart hook is the
guaranteed recovery path.

The **SessionEnd hook** runs housekeeping: purge expired soft-deletes, trim the
event log.

Retrieval alone is not enough — memory has to be pushed back at the right
moment.

## MCP tools

| Tool | Purpose | Notable arguments |
|---|---|---|
| `memory_briefing` | Call once at session start: project state + relevant memories, budgeted | `query`, `max_tokens`, `namespace` |
| `memory_remember` | Store a fact/decision/preference/task/gotcha; auto-chunk, dedupe, hint | `content`, `kind`, `tags`, `importance`, `pinned`, `ttl_days` |
| `memory_recall` | Hybrid retrieval; compact entries with id, title, summary, score | `query`, `limit`, `kinds`, `tags`, `max_tokens`, `explain`, `format` |
| `memory_get` | Fetch full text by id | `ids` |
| `memory_update` | Revise; writes a new version and keeps history | `id`, `content`, `pinned`, ... |
| `memory_forget` | Soft-delete by default; `hard=true` deletes permanently | `ids`, `hard` |
| `memory_link` | Typed relation between memories (`refines`, `contradicts`, `depends-on`, ...) | `src`, `dst`, `relation`, `weight` |
| `memory_stats` | Counts, kind distribution, database size, embedder | `format` |
| `memory_export` | Export the whole database to JSON | `path` |

Tool descriptions encode *when* to use them, so the model does not need extra
prompting. `initialize` also returns server instructions covering when to
write, when to read and how to revise.

## CLI

`python -m supermemory ...` and `supermemory ...` are equivalent; the latter
requires `~/.supermemory/bin` on `PATH` (the installer creates it).

```bash
# setup and health
supermemory doctor
supermemory init
supermemory stats

# write and read
supermemory remember "Content" -t "Title" -k decision --tag architecture -i 0.9
supermemory recall "how does retrieval work"
supermemory recall "how does retrieval work" --explain
supermemory get M12
supermemory list --kind decision
supermemory briefing --query "implement MCP"
supermemory bench bench/cases.jsonl       # retrieval regression suite

# revise and maintain
supermemory update M12 --content "Corrected content"
supermemory forget M12                     # soft delete
supermemory link M12 M7 --relation refines
supermemory maintenance

# portability and configuration
supermemory export backup.json
supermemory import backup.json
supermemory reembed --provider openai
supermemory config
supermemory config --set ranking.temperature=0.5

# integration
supermemory mcp                            # run the stdio MCP server
supermemory hook session-start             # hook entry point (reads JSON on stdin)
```

## Retrieval quality

Two tools keep ranking changes honest.

### Why did this result win?

`supermemory recall "query" --explain` appends the feature breakdown to each
hit:

```text
M3 [decision 0.63] retrieval layer
    Retrieval uses BM25 and vector scores.
    ↳ lex 1.00 · vec 0.42 · imp 0.46 · rec 1.00 · rein 0.00 · pin 0 · score 0.626 · attn 1.0000
```

The same switch exists over MCP: `memory_recall` with `"explain": true`.

### Benchmark

`bench/cases.jsonl` is a self-contained corpus plus labelled queries. Each run
seeds a temporary database and reports Recall@k, MRR, nDCG@k, negative-case
behaviour and token cost:

```bash
python -m supermemory bench bench/cases.jsonl
python -m supermemory bench bench/cases.jsonl --json
python -m supermemory bench bench/cases.jsonl --fail-under-recall 0.85 --fail-under-mrr 0.75
```

CI runs the suite, so a ranking change that breaks retrieval fails the build.
Add a case for every retrieval bug you fix; see [bench/README.md](bench/README.md)
for the file format. Benchmarks use the built-in defaults — pass
`--config ~/.supermemory/config.json` to measure a tuned configuration.

## Configuration

`~/.supermemory/config.json` is optional; unset values use defaults. The most
useful knobs:

```json
{
  "ranking": {
    "weights": { "lexical": 0.36, "vector": 0.28, "importance": 0.13,
                 "recency": 0.09, "reinforcement": 0.06, "pinned": 0.08 },
    "temperature": 0.35,
    "half_life_days": 45,
    "mmr_lambda": 0.22
  },
  "recall": { "default_limit": 8, "max_tokens": 1400, "vector_floor": 0.15 },
  "briefing": { "max_tokens": 900 },
  "store": { "chunk_max_tokens": 260, "namespace_mode": "cwd" }
}
```

- `temperature` lower → attention concentrates on the top hit; higher → flatter
  distribution.
- `mmr_lambda` higher → stronger diversity penalty; `0` disables MMR.
- `half_life_days` lower → recent memories dominate.
- `namespace_mode`: `cwd` (default), `fixed` (use `fixed_namespace`), `none`
  (everything goes to `global`).

Environment variables:

| Variable | Effect |
|---|---|
| `SUPERMEMORY_HOME` | Data directory (default `~/.supermemory`) |
| `SUPERMEMORY_NAMESPACE` | Force a namespace; `*` searches all |
| `SUPERMEMORY_PROJECT_CWD` | Project directory used to derive the namespace |
| `SUPERMEMORY_EMBED_PROVIDER` | `local`, `openai`, or `none` |
| `SUPERMEMORY_EMBED_MODEL` / `_BASE_URL` / `_DIM` | Remote embedder settings |
| `SUPERMEMORY_EMBED_API_KEY` | API key for the remote embedder |
| `CLAUDE_PROJECT_DIR` | Read automatically when deriving the namespace |

## Repository layout

```text
supermemory/
  config.py        defaults, paths, namespace resolution
  text.py          tokenizer, token estimate, summaries, chunking
  embed.py         local hashed embeddings / OpenAI-compatible embeddings
  db.py            schema, FTS5, connection setup
  ranking.py       attention scoring + MMR
  core.py          MemoryStore: writes, recall, versions, briefing, export
  render.py        token-aware output rendering
  mcp_server.py    dependency-free MCP stdio server
  cli.py           command line interface
  hooks.py         Claude Code hook entry points
  bench.py         labelled retrieval benchmark and metrics
install.py         installer
run.py             single launcher for MCP / hooks / CLI
bench/             benchmark cases and format documentation
tests/             32 tests, including a real MCP subprocess handshake
```

## Design decisions

- **No automatic summaries.** A summary is the first sentence, not an LLM
  rewrite. That saves an inference call and prevents summary drift — memory
  saying one thing and its summary another. When semantic compression is
  needed, the agent writes its own `kind=decision` memory.
- **No automatic merging.** See above: negation semantics get eaten.
- **No automatic decay deletion.** Recency only affects ranking; it is never
  written back. Memories do not disappear because you have not used them for a
  while.
- **Hooks fail open.** Every exception is printed to stderr and returns 0. A
  broken memory system must not break normal coding.
- **Hand-rolled MCP instead of an SDK.** The server depends only on the CPython
  standard library plus numpy, so it runs on any machine with Python.
- **Bilingual docs, one codebase.** User-facing CLI text is Chinese today;
  every feature is documented in both README languages.

## FAQ

**No briefing after restart?**
The database is empty (nothing was written yet), or the hooks did not take
effect. Check `python -m supermemory stats`; test the hook manually:

```bash
echo '{"source":"startup","cwd":"/path/to/project"}' | python -m supermemory hook session-start
```

**The namespace looks wrong.**
Namespaces are derived from the project directory name. A stdio MCP process
does not necessarily start in the project root, so SuperMemory prefers
`SUPERMEMORY_PROJECT_CWD` → `CLAUDE_PROJECT_DIR` → the process cwd. If it is
still wrong, set `SUPERMEMORY_NAMESPACE` explicitly or add it to the MCP
server's `env`.

**Chinese recall is weak.**
Run `doctor` and confirm FTS5 ✓. Then check for one-character queries — the
query tokenizer drops single CJK characters when bigrams are available.

**Can two agents write at the same time?**
Yes. The database uses WAL, a 30-second busy timeout, and `BEGIN IMMEDIATE` for
writes. The MCP server, hooks and CLI all share one file safely.

**How do I move to another machine?**
Copy `~/.supermemory/memory.db` together with its `-wal` file, or use
`supermemory export` on the old machine and `supermemory import` on the new one.

**The database is getting large.**
`supermemory maintenance` purges soft-deleted rows past their retention window
and compacts the FTS index.

**Is any data sent over the network?**
No, not in the default configuration. If you enable the OpenAI-compatible
embedder, memory text is sent to the endpoint you configure.

## Development

```bash
python -m unittest discover -s tests -v
python -m supermemory bench bench/cases.jsonl
python install.py --dev       # run the checkout in place
```

No third-party test framework, no build step. Tests isolate themselves with a
temporary `SUPERMEMORY_HOME` and never touch your real database. CI runs the
unit suite and the retrieval benchmark on Linux (Python 3.10–3.14) and Windows.

See [AGENTS.md](AGENTS.md) for the architecture, invariants and contribution
workflow, and [CONTRIBUTING.md](CONTRIBUTING.md) for the short version.

## License

[MIT](LICENSE) © 2026 Yuziki1227
